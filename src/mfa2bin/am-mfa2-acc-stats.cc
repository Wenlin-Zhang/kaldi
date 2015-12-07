// mfa2bin/am-mfa2-acc-stats.cc

// Copyright 2015   Wen-Lin Zhang

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "mfa2/am-mfa2.h"
#include "mfa/estimate-am-mfa-types.h"
#include "hmm/transition-model.h"
#include "mfa2/estimate-am-mfa2.h"
#include "hmm/posterior.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Accumulate stats for Am-Mfa2 training.\n"
        "Usage: am-mfa2-acc-stats [options] <model-in> <feature-rspecifier> <posteriors-rspecifier> <stats-out>\n"
        "e.g.: am-mfa2-acc-stats 1.mdl 1.ali scp:train.scp 'ark:ali-to-post 1.ali ark:-|' 1.acc\n";

    ParseOptions po(usage);
    bool binary = true;
    std::string update_flags_str = "ywS";
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which AmMfa parameters to accumulate "
                "stats for: subset of ywS.");
    AmMfa2GselectDirectConfig gsDirectConfig;
    gsDirectConfig.Register(&po);
    po.Read(argc, argv);

    kaldi::AmMfaUpdateFlagsType acc_flags = StringToAmMfaUpdateFlags(update_flags_str);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        accs_wxfilename = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    // Initialize the readers before the model, as the model can
    // be large, and we don't want to call fork() after reading it if
    // virtual memory may be low.
    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    AmMfa2 am_mfa2;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa2.Read(ki.Stream(), binary);
      am_mfa2.SetPreSelGaussDirect(&gsDirectConfig);
      am_mfa2.PreCompute();
    }

    Vector<double> transition_accs;
    trans_model.InitStats(&transition_accs);
    MleAmMfa2Accs ammfa2_accs;
    ammfa2_accs.ResizeAccumulators(am_mfa2);

    double tot_like = 0.0;
    double tot_t = 0;

    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!posteriors_reader.HasKey(utt)) {
        num_no_posterior++;
      } else {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(utt);

        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (posterior.size()) <<
              " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        BaseFloat tot_like_this_file = 0.0, tot_weight = 0.0;

        for (size_t i = 0; i < posterior.size(); i++) {

          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;
            if (acc_flags & kaldi::kAmMfaTransitions)
              trans_model.Accumulate(weight, tid, &transition_accs);
            tot_like_this_file += ammfa2_accs.Accumulate(am_mfa2, mat.Row(i), pdf_id,
                                                       weight, acc_flags)
                                                       * weight;
            tot_weight += weight;
          }
        }

        KALDI_VLOG(2) << "Average like for this file is "
                      << (tot_like_this_file/tot_weight) << " over "
                      << tot_weight <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += tot_weight;
        if (num_done % 50 == 0) {
          KALDI_LOG << "Processed " << num_done << " utterances; for utterance "
                    << utt << " avg. like is "
                    << (tot_like_this_file/tot_weight)
                    << " over " << tot_weight <<" frames.";
        }
      }
    }
    KALDI_LOG << "Overall like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    {
      Output ko(accs_wxfilename, binary);
      // TODO(arnab): Ideally, we shouldn't be writing transition accs if not
      // asked for, but that will complicate reading later. To be fixed?
      transition_accs.Write(ko.Stream(), binary);
      ammfa2_accs.Write(ko.Stream(), binary);
    }
    KALDI_LOG << "Written accs.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


