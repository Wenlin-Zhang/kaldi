// mfa2bin/am-mfa2-post-to-gpost.cc

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
#include "hmm/posterior.h"
#include "hmm/transition-model.h"
#include "mfa2/estimate-am-mfa2.h"




int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Convert posteriors to Gaussian-level posteriors for AmMfa2 training.\n"
        "Usage: am-mfa2-post-to-gpost [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <gpost-wspecifier>\n"
        "e.g.: am-mfa2-post-to-gpost 1.mdl 1.ali scp:train.scp 'ark:ali-to-post ark:1.ali ark:-|' ark:-";

    ParseOptions po(usage);
    BaseFloat rand_prune = 0.0;
    po.Register("rand-prune", &rand_prune, "Randomized pruning of posteriors less than this");
    AmMfa2GselectDirectConfig gsDirectConfig;
    gsDirectConfig.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        gpost_wspecifier = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

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

    double tot_like = 0.0;
    kaldi::int64 tot_t = 0;

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);
    GaussPostWriter gpost_writer(gpost_wspecifier);

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

        GaussPost gpost(posterior.size());  // posterior.size() == T.
        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Posterior vector has wrong size "<< (posterior.size()) << " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        for (size_t i = 0; i < posterior.size(); i++) {

          gpost[i].reserve(posterior[i].size());

          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;

            Vector<BaseFloat> this_post_vec;
            BaseFloat like = am_mfa2.LogLikelihood(pdf_id, mat.Row(i),  &this_post_vec);
            this_post_vec.Scale(weight);

            if (rand_prune > 0.0) {
              for (int32 k = 0; k < this_post_vec.Dim(); k++)
                this_post_vec(k) = RandPrune(this_post_vec(k), rand_prune);
            }
            if (!this_post_vec.IsZero())
               gpost[i].push_back(std::make_pair(tid, this_post_vec));

            tot_weight += weight;
            tot_like_this_file += weight * like;
          }
        }

        KALDI_LOG << "Average like for this file is "
                  << (tot_like_this_file/posterior.size()) << " over "
                  << posterior.size() <<" frames.";
        tot_like += tot_like_this_file;
        tot_t += posterior.size();
        if (num_done % 10 == 0)
          KALDI_LOG << "Avg like per frame so far is "
                    << (tot_like/tot_t);
        gpost_writer.Write(utt, gpost);
      }
    }

    KALDI_LOG << "Overall like per frame (Gaussian only) = "
              << (tot_like/tot_t) << " over " << tot_t << " frames.";

    KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
              << " with no posteriors, " << num_other_error
              << " with other errors.";

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


