// mfabin/am-mfa-post-to-mfa-post-sum.cc

// Copyright 2014   Wen-Lin Zhang

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
#include "mfa/am-mfa.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"
#include "mfa/estimate-am-mfa.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Convert posteriors to MFA-based Gaussian posterior sum for AmMfa training.\n"
        "Usage: am-mfa-post-to-mfa-post-sum [options] <model-in> <feature-rspecifier> "
        "<posteriors-rspecifier> <mfa-post-sum-mat-filename>\n"
        "e.g.: am-mfa-post-to-gpost 1.mdl 1.ali scp:train.scp 'ark:ali-to-post ark:1.ali ark:-|' ark:-";

    bool useMFAWeights = false;
    bool binary_write = true;
    MfaGselectConfig gselectOpt;
    bool use_gselect = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("use-mfa-weights", &useMFAWeights, "Wether use the MFA weight to calculate the posteriors (false).");
    po.Register("use-gselect", &use_gselect, "Whether use gaussion selection during accumulation.");
    gselectOpt.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        posteriors_rspecifier = po.GetArg(3),
        mfa_post_sum_mat_filename = po.GetArg(4);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    MFA mfa;
    TransitionModel trans_model;
    int S;
    {
      AmMfa am_mfa;
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
      S = am_mfa.NumStates();
      mfa.CopyFromMFA(am_mfa.GetMFA());
    }
    mfa.PreCompute();
    if (use_gselect == true)
    {
      KALDI_LOG << "Use gselect, prune to " << gselectOpt.diag_gmm_nbest << " Gaussian per frame.";
      mfa.SetPreSelGauss(true, &gselectOpt);
    }

    SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
    RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

    Matrix<BaseFloat> mfa_post_sum_mat(S, mfa.NumComps());
    std::vector<int32> gselect;
    int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
    for (; !feature_reader.Done(); feature_reader.Next()) {
      std::string utt = feature_reader.Key();
      if (!posteriors_reader.HasKey(utt)) {
        num_no_posterior++;
      }
      else
      {
        const Matrix<BaseFloat> &mat = feature_reader.Value();
        const Posterior &posterior = posteriors_reader.Value(utt);

        if (posterior.size() != mat.NumRows()) {
          KALDI_WARN << "Alignments has wrong size "<< (posterior.size()) <<
              " vs. "<< (mat.NumRows());
          num_other_error++;
          continue;
        }

        num_done++;
        for (size_t i = 0; i < posterior.size(); i++) {
          Vector<BaseFloat> this_post_vec(mfa.NumComps());
          if (use_gselect == true)
          {
            gselect.resize(0);
            mfa.GaussianSelection(mat.Row(i), &gselect);
          }
          for (size_t k = 0; k < mfa.NumComps(); ++k) {
            if (gselect.size() == 0 || std::find(gselect.begin(), gselect.end(), k) != gselect.end())
            {
              this_post_vec(k) = mfa.LogLikelihood(mat.Row(i), k);
              if (useMFAWeights == true)
                this_post_vec(k) += log((double) mfa.GetWeight(k));
            } else
              this_post_vec(k) = kLogZeroBaseFloat;
          }
          this_post_vec.ApplySoftMax();

          for (size_t j = 0; j < posterior[i].size(); j++) {
            int32 tid = posterior[i][j].first,  // transition identifier.
                pdf_id = trans_model.TransitionIdToPdf(tid);
            BaseFloat weight = posterior[i][j].second;

            mfa_post_sum_mat.Row(pdf_id).AddVec(weight, this_post_vec);
          }
        }
      }
      if (num_done % 50 == 0)
        KALDI_LOG << "Done " << num_done << " files, " << num_no_posterior
                  << " with no posteriors, " << num_other_error
                  << " with other errors.";
    }

    /// Write the results to table
    WriteKaldiObject(mfa_post_sum_mat, mfa_post_sum_mat_filename, binary_write);

    KALDI_LOG << "Written stat to " << mfa_post_sum_mat_filename;

    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


