// mfabin/am-mfa-shrink-by-mfa-post-sum.cc

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
#include "hmm/transition-model.h"
#include "mfa/estimate-am-mfa.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  try {
    const char *usage =
        "Shrink the AmMfa acoustic model by the MFA posterior sum vectors.\n"
        "Usage: am-mfa-shrink-by-mfa-post-sum [options] <model-in> <mfa-post-sum-filename> <model-out>\n"
        "e.g.: am-mfa-post-to-gpost 1.mdl 1.ali scp:train.scp 'ark:ali-to-post ark:1.ali ark:-|' ark:-";

    bool binary_write = true;
    int32 maxComp = 100;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("max-comp", &maxComp, "maximal count of component for each state (100).");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        mfa_post_sum_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    using namespace kaldi;
    typedef kaldi::int32 int32;

    AmMfa am_mfa;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
    }

    /// Read the MFA posterior sum matrix
    Matrix<BaseFloat> mfa_post_sum_mat;
    ReadKaldiObject(mfa_post_sum_filename, &mfa_post_sum_mat);

    /// Shrink the AmMfa acoustic model
    MleAmMfaUpdater::ShrinkAmMfa(&am_mfa, mfa_post_sum_mat, maxComp);

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_mfa.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;

    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


