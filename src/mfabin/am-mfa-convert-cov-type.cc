// mfabin/am-mfa-convert-cov-type.cc

// Copyright 2013  Wen-Lin Zhang

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
#include "mfa/mfa.h"
#include "mfa/am-mfa.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Convert the covariance type of an MoFA acoustic model.\n"
        "Usage:  am-mfa-convert-cov-type [options] <am-mfa-model-in> <am-mfa-model-out>\n ";

    ParseOptions po(usage);
    bool binary = true;
    std::string strType = "full";
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("type", &strType, "Covariance type");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ammfa_in_filename = po.GetArg(1),
        ammfa_out_filename = po.GetArg(2);

    AmMfa am_mfa;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(ammfa_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
      //am_mfa.PreCompute();
    }

    if (strType != "full" && strType != "f" && strType != "diag" && strType != "d")
      KALDI_ERR << "Invalid covariance type: " << strType.c_str() << ". "
                << "The covariance type must be full(f) or diag(d).";
    switch(am_mfa.GetMFA().CovarianceType())
    {
      case DIAG:
        if (strType == "full" || strType == "f")
        {
          KALDI_LOG << "Convert the covariance type from diag to full.";
          am_mfa.ConvertToFullCov();
        }
        else if (strType == "diag" || strType == "d")
        {
          KALDI_LOG << "The covariance type is already diag, no thing will happen.";
        }
        break;
      case FULL:
        if (strType == "diag" || strType == "d")
        {
          KALDI_LOG << "Convert the covariance type from full to diag.";
          am_mfa.ConvertToDiagCov();
        }
        else if (strType == "full" || strType == "f")
        {
          KALDI_LOG << "The covariance type is already full, no thing will happen.";
        }
        break;
      default:
        KALDI_ERR << "Invalid covariance type.";
        break;
    }

    {
      kaldi::Output ko(ammfa_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_mfa.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Written model to " << ammfa_out_filename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
