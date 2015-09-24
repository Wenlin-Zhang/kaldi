// mfabin/am-mfa-to-diag-gmm.cc

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
#include "thread/kaldi-thread.h"
#include "gmm/am-diag-gmm.h"
#include "hmm/transition-model.h"
#include "mfa/am-mfa.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Convert a MFA-based model to a diagonal GMM-based model.\n"
        "Usage: am-mfa-to-diag-gmm [options] <model-in> <model-out>\n";

    ParseOptions po(usage);
    bool binary_write = true;
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string ammfa_in_filename = po.GetArg(1),
        am_diag_gmm_out_filename = po.GetArg(2);

    // read in the AmMfa model
    AmMfa am_mfa;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(ammfa_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
    }

    // Convert to AmDiagGmm model
    AmDiagGmm am_dgmm;
    am_mfa.ConvertToAmDiagGmm(&am_dgmm);

    // write out the AmDiagGmm model
    {
      Output ko(am_diag_gmm_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_dgmm.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << am_diag_gmm_out_filename;
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


