// mfabin/mfa-init.cc

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
#include "gmm/full-gmm.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Init an MoFA model using a full-covariance GMM.\n"
        "Usage:  mfa-init [options] <fgmm-model-in> <mfa-model-out>\n "
        "e.g.:   mfa-init final.fgmm 1.mfa\n";

    ParseOptions po(usage);
    bool binary = true;
    InitMFAOptions opts;
    po.Register("binary", &binary, "Write output in binary mode");
    opts.Register(&po);

    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_fname = po.GetArg(1),
        model_out_fname = po.GetArg(2);

    FullGmm fgmm;
    {
      bool binary_read;
      Input ki(model_in_fname, &binary_read);
      fgmm.Read(ki.Stream(), binary_read);
    }

    MFA mfa;
    mfa.Init(fgmm, opts);

    KALDI_LOG << "Done! The dimensons of local subspace are : \n";
    for(int i = 0; i < mfa.NumComps(); ++ i)
      KALDI_LOG << mfa.GetLocalDim(i) << "\t";

    KALDI_LOG << ".\n";

    WriteKaldiObject(mfa, model_out_fname, binary);
    KALDI_LOG << "Written MFA model to " << model_out_fname;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
