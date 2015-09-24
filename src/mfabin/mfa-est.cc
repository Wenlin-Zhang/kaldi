// mfabin/mfa-est.cc

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
#include "mfa/mle-mfa.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef int32 int32;

    const char *usage =
        "Estimate an MoFA model from the accumulated stats.\n"
        "Usage:  mfa-est [options] <model-in> <stats-in> <model-out>\n";

    bool binary_write = true;
    bool remove_comp = false;
    BaseFloat min_occ = 10.0;
    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("remove-comp", &remove_comp, "Whether remove low occupation components");
    po.Register("min-occ", &min_occ, "Minimal component occupation number");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    MFA mfa;
    {
      bool binary_read;
      Input ki(model_in_filename, &binary_read);
      mfa.Read(ki.Stream(), binary_read);
    }

    AccumMFA mfa_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      mfa_accs.Read(ki.Stream(), binary, true /* add accs, doesn't matter */);
    }

    mfa_accs.MleMFAUpdate(&mfa, remove_comp, min_occ);

    WriteKaldiObject(mfa, model_out_filename, binary_write);

    KALDI_LOG << "Written model to " << model_out_filename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
