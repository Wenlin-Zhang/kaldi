// mfa2bin/am-mfa2-copy-am-mfa.cc

// Copyright 2015  Wen-Lin Zhang

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
#include "mfa2/am-mfa2.h"
#include "hmm/transition-model.h"
#include "tree/context-dep.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;

    const char *usage =
        "Create an AmMfa2 acoustic model form an AmMfa acoustic model.\n"
        "Usage:  am-mfa-to-am-mfa2 [options] <am-mfa-model-in> <am-mfa2-model-out>\n ";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string ammfa_in_filename = po.GetArg(1),
        ammfa2_out_filename = po.GetArg(2);

    AmMfa am_mfa;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(ammfa_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
    }

    AmMfa2 am_mfa2;
    am_mfa2.CopyFromAmMfa(am_mfa);
    {
      kaldi::Output ko(ammfa2_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      am_mfa2.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Written model to " << ammfa2_out_filename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
