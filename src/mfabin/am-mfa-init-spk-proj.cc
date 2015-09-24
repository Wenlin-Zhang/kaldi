// mfabin/am-mfa-init-spk-proj.cc

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
        "Init the speaker projection matrix of an MoFA acoustic model.\n"
        "Usage:  am-mfa-init-spk-proj [options] <mfa-model-in> <spk-space-dim> <am-mfa-model-out>\n ";

    ParseOptions po(usage);
    bool binary = true;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }

    std::string ammfa_in_filename = po.GetArg(1),
        str_spk_space_dim = po.GetArg(2),
        ammfa_out_filename = po.GetArg(3);

    int spk_space_dim = 0;
    if (false == ConvertStringToInteger(str_spk_space_dim, &spk_space_dim))
    {
      KALDI_ERR << "The dimension of the speaker space must be an integer.";
    }
    KALDI_ASSERT(spk_space_dim >= 0);

    TransitionModel trans_model;
    AmMfa am_mfa;
    {
      bool binary;
      Input ki(ammfa_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
      am_mfa.PreCompute();
    }

    KALDI_LOG << "Init the speaker space by dimension: " << spk_space_dim;

    am_mfa.InitSpkSpace(spk_space_dim);
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
