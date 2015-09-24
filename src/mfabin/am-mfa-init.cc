// mfabin/am-mfa-init.cc

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
        "Init an MoFA acoustic model using a background MFA.\n"
        "Usage:  am-mfa-init [options] <topology> <tree> <mfa-model-in> <am-mfa-model-out>\n ";

    ParseOptions po(usage);
    bool binary = true;
    int32 spk_space_dim = 0;
    po.Register("binary", &binary, "Write output in binary mode");
    po.Register("spk-space-dim", &spk_space_dim, "Speaker space dimension.");
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string topo_in_filename = po.GetArg(1),
        tree_in_filename = po.GetArg(2),
        mfa_model_filename = po.GetArg(3),
        ammfa_out_filename = po.GetArg(4);

    ContextDependency ctx_dep;
    {
      bool binary_in;
      Input ki(tree_in_filename.c_str(), &binary_in);
      ctx_dep.Read(ki.Stream(), binary_in);
    }

    HmmTopology topo;
    ReadKaldiObject(topo_in_filename, &topo);
    TransitionModel trans_model(ctx_dep, topo);

    MFA mfa;
    {
      bool binary_read;
      kaldi::Input ki(mfa_model_filename, &binary_read);
      mfa.Read(ki.Stream(), binary_read);
    }

    AmMfa ammfa;
    ammfa.InitializeFromMfa(mfa, trans_model.NumPdfs(), spk_space_dim);
    {
      kaldi::Output ko(ammfa_out_filename, binary);
      trans_model.Write(ko.Stream(), binary);
      ammfa.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Written model to " << ammfa_out_filename;

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
