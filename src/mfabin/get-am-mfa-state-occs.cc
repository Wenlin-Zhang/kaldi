// mfabin/get-am-mfa-state-occs.cc

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
#include "mfa/estimate-am-mfa.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Get the state occs from the AmMfa model stats file.\n"
        "Usage: get-am-mfa-state-occs [options] <stats-in> <state-occs-out>\n";

    ParseOptions po(usage);
    bool binary_write = true;
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Read(argc, argv);
    if (po.NumArgs() != 2) {
      po.PrintUsage();
      exit(1);
    }
    std::string stats_filename = po.GetArg(1), occs_out_filename = po.GetArg(2);

    Vector<double> transition_accs;
    MleAmMfaAccs ammfa_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      ammfa_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    // get state occupations
    Vector<BaseFloat> state_occs;
    state_occs.Resize(ammfa_accs.NumStates());
    for (int j = 0; j < ammfa_accs.NumStates(); j++)
       state_occs(j) = ammfa_accs.StateOcc(j);
    WriteKaldiObject(state_occs, occs_out_filename, binary_write);

    KALDI_LOG << "Written state occs to " << occs_out_filename;
    return 0;

  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


