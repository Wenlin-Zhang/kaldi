// mfabin/mfa-sum-accs.cc
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

#include "util/common-utils.h"
#include "mfa/mle-mfa.h"


int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;

    const char *usage =
        "Sum multiple accumulated stats files for MFA training.\n"
        "Usage: mfa-sum-accs [options] stats-out stats-in1 stats-in2 ...\n";

    bool binary = true;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary, "Write output in binary mode");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string stats_out_filename = po.GetArg(1);
    kaldi::AccumMFA mfa_accs;

    for (int i = 2, max = po.NumArgs(); i <= max; i++) {
      std::string stats_in_filename = po.GetArg(i);
      bool binary_read;
      kaldi::Input ki(stats_in_filename, &binary_read);
      mfa_accs.Read(ki.Stream(), binary_read, true /*add read values*/);
    }

    // Write out the accs
    {
      kaldi::Output ko(stats_out_filename, binary);
      mfa_accs.Write(ko.Stream(), binary);
    }

    KALDI_LOG << "Written stats to " << stats_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}

