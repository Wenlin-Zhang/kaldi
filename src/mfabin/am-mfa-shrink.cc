// mfabin/am-mfa-shrink.cc

// Copyright 2013   Wen-Lin Zhang

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
#include "mfa/am-mfa.h"
#include "hmm/transition-model.h"
#include "mfa/estimate-am-mfa.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;

    const char *usage = "Shrink the AmMfa model.\n"
        "Usage: am-mfa-shrink [options] model-in model-out.\n";

    bool binary_write = true;
    BaseFloat minW = 1.0e-5;
    kaldi::ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("min-w", &minW, "The minimal weight (Only weight > minW will be reserved.)");
    po.Read(argc, argv);

    if (po.NumArgs() < 2) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_in_filename = po.GetArg(1), model_out_filename = po.GetArg(2);

    AmMfa am_mfa;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
    }

    MleAmMfaUpdater::ShrinkAmMfa(&am_mfa, minW);

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_mfa.Write(ko.Stream(), binary_write);
    }

    KALDI_LOG << "Written model to " << model_out_filename;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return -1;
  }
}
