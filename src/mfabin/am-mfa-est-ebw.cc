// mfabin/am-mfa-est-ebw.cc

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
#include "hmm/transition-model.h"
#include "mfa/estimate-am-mfa-ebw.h"


int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  using std::string;
  try {
    const char *usage =
        "Estimate AmMfa model parameters discriminatively using Extended\n"
        "Baum-Welch style of update\n"
        "Usage: am-mfa-est-ebw [options] <model-in> <num-stats-in> <den-stats-in> <model-out>\n";


    string update_flags_str = "yMwSmt";
    bool binary_write = true;
    EbwAmMfaOptions opts;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of vMNwcSt.");
    opts.Register(&po);
    AmMfaGselectDirectConfig gsDirectConfig;
    gsDirectConfig.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }
    string model_in_filename = po.GetArg(1),
        num_stats_filename = po.GetArg(2),
        den_stats_filename = po.GetArg(3),
        model_out_filename = po.GetArg(4);
    
    AmMfaUpdateFlagsType update_flags = StringToAmMfaUpdateFlags(update_flags_str);

    AmMfa am_mfa;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
      am_mfa.SetPreSelGaussDirect(&gsDirectConfig);
      am_mfa.PreCompute();
    }

    MleAmMfaAccs ammfa_num_accs;
    {
      bool binary;
      Vector<double> transition_accs; // won't be used.
      Input ki(num_stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      ammfa_num_accs.Read(ki.Stream(), binary, false);  // false == add; doesn't matter.
    }
    MleAmMfaAccs ammfa_den_accs;
    {
      bool binary;
      Vector<double> transition_accs; // won't be used.
      Input ki(den_stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      ammfa_den_accs.Read(ki.Stream(), binary, false);  // false == add; doesn't matter.
    }
    
    ammfa_num_accs.Check(am_mfa, true); // Will check consistency and print some diagnostics.
    ammfa_den_accs.Check(am_mfa, true); // Will check consistency and print some diagnostics.

    {  // Update AmMfa.
      BaseFloat auxf_impr, count;
      kaldi::EbwAmMfaUpdater ammfa_updater(opts);
      ammfa_updater.Update(ammfa_num_accs, ammfa_den_accs, &am_mfa,
                          update_flags, &auxf_impr, &count);
      KALDI_LOG << "Overall auxf impr/frame from SGMM update is " << (auxf_impr/count)
                << " over " << count << " frames.";
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_mfa.Write(ko.Stream(), binary_write);
    }
    
    KALDI_LOG << "Wrote model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
