// mfabin/am-mfa-est.cc

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
#include "mfa/am-mfa.h"
#include "hmm/transition-model.h"
#include "mfa/estimate-am-mfa.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Estimate AmMfa model parameters from accumulated stats.\n"
        "Usage: am-mfa-est [options] <model-in> <stats-in> <model-out>\n";

    bool binary_write = true;
    std::string update_flags_str = "yMwStN";
    std::string occs_out_filename;
    kaldi::MleTransitionUpdateConfig tcfg;
    kaldi::MleAmMfaOptions ammfa_opts;
    bool remove_speaker_space = false;

    ParseOptions po(usage);
    po.Register("binary", &binary_write, "Write output in binary mode");
    po.Register("update-flags", &update_flags_str, "Which SGMM parameters to "
                "update: subset of yMwStN.");
    po.Register("write-occs", &occs_out_filename, "File to write state "
                "occupancies to.");
    po.Register("remove-speaker-space", &remove_speaker_space, "Remove speaker-specific "
                "projections N");
    AmMfaGselectDirectConfig gsDirectConfig;
    gsDirectConfig.Register(&po);
    tcfg.Register(&po);
    ammfa_opts.Register(&po);

    po.Read(argc, argv);
    if (po.NumArgs() != 3) {
      po.PrintUsage();
      exit(1);
    }
    std::string model_in_filename = po.GetArg(1),
        stats_filename = po.GetArg(2),
        model_out_filename = po.GetArg(3);

    kaldi::AmMfaUpdateFlagsType update_flags =
        StringToAmMfaUpdateFlags(update_flags_str);

    AmMfa am_mfa;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_in_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
      am_mfa.SetPreSelGaussDirect(&gsDirectConfig);
    }

    Vector<double> transition_accs;
    MleAmMfaAccs ammfa_accs;
    {
      bool binary;
      Input ki(stats_filename, &binary);
      transition_accs.Read(ki.Stream(), binary);
      ammfa_accs.Read(ki.Stream(), binary, true);  // true == add; doesn't matter here.
    }

    if (update_flags & kAmMfaTransitions) {  // Update transition model.
      BaseFloat objf_impr, count;
      trans_model.MleUpdate(transition_accs, tcfg, &objf_impr, &count);
      KALDI_LOG << "Transition model update: Overall " << (objf_impr/count)
                << " log-like improvement per frame over " << (count)
                << " frames.";
    }

    ammfa_accs.Check(am_mfa, true); // Will check consistency and print some diagnostics.

    { // Do the update.
      kaldi::MleAmMfaUpdater updater(ammfa_opts);
      updater.Update(ammfa_accs, &am_mfa, update_flags);
    }

    if (remove_speaker_space) {
      KALDI_LOG << "Removing speaker space (projections N_)";
      am_mfa.RemoveSpeakerSpace();
    }

    {
      Output ko(model_out_filename, binary_write);
      trans_model.Write(ko.Stream(), binary_write);
      am_mfa.Write(ko.Stream(), binary_write);
    }

    int32 total_comp = am_mfa.TotalNumComps();
    KALDI_LOG << "Update AmMfa model complete. Total = " << total_comp
        << " Gaussians / " << am_mfa.NumStates() << " states. Average = " << (BaseFloat)total_comp / am_mfa.NumStates()
        << "Gaussian per state.";
    // get state occupations if necessary
    if (occs_out_filename.empty() == false) {
      Vector<BaseFloat> state_occs;
      state_occs.Resize(ammfa_accs.NumStates());
      for (int j = 0; j < ammfa_accs.NumStates(); j++)
        state_occs(j) = ammfa_accs.StateOcc(j);
      WriteKaldiObject(state_occs, occs_out_filename, binary_write);
    }


    KALDI_LOG << "Written model to " << model_out_filename;
    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}


