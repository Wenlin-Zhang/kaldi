// mfa2bin/am-mfa2-est.cc

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
#include "thread/kaldi-thread.h"
#include "mfa2/am-mfa2.h"
#include "hmm/transition-model.h"
#include "mfa2/estimate-am-mfa2.h"

int main(int argc, char *argv[]) {
	try {
		using namespace kaldi;
		typedef kaldi::int32 int32;
		const char *usage =
				"Estimate AmMfa2 model parameters from accumulated stats.\n"
						"Usage: am-mfa2-est [options] <model-in> <stats-in> <model-out>\n";

		bool binary_write = true;
		std::string update_flags_str = "ywS";
		std::string occs_out_filename;

		ParseOptions po(usage);
		po.Register("binary", &binary_write, "Write output in binary mode");
		po.Register("update-flags", &update_flags_str,
				"Which AmMfa2 parameters to "
						"update: subset of ywS.");
		po.Register("write-occs", &occs_out_filename, "File to write state "
				"occupancies to.");

		AmMfa2GselectDirectConfig gsDirectConfig;
		kaldi::MleTransitionUpdateConfig tcfg;
		kaldi::MleAmMfa2Options ammfa2_opts;
		gsDirectConfig.Register(&po);
		tcfg.Register(&po);
		ammfa2_opts.Register(&po);

		po.Read(argc, argv);
		if (po.NumArgs() != 3) {
			po.PrintUsage();
			exit(1);
		}
		std::string model_in_filename = po.GetArg(1), stats_filename =
				po.GetArg(2), model_out_filename = po.GetArg(3);

		kaldi::AmMfaUpdateFlagsType update_flags = StringToAmMfaUpdateFlags(
				update_flags_str);

		AmMfa2 am_mfa2;
		TransitionModel trans_model;
		{
			bool binary;
			Input ki(model_in_filename, &binary);
			trans_model.Read(ki.Stream(), binary);
			am_mfa2.Read(ki.Stream(), binary);
		}

		Vector<double> transition_accs;
		MleAmMfa2Accs ammfa2_accs;
		{
			bool binary;
			Input ki(stats_filename, &binary);
			transition_accs.Read(ki.Stream(), binary);
			ammfa2_accs.Read(ki.Stream(), binary, true); // true == add; doesn't matter here.
		}

		if (update_flags & kAmMfaTransitions) {  // Update transition model.
			BaseFloat objf_impr, count;
			trans_model.MleUpdate(transition_accs, tcfg, &objf_impr, &count);
			KALDI_LOG<< "Transition model update: Overall " << (objf_impr/count)
			<< " log-like improvement per frame over " << (count)
			<< " frames.";
		}

		ammfa2_accs.Check(am_mfa2, true); // Will check consistency and print some diagnostics.

		{ // Do the update.
			kaldi::MleAmMfa2Updater updater(ammfa2_opts);
			updater.Update(ammfa2_accs, &am_mfa2, update_flags);
		}

		{
			Output ko(model_out_filename, binary_write);
			trans_model.Write(ko.Stream(), binary_write);
			am_mfa2.Write(ko.Stream(), binary_write);
		}

		int32 total_comp = am_mfa2.TotalNumComps();
		KALDI_LOG<< "Update AmMfa2 model complete. Total = " << total_comp
		<< " Gaussians / " << am_mfa2.NumStates() << " states. Average = " << (BaseFloat)total_comp / am_mfa2.NumStates()
		<< "Gaussian per state.";
		// get state occupations if necessary
		if (occs_out_filename.empty() == false) {
			Vector<BaseFloat> state_occs;
			state_occs.Resize(ammfa2_accs.NumStates());
			for (int j = 0; j < ammfa2_accs.NumStates(); j++)
				state_occs(j) = ammfa2_accs.StateOcc(j);
			WriteKaldiObject(state_occs, occs_out_filename, binary_write);
		}

		KALDI_LOG<< "Written model to " << model_out_filename;
		return 0;
	} catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}

