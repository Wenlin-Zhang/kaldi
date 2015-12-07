// mfabin/am-mfa-acc-stats2.cc

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
#include "mfa2/am-mfa2.h"
#include "mfa/estimate-am-mfa-types.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"
#include "mfa2/estimate-am-mfa2.h"

int main(int argc, char *argv[]) {
	using namespace kaldi;
	try {
		const char *usage =
				"Accumulate numerator and denominator stats for discriminative training\n"
						"of AmMfa (input is posteriors of mixed sign)\n"
						"Usage: am-mfa-acc-stats2 [options] <model-in> <feature-rspecifier> "
						"<posteriors-rspecifier> <num-stats-out> <den-stats-out>\n"
						"e.g.: am-mfa-acc-stats2 1.mdl 1.ali scp:train.scp ark:1.posts num.acc den.acc\n";

		ParseOptions po(usage);
		bool binary = true;
		std::string update_flags_str = "ywS";
		po.Register("binary", &binary, "Write output in binary mode");
		po.Register("update-flags", &update_flags_str,
				"Which AmMfa parameters to accumulate "
						"stats for: subset of ywS.");
		AmMfa2GselectDirectConfig gsDirectConfig;
		gsDirectConfig.Register(&po);

		po.Read(argc, argv);

		kaldi::AmMfaUpdateFlagsType acc_flags = StringToAmMfaUpdateFlags(
				update_flags_str);

		if (po.NumArgs() != 5) {
			po.PrintUsage();
			exit(1);
		}

		std::string model_filename = po.GetArg(1),
				feature_rspecifier = po.GetArg(2),
				posteriors_rspecifier = po.GetArg(3),
				num_accs_wxfilename = po.GetArg(4),
				den_accs_wxfilename = po.GetArg(5);

		using namespace kaldi;
		typedef kaldi::int32 int32;
		typedef kaldi::int64 int64;

		// Initialize the readers before the model, as the model can
		// be large, and we don't want to call fork() after reading it if
		// virtual memory may be low.
		SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
		RandomAccessPosteriorReader posteriors_reader(posteriors_rspecifier);

		AmMfa2 am_mfa2;
		TransitionModel trans_model;
		{
			bool binary;
			Input ki(model_filename, &binary);
			trans_model.Read(ki.Stream(), binary);
			am_mfa2.Read(ki.Stream(), binary);
			am_mfa2.PreCompute();
		}

		Vector<double> num_transition_accs, den_transition_accs;
		if (acc_flags & kaldi::kAmMfaTransitions) {
			trans_model.InitStats(&num_transition_accs);
			trans_model.InitStats(&den_transition_accs);
		}
		MleAmMfa2Accs num_accs, den_accs;
		num_accs.ResizeAccumulators(am_mfa2);
		den_accs.ResizeAccumulators(am_mfa2);

		double tot_like = 0.0, tot_weight = 0.0, tot_abs_weight = 0.0;
		int64 tot_frames = 0;

		int32 num_done = 0, num_no_posterior = 0, num_other_error = 0;
		for (; !feature_reader.Done(); feature_reader.Next()) {
			std::string utt = feature_reader.Key();
			if (!posteriors_reader.HasKey(utt)) {
				num_no_posterior++;
			} else {
				const Matrix<BaseFloat> &mat = feature_reader.Value();
				const Posterior &posterior = posteriors_reader.Value(utt);

				if (posterior.size() != mat.NumRows()) {
					KALDI_WARN<< "Alignments has wrong size "<< (posterior.size()) <<
					" vs. "<< (mat.NumRows());
					num_other_error++;
					continue;
				}

				num_done++;
				BaseFloat tot_like_this_file = 0.0, tot_weight_this_file = 0.0,
						tot_abs_weight_this_file = 0.0;

				for (size_t i = 0; i < posterior.size(); i++) {
					for (size_t j = 0; j < posterior[i].size(); j++) {
						int32 tid = posterior[i][j].first, pdf_id =
								trans_model.TransitionIdToPdf(tid);
						BaseFloat weight = posterior[i][j].second, abs_weight =
								std::abs(weight);

						if (acc_flags & kaldi::kAmMfaTransitions) {
							trans_model.Accumulate(abs_weight, tid,
									weight > 0 ?
											&num_transition_accs :
											&den_transition_accs);
						}
						tot_like_this_file +=
								(weight > 0 ? num_accs : den_accs).Accumulate(
										am_mfa2, mat.Row(i), pdf_id, abs_weight,
										acc_flags) * weight;
						tot_weight_this_file += weight;
						tot_abs_weight_this_file += abs_weight;
					}
				}

				tot_like += tot_like_this_file;
				tot_weight += tot_weight_this_file;
				tot_abs_weight += tot_abs_weight_this_file;
				tot_frames += posterior.size();
				if (num_done % 50 == 0)
					KALDI_LOG<< "Processed " << num_done << " utterances.";
				}
			}
		KALDI_LOG<< "Overall weighted acoustic likelihood per frame was "
		<< (tot_like/tot_frames) << " over " << tot_frames << " frames; "
		<< "average weight per frame is " << (tot_weight/tot_frames)
		<< ", average abs(weight) per frame is "
		<< (tot_abs_weight/tot_frames);

		KALDI_LOG<< "Done " << num_done << " files, " << num_no_posterior
		<< " with no posteriors, " << num_other_error
		<< " with other errors.";

		{
			Output ko(num_accs_wxfilename, binary);
			// TODO(arnab): Ideally, we shouldn't be writing transition accs if not
			// asked for, but that will complicate reading later. To be fixed?
			num_transition_accs.Write(ko.Stream(), binary);
			num_accs.Write(ko.Stream(), binary);
		}
		{
			Output ko(den_accs_wxfilename, binary);
			den_transition_accs.Write(ko.Stream(), binary);
			den_accs.Write(ko.Stream(), binary);
		}
		KALDI_LOG<< "Written accs.";
		return (num_done != 0 ? 0 : 1);
	} catch (const std::exception &e) {
		std::cerr << e.what();
		return -1;
	}
}

