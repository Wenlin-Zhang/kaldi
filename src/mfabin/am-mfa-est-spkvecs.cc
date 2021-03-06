// mfabin/am-mfa-est-spkvecs.cc

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

#include <string>
using std::string;
#include <vector>
using std::vector;

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "mfa/am-mfa.h"
#include "mfa/estimate-am-mfa.h"
#include "hmm/posterior.h"
#include "hmm/transition-model.h"

namespace kaldi {

void AccumulateForUtterance(const Matrix<BaseFloat> &feats,
                            const Posterior &post,
                            const TransitionModel &trans_model,
                            const AmMfa &am_mfa,
                            const vector< vector<int32> > &gselect,
                            const AmMfaPerSpkDerivedVars& spk_vars,
                            MleAmMfaSpeakerAccs *spk_stats) {
  std::vector<int32> tmp_gselect;
  for (size_t i = 0; i < post.size(); i++) {
    const std::vector<int32>* this_gselect = NULL;
    if (!gselect.empty())
      this_gselect = &(gselect[i]);
    else if (am_mfa.IsPreSelGauss() == true)
    {
      am_mfa.GaussianSelection(feats.Row(i), &tmp_gselect);
      this_gselect = &tmp_gselect;
    }

    for (size_t j = 0; j < post[i].size(); j++) {
      int32 pdf_id = trans_model.TransitionIdToPdf(post[i][j].first);
      spk_stats->Accumulate(am_mfa, feats.Row(i), &spk_vars, pdf_id, post[i][j].second, this_gselect);
    }
  }
}

}  // end namespace kaldi

int main(int argc, char *argv[]) {
  try {
    typedef kaldi::int32 int32;
    using namespace kaldi;
    const char *usage =
        "Estimate AmMfa speaker vectors, either per utterance or for the "
        "supplied set of speakers (with spk2utt option).\n"
        "Reads Gaussian-level posteriors. Writes to a table of vectors.\n"
        "Usage: am-mfa-est-spkvecs [options] <model-in> <feature-rspecifier> "
        "<post-rspecifier> <vecs-wspecifier>\n";

    ParseOptions po(usage);
    BaseFloat min_count = 100;
    string spk2utt_rspecifier, spkvecs_rspecifier;
    string gselect_rspecifier;
    bool use_gselect = false;
    AmMfaGselectConfig gsConfig;
    gsConfig.Register(&po);

    po.Register("gselect", &gselect_rspecifier,
                "File to read precomputed per-frame Gaussian indices from.");
    po.Register("use-gselect", &use_gselect,
                "whether doing per-frame gselect if the gselect rspecifier is not provided.");
    po.Register("spk2utt", &spk2utt_rspecifier,
                "File to read speaker to utterance-list map from.");
    po.Register("spkvec-min-count", &min_count,
                "Minimum count needed to estimate speaker vectors");
    po.Register("spk-vecs", &spkvecs_rspecifier, "Speaker vectors to use during aligment (rspecifier)");
    AmMfaGselectDirectConfig gsDirectConfig;
    gsDirectConfig.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    string model_rxfilename = po.GetArg(1),
        feature_rspecifier = po.GetArg(2),
        post_rspecifier = po.GetArg(3),
        vecs_wspecifier = po.GetArg(4);

    TransitionModel trans_model;
    AmMfa am_mfa;
    {
      bool binary;
      Input ki(model_rxfilename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa.Read(ki.Stream(), binary);
      am_mfa.SetPreSelGaussDirect(&gsDirectConfig);
      if (use_gselect == true)
      {
        KALDI_LOG << "Do per-frame Gauss pre-selection if the gselect_rspecifier is not provided.";
        am_mfa.SetPreSelGauss(true, &gsConfig);
      }
      am_mfa.PreCompute();
    }
    MleAmMfaSpeakerAccs spk_stats(am_mfa);

    RandomAccessPosteriorReader post_reader(post_rspecifier);
    RandomAccessInt32VectorVectorReader gselect_reader(gselect_rspecifier);
    RandomAccessBaseFloatVectorReader spkvecs_reader(spkvecs_rspecifier);

    BaseFloatVectorWriter vecs_writer(vecs_wspecifier);

    double tot_impr = 0.0, tot_t = 0.0;
    int32 num_done = 0, num_no_post = 0, num_other_error = 0;
    std::vector<std::vector<int32> > empty_gselect;

    if (!spk2utt_rspecifier.empty()) {  // per-speaker adaptation
      SequentialTokenVectorReader spk2utt_reader(spk2utt_rspecifier);
      RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);

      for (; !spk2utt_reader.Done(); spk2utt_reader.Next()) {
        spk_stats.Clear();

        string spk = spk2utt_reader.Key();
        const vector<string> &uttlist = spk2utt_reader.Value();

        AmMfaPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(spk)) {
            spk_vars.v_s = spkvecs_reader.Value(spk);
            am_mfa.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << spk;
          }
        }  // else spk_vars is "empty"

        for (size_t i = 0; i < uttlist.size(); i++) {
          std::string utt = uttlist[i];
          if (!feature_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find features for utterance " << utt;
            continue;
          }
          if (!post_reader.HasKey(utt)) {
            KALDI_WARN << "Did not find posteriors for utterance " << utt;
            num_no_post++;
            continue;
          }
          const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
          const Posterior &post = post_reader.Value(utt);
          if (static_cast<int32>(post.size()) != feats.NumRows()) {
            KALDI_WARN << "Posterior vector has wrong size " << (post.size())
                       << " vs. " << (feats.NumRows());
            num_other_error++;
            continue;
          }

          bool has_gselect = false;
          if (gselect_reader.IsOpen()) {
            has_gselect = gselect_reader.HasKey(utt)
                && gselect_reader.Value(utt).size() == feats.NumRows();
            if (!has_gselect)
              KALDI_WARN
                  << "No Gaussian-selection info available for utterance "
                  << utt << " (or wrong size)";
          }
          const std::vector<std::vector<int32> > *gselect = (
              has_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

          AccumulateForUtterance(feats, post, trans_model, am_mfa, *gselect, spk_vars, &spk_stats);
          num_done++;
        }  // end looping over all utterances of the current speaker

        BaseFloat impr, spk_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_mfa.SpkSpaceDim(), kSetZero);
          if (spk_vars.v_s.Dim() != 0) spk_vec.CopyFromVec(spk_vars.v_s);
          spk_stats.Update(min_count, &spk_vec, &impr, &spk_tot_t);
          vecs_writer.Write(spk, spk_vec);
        }
        KALDI_LOG << "For speaker " << spk << ", auxf-impr from speaker vector is "
                  << (impr/spk_tot_t) << ", over " << spk_tot_t << " frames.";
        tot_impr += impr;
        tot_t += spk_tot_t;
      }  // end looping over speakers
    } else {  // per-utterance adaptation
      SequentialBaseFloatMatrixReader feature_reader(feature_rspecifier);
      for (; !feature_reader.Done(); feature_reader.Next()) {
        string utt = feature_reader.Key();
        if (!post_reader.HasKey(utt)) {
          KALDI_WARN << "Did not find posts for utterance "
                     << utt;
          num_no_post++;
          continue;
        }
        const Matrix<BaseFloat> &feats = feature_reader.Value();

        AmMfaPerSpkDerivedVars spk_vars;
        if (spkvecs_reader.IsOpen()) {
          if (spkvecs_reader.HasKey(utt)) {
            spk_vars.v_s = spkvecs_reader.Value(utt);
            am_mfa.ComputePerSpkDerivedVars(&spk_vars);
          } else {
            KALDI_WARN << "Cannot find speaker vector for " << utt;
          }
        }  // else spk_vars is "empty"
        const Posterior &post = post_reader.Value(utt);

        if (static_cast<int32>(post.size()) != feats.NumRows()) {
          KALDI_WARN << "Posterior has wrong size " << (post.size())
              << " vs. " << (feats.NumRows());
          num_other_error++;
          continue;
        }
        num_done++;

        spk_stats.Clear();

        bool has_gselect = false;
        if (gselect_reader.IsOpen()) {
          has_gselect = gselect_reader.HasKey(utt)
              && gselect_reader.Value(utt).size() == feats.NumRows();
          if (!has_gselect)
            KALDI_WARN << "No Gaussian-selection info available for utterance "
                       << utt << " (or wrong size)";
        }
        const std::vector<std::vector<int32> > *gselect = (
            has_gselect ? &gselect_reader.Value(utt) : &empty_gselect);

        AccumulateForUtterance(feats, post, trans_model, am_mfa, *gselect, spk_vars, &spk_stats);

        BaseFloat impr, utt_tot_t;
        {  // Compute the spk_vec and write it out.
          Vector<BaseFloat> spk_vec(am_mfa.SpkSpaceDim(), kSetZero);
          if (spk_vars.v_s.Dim() != 0) spk_vec.CopyFromVec(spk_vars.v_s);
          spk_stats.Update(min_count, &spk_vec, &impr, &utt_tot_t);
          vecs_writer.Write(utt, spk_vec);
        }
        KALDI_LOG << "For utterance " << utt << ", auxf-impr from speaker vectors is "
                  << (impr/utt_tot_t) << ", over " << utt_tot_t << " frames.";
        tot_impr += impr;
        tot_t += utt_tot_t;
      }
    }

    KALDI_LOG << "Overall auxf impr per frame is "
              << (tot_impr / tot_t) << " over " << tot_t << " frames.";
    KALDI_LOG << "Done " << num_done << " files, " << num_no_post
              << " with no posts, " << num_other_error << " with other errors.";
    return (num_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}

