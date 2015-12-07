// mfa2bin/am-mfa2-rescore-lattice.cc

// Copyright 2015   Wen-Lin Zhang

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
#include "util/stl-utils.h"
#include "mfa2/am-mfa2.h"
#include "hmm/transition-model.h"
#include "fstext/fstext-lib.h"
#include "lat/kaldi-lattice.h"
#include "lat/lattice-functions.h"
#include "mfa2/decodable-am-mfa2.h"

int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    typedef kaldi::int64 int64;
    using fst::SymbolTable;
    using fst::VectorFst;
    using fst::StdArc;

    const char *usage =
        "Replace the acoustic scores on a lattice using a new model.\n"
        "Usage: am-mfa2-rescore-lattice [options] <model-in> <lattice-rspecifier> "
        "<feature-rspecifier> <lattice-wspecifier>\n"
        " e.g.: am-mfa2-rescore-lattice 1.mdl ark:1.lats scp:trn.scp ark:2.lats\n";
    kaldi::ParseOptions po(usage);

    kaldi::BaseFloat old_acoustic_scale = 0.0;
    po.Register("old-acoustic-scale", &old_acoustic_scale,
                "Add the current acoustic scores with some scale.");

    AmMfa2GselectDirectConfig gsDirectConfig;
    gsDirectConfig.Register(&po);
    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string model_filename = po.GetArg(1),
        lats_rspecifier = po.GetArg(2),
        feature_rspecifier = po.GetArg(3),
        lats_wspecifier = po.GetArg(4);

    AmMfa2 am_mfa2;
    TransitionModel trans_model;
    {
      bool binary;
      Input ki(model_filename, &binary);
      trans_model.Read(ki.Stream(), binary);
      am_mfa2.Read(ki.Stream(), binary);
      am_mfa2.PreCompute();
    }

    RandomAccessBaseFloatMatrixReader feature_reader(feature_rspecifier);
    // Read as regular lattice
    SequentialCompactLatticeReader compact_lattice_reader(lats_rspecifier);
    // Write as compact lattice.
    CompactLatticeWriter compact_lattice_writer(lats_wspecifier);

    int32 n_done = 0, num_no_feats = 0, num_other_error = 0;
    for (; !compact_lattice_reader.Done(); compact_lattice_reader.Next()) {
      std::string utt = compact_lattice_reader.Key();
      if (!feature_reader.HasKey(utt)) {
        KALDI_WARN << "No feature found for utterance " << utt << ". Skipping";
        num_no_feats++;
        continue;
      }

      CompactLattice clat = compact_lattice_reader.Value();
      compact_lattice_reader.FreeCurrent();
      if (old_acoustic_scale != 1.0)
        fst::ScaleLattice(fst::AcousticLatticeScale(old_acoustic_scale), &clat);

      kaldi::uint64 props = clat.Properties(fst::kFstProperties, false);
      if (!(props & fst::kTopSorted)) {
        if (fst::TopSort(&clat) == false)
          KALDI_ERR << "Cycles detected in lattice.";
      }

      vector<int32> state_times;
      int32 max_time = kaldi::CompactLatticeStateTimes(clat, &state_times);
      const Matrix<BaseFloat> &feats = feature_reader.Value(utt);
      if (feats.NumRows() != max_time) {
        KALDI_WARN << "Skipping utterance " << utt << " since number of time "
                   << "frames in lattice ("<< max_time << ") differ from "
                   << "number of feature frames (" << feats.NumRows() << ").";
        num_other_error++;
        continue;
      }

      DecodableAmMfa2 am_mfa2_decodable(am_mfa2, trans_model, feats);

      if (kaldi::RescoreCompactLattice(&am_mfa2_decodable, &clat)) {
                compact_lattice_writer.Write(utt, clat);
                n_done++;
      }
      else
        num_other_error++;
    }

    KALDI_LOG << "Done " << n_done << " lattices.";
    return (n_done != 0 ? 0 : 1);
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
