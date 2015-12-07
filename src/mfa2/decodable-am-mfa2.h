// mfa2/decodable-am-mfa2.h
//
// Copyright 2015 Wen-Lin Zhang
//
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

#ifndef KALDI_DECODABLE_AM_MFA2_H_
#define KALDI_DECODABLE_AM_MFA2_H_

#include <vector>

#include "base/kaldi-common.h"
#include "mfa2/am-mfa2.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"

namespace kaldi {

class DecodableAmMfa2 : public DecodableInterface {
 public:
  DecodableAmMfa2( const AmMfa2 &am,
                  const TransitionModel &tm,
                  const Matrix<BaseFloat> &feats):
                    acoustic_model_(am),  trans_model_(tm),
                    feature_matrix_(feats), previous_frame_(-1)
  {
    ResetLogLikeCache();
  }

  // Note, frames are numbered from zero, but transition indices are 1-based!
  // This is for compatibility with OpenFST.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid));
  }
  int32 NumFrames()const { return feature_matrix_.NumRows(); }
  virtual int32 NumIndices()const { return trans_model_.NumTransitionIds(); }

  virtual bool IsLastFrame(int32 frame)const {
    KALDI_ASSERT(frame < NumFrames());
    return (frame == NumFrames() - 1);
  }

 protected:
  void ResetLogLikeCache();
  virtual BaseFloat LogLikelihoodZeroBased(int32 frame, int32 pdf_id);

  const AmMfa2 &acoustic_model_;
  const TransitionModel &trans_model_;  ///< for tid to pdf mapping
  const Matrix<BaseFloat> &feature_matrix_;
  int32 previous_frame_;

  /// Defines a cache record for a state
  struct LikelihoodCacheRecord {
    BaseFloat log_like;  ///< Cache value
    int32 hit_time;     ///< Frame for which this value is relevant
  };

  /// Cached per-frame quantities used in SGMM likelihood computation.
  std::vector<LikelihoodCacheRecord> log_like_cache_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmMfa2);
};

class DecodableAmMfa2Scaled : public DecodableAmMfa2 {
 public:
  DecodableAmMfa2Scaled(const AmMfa2 &am,
                       const TransitionModel &tm,
                       const Matrix<BaseFloat> &feats,
                       BaseFloat scale)
      : DecodableAmMfa2(am, tm, feats),
        scale_(scale) {}

  // Note, frames are numbered from zero but transition-ids from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid))
            * scale_;
  }

 private:
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmMfa2Scaled);
};


}  // namespace kaldi

#endif  // KALDI_DECODABLE_AM_MFA_H_
