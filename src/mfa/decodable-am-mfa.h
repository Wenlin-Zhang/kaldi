// mfa/decodable-am-mfa.h
//
// Copyright 2013 Wen-Lin Zhang
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

#ifndef KALDI_DECODABLE_AM_MFA_H_
#define KALDI_DECODABLE_AM_MFA_H_

#include <vector>

#include "base/kaldi-common.h"
#include "mfa/am-mfa.h"
#include "hmm/transition-model.h"
#include "itf/decodable-itf.h"

namespace kaldi {

class DecodableAmMfa : public DecodableInterface {
 public:
  DecodableAmMfa( const AmMfa &am,
                  const AmMfaPerSpkDerivedVars &spk,  // may be empty
                  const TransitionModel &tm,
                  const Matrix<BaseFloat> &feats,
                  const std::vector<std::vector<int32> > &gselect_all):
                    acoustic_model_(am), spk_(spk), trans_model_(tm),
                    feature_matrix_(feats), gselect_all_(gselect_all),
                    previous_frame_(-1)
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

  const AmMfa &acoustic_model_;
  const AmMfaPerSpkDerivedVars &spk_;
  const TransitionModel &trans_model_;  ///< for tid to pdf mapping
  const Matrix<BaseFloat> &feature_matrix_;
  const std::vector<std::vector<int32> > gselect_all_;  ///< if nonempty,
                                                        ///< precomputed gaussian indices.
  int32 previous_frame_;

  /// Defines a cache record for a state
  struct LikelihoodCacheRecord {
    BaseFloat log_like;  ///< Cache value
    int32 hit_time;     ///< Frame for which this value is relevant
  };

  /// Cached per-frame quantities used in SGMM likelihood computation.
  std::vector<LikelihoodCacheRecord> log_like_cache_;

  /// Cache for per-frame MFA-related computing
  AmMfaDecodingMFACache mfa_cache_;
 private:
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmMfa);
};

class DecodableAmMfaScaled : public DecodableAmMfa {
 public:
  DecodableAmMfaScaled(const AmMfa &am,
                       const AmMfaPerSpkDerivedVars &spk,  // may be empty
                       const TransitionModel &tm,
                       const Matrix<BaseFloat> &feats,
                       const std::vector<std::vector<int32> > &gselect_all,
                       BaseFloat scale)
      : DecodableAmMfa(am, spk, tm, feats, gselect_all),
        scale_(scale) {}

  // Note, frames are numbered from zero but transition-ids from one.
  virtual BaseFloat LogLikelihood(int32 frame, int32 tid) {
    return LogLikelihoodZeroBased(frame, trans_model_.TransitionIdToPdf(tid))
            * scale_;
  }

 private:
  BaseFloat scale_;
  KALDI_DISALLOW_COPY_AND_ASSIGN(DecodableAmMfaScaled);
};


}  // namespace kaldi

#endif  // KALDI_DECODABLE_AM_MFA_H_
