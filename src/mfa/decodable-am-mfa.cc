// mfa/decodable-am-mfa.cc

// Copyright 2013 Wen-Lin Zhang

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

#include <vector>
using std::vector;

#include "mfa/decodable-am-mfa.h"

namespace kaldi {

BaseFloat DecodableAmMfa::LogLikelihoodZeroBased(int32 frame, int32 pdf_id) {
  KALDI_ASSERT(frame >= 0 && frame < NumFrames());
  KALDI_ASSERT(pdf_id >= 0 && pdf_id < NumIndices());

  mfa_cache_.SetFrame(frame);

  if (log_like_cache_[pdf_id].hit_time == frame) {
    return log_like_cache_[pdf_id].log_like;  // return cached value, if found
  }

  const VectorBase<BaseFloat> &data = feature_matrix_.Row(frame);
  // check if everything is in order
  if (acoustic_model_.FeatureDim() != data.Dim()) {
    KALDI_ERR << "Dim mismatch: data dim = "  << data.Dim()
        << "vs. model dim = " << acoustic_model_.FeatureDim();
  }

  BaseFloat loglike = 0.0;
  if (gselect_all_.empty())
  {
    if (acoustic_model_.IsPreSelGauss() == false)
      loglike = acoustic_model_.LogLikelihood(pdf_id, data, &spk_, NULL, NULL, &mfa_cache_);
    else
    {
      std::vector<int32> gselect;
      loglike = acoustic_model_.GaussianSelection(data, &gselect);
    }
  }
  else {
    KALDI_ASSERT(frame < gselect_all_.size());
    loglike = acoustic_model_.LogLikelihood(pdf_id, data, &spk_, NULL, &(gselect_all_[frame]), &mfa_cache_);
  }

  if (KALDI_ISNAN(loglike) || KALDI_ISINF(loglike))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  log_like_cache_[pdf_id].log_like = loglike;
  log_like_cache_[pdf_id].hit_time = frame;
  return loglike;
}

void DecodableAmMfa::ResetLogLikeCache() {
  if (log_like_cache_.size() != acoustic_model_.NumStates()) {
    log_like_cache_.resize(acoustic_model_.NumStates());
  }
  vector<LikelihoodCacheRecord>::iterator it = log_like_cache_.begin(),
      end = log_like_cache_.end();
  for (; it != end; ++it) { it->hit_time = -1; }
  mfa_cache_.ResetCache();
}

}  // namespace kaldi
