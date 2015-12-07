// mfa2/decodable-am-mfa2.cc

// Copyright 2015 Wen-Lin Zhang

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

#include "mfa2/decodable-am-mfa2.h"

namespace kaldi {

BaseFloat DecodableAmMfa2::LogLikelihoodZeroBased(int32 frame, int32 pdf_id) {
  KALDI_ASSERT(frame >= 0 && frame < NumFrames());
  KALDI_ASSERT(pdf_id >= 0 && pdf_id < NumIndices());

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
	  loglike = acoustic_model_.LogLikelihood(pdf_id, data, NULL, NULL);
  }
  else {
    KALDI_ASSERT(frame < gselect_all_.size());
    loglike = acoustic_model_.LogLikelihood(pdf_id, data, NULL, &(gselect_all_[frame]));
  }

  if (KALDI_ISNAN(loglike) || KALDI_ISINF(loglike))
    KALDI_ERR << "Invalid answer (overflow or invalid variances/features?)";
  log_like_cache_[pdf_id].log_like = loglike;
  log_like_cache_[pdf_id].hit_time = frame;
  return loglike;
}

void DecodableAmMfa2::ResetLogLikeCache() {
  if (log_like_cache_.size() != acoustic_model_.NumStates()) {
    log_like_cache_.resize(acoustic_model_.NumStates());
  }
  vector<LikelihoodCacheRecord>::iterator it = log_like_cache_.begin(),
      end = log_like_cache_.end();
  for (; it != end; ++it) { it->hit_time = -1; }
}

}  // namespace kaldi
