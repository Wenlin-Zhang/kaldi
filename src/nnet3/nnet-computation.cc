// Copyright      2015  Johns Hopkins University (author: Daniel Povey)

// See ../../COPYING for clarification regarding multiple authors
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

#include <iterator>
#include <sstream>
#include "nnet3/nnet-computation.h"

namespace kaldi {
namespace nnet3 {

bool ComputationRequest::NeedDerivatives() const {
  if (model_to_update != NULL) return true;
  for (size_t i = 0; i < inputs.size(); i++)
    if (inputs[i].has_deriv)  // derivative requested for this input
      return true;
  return false;
}

NnetComputation::~NnetComputation() {
  for (size_t i = 0; i < component_precomputed_indexes.size(); i++)
    delete component_precomputed_indexes[i];
}

} // namespace nnet3
} // namespace kaldi