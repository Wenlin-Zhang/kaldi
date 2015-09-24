// mfa/estimate-am-mfa-types.h

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

#ifndef KALDI_ESTIMATE_AM_MFA_TYPES_H_
#define KALDI_ESTIMATE_AM_MFA_TYPES_H_ 1

#include <string>
#include <vector>

#include "gmm/model-common.h"
#include "util/parse-options.h"

namespace kaldi {

enum AmMfaUpdateFlags {  /// The letters correspond to the variable names.
  kAmMfaPhoneVectors = 0x001,  /// y
  kAmMfaPhoneProjections = 0x002,  /// M
  kAmMfaPhoneWeights = 0x004,  /// w
  kAmMfaPhoneInvCovMatrix = 0x008,  /// R
  kAmMfaCovarianceMatrix = 0x010,  /// S
  kAmMfaFAMeans = 0x020,  /// m
  kAmMfaTransitions = 0x040,  /// t .. not really part of AmMfa.
  kAmMfaSpeakerProjections = 0x080,  /// N
  kAmMfaRemoveSpkrProjections = 0x100,  /// U .. a flag that removes the speaker vector stat.
  kAmMfaAll = 0x0FF   /// a (won't normally use this).
};

typedef uint16 AmMfaUpdateFlagsType;  ///< Bitwise OR of the above flags.
inline AmMfaUpdateFlagsType StringToAmMfaUpdateFlags(std::string str)
{
  AmMfaUpdateFlagsType flags = 0;
  for (const char *c = str.c_str(); *c != '\0'; c++) {
    switch (*c) {
      case 'y': flags |= kAmMfaPhoneVectors; break;
      case 'M': flags |= kAmMfaPhoneProjections; break;
      case 'w': flags |= kAmMfaPhoneWeights; break;
      case 'R': flags |= kAmMfaPhoneInvCovMatrix; break;
      case 'S': flags |= kAmMfaCovarianceMatrix; break;
      case 't': flags |= kAmMfaTransitions; break;
      case 'm': flags |= kAmMfaFAMeans; break;
      case 'N': flags |= kAmMfaSpeakerProjections; break;
      case 'U': flags |= kAmMfaRemoveSpkrProjections; break;
      case 'a': flags |= kAmMfaAll; break;
      default:
        KALDI_ERR << "Invalid element " << CharToString(*c)
                  << " of AmMfaUpdateFlagsType option string "
                  << str;
        break;
    }
  }
  return flags;
}

/// The method for weight update
enum AmMfaWeightUpdateMethodFlags {
  kAmMfaWeightDirect     = 0,    // Direct estimation of the weight
  kAmMfaWeightShrinkHard = 1,    // prune component if low occupation ( < s0_thresh)
  kAmMfaWeightShrinkAver = 2,    // prune component if below a minimal occupation ( < param / I )
  kAmMfaWeightShrinkSoft = 3,    // prune component if soft low occupation ( < param * 100% acc weight)
  kAmMfaWeightFloor      = 4,    // fixed weight count and floor the weight
  kAmMfaWeightFactor     = 5,    // prune component by a factor (param)
  kAmMfaWeightShrinkAver2= 6,    // prune component if below a minimal occupation ( < param / count)
  kAmMfaWeightShrinkThreshold = 7, // prune component using a fixed threshold (e.g. 1.0e-5)
};

}

#endif
