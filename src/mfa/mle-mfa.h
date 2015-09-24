// mfa/mle-mfa.h

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


#ifndef KALDI_MLE_MFA_H_
#define KALDI_MLE_MFA_H_ 1

#include <vector>

#include "mfa/mfa.h"
#include "util/common-utils.h"
#include "itf/options-itf.h"

namespace kaldi {

class AccumMFA
{
public:

  AccumMFA();

  AccumMFA(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec);

  AccumMFA(const MFA& mfa);

  ~AccumMFA();

  void Resize(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec);

  void Clear();

  int32 NumGauss() { return num_comp_; }

  int32 Dim() { return dim_; }

  std::vector<std::size_t> LocalDims() { return k_vec_; }

  int32 GetLocalDim(std::size_t i)
  {
    if (i < 0 || i > k_vec_.size())
    {
      KALDI_ERR << "AccumMFA:: GetLocalDim invalid parameter (" << i << ").";
    }
    return k_vec_[i];
  }

  void Read(std::istream &in_stream, bool binary, bool add);

  void Write(std::ostream &out_stream, bool binary) const;

  /// Check the compatibility
  bool Check(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec);

  /// Accumulate one observation, return the log likelihood
  float AccumulateForObservation(MFA& mfa, const VectorBase<BaseFloat> &data, BaseFloat weight = 1.0);

  /// for computing the maximum-likelihood estimates of the parameters of
  /// a MoFA model.
  void MleMFAUpdate(MFA *pMFA, bool deleteLowOccComp = false, BaseFloat minOcc = 10);

protected:
  /**
   * fa_stat is an accumulator of sufficient statistics needed to
   * recompute parameters for factor analysis and other per-factor
   * information needed in the training.
   */
  struct fa_stat {

    /**
     * Constructor.
     *
     * @param d the dimensionality of the observed continuous
     *          variable \f$X\f$
     * @param k the dimensionality of the latent continuous
     *          variable \f$Y\f$
     */
    fa_stat(std::size_t d, std::size_t k)
    {
      s0_ = 0.0;
      W1_.Resize(d, k + 1);
      W2_.Resize(k + 1);
      sigma_.Resize(d);
    }

    //! Destructor.
    ~fa_stat()
    {

    }

    /**
     *  accumulate posteriors
     */
    BaseFloat s0_;

    /**
     * An \f$d \times k\f$ matrix to store the accumulator for \f$W\f$
     *
     * \f$W_1 = \sum(\langle yx^\top \rangle  -
     * \mu\langle x\rangle^\top )\f$
     */
     Matrix<BaseFloat> W1_;

     /**
      * An \f$m \times m\f$ matrix to store the accumulator for \f$W\f$
      *
      * \f$W_2 = \sum(\langle xx^\top \rangle )\f$
      */
     SpMatrix<BaseFloat> W2_;

     /**
      * An \f$n \times 1\f$ accumulator for \f$sigma\f$
      */
     Vector<BaseFloat> sigma_;
    };

    // statistics vector
    std::vector<fa_stat*> fa_stat_vec_;

    // dimension of the observation data
    std::size_t dim_;

    // number of factor components
    std::size_t num_comp_;

    // vector of the dimensions of the local subspaces
    std::vector<std::size_t> k_vec_;

    // observation count
    BaseFloat count_;

    // accumulated log likelihood
    BaseFloat llk_;

};

}

#endif
