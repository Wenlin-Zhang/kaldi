// mfa/estimate-am-mfa-ebw.h

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

#ifndef KALDI_MFA_ESTIMATE_AM_MFA_EBW_H_
#define KALDI_MFA_ESTIMATE_AM_MFA_EBW_H_ 1

#include <string>
#include <vector>

#include "gmm/model-common.h"
#include "itf/options-itf.h"
#include "mfa/estimate-am-mfa.h"

namespace kaldi {

/**
   This header implements a form of Extended Baum-Welch training for SGMMs.
   If you are confused by this comment, see Dan Povey's thesis for an explanation of
   Extended Baum-Welch.
   A note on the EBW (Extended Baum-Welch) updates for the SGMMs... In general there is
   a parameter-specific value D that is similar to the D in EBW for GMMs.  The value of
   D is generally set to:
     E * (denominator-count for that parameter)   +   tau-value for that parameter
   where the tau-values are user-specified parameters that are specific to the type of
   the parameter (e.g. phonetic vector, subspace projection, etc.).  Things are a bit
   more complex for this update than for GMMs, because it's not just a question of picking
   a tau-value for smoothing: there is sometimes a scatter-matrix of some kind (e.g.
   an outer product of vectors, or something) that defines a quadratic objective function
   that we'll add as smoothing.  We have to pick where to get this scatter-matrix from.
   We feel that it's appropriate for the "E" part of the D to get its scatter-matrix from
   denominator stats, and the tau part of the D to get half its scatter-matrix from the
   both the numerator and denominator stats, assigned a weight proportional to how much
   stats there were.  When you see the auxiliary function written out, it's clear why this
   makes sense.

 */

struct EbwAmMfaOptions {
  BaseFloat tau_y;   ///<  Smoothing constant for updates of phone vectors y_{ji}
  BaseFloat lrate_y; ///< Learning rate used in updating y_{ji} -- default 0.5
  BaseFloat tau_M;   ///<  Smoothing constant for the M quantities (phone-subspace projections)
  BaseFloat lrate_M; ///< Learning rate used in updating M-- default 0.5
  BaseFloat tau_N;   ///<  Smoothing constant for the N quantities (speaker-subspace projections)
  BaseFloat lrate_N; ///< Learning rate used in updating N-- default 0.5
  BaseFloat tau_Sigma;   ///< Tau value for smoothing covariance-matrices Sigma.
  BaseFloat lrate_Sigma; ///< Learning rate used in updating Sigma-- default 0.5
  BaseFloat tau_mu;   ///<  Smoothing constant for the MFA means
  BaseFloat lrate_mu; ///< Learning rate used in updating the MFA means -- default 0.5
  BaseFloat tau_w;
  BaseFloat lrate_w;
  
  BaseFloat s0_thresh_; ///< Threshold for updating parameters except for y
  //BaseFloat s0_y_thresh_; ///< Threshold for updating  y

  BaseFloat cov_min_value; ///< E.g. 0.5-- the maximum any eigenvalue of a covariance
  /// is allowed to change.  [this is the minimum; the maximum is the inverse of this,
  /// i.e. 2.0 in this case.  For example, 0.9 would constrain the covariance quite tightly,
  /// 0.1 would be a loose setting.
  
  BaseFloat max_cond; ///< large value used in SolveQuadraticProblem.
  BaseFloat epsilon;  ///< very small value used in SolveQuadraticProblem; workaround
  /// for an issue in some implementations of SVD.
  
  EbwAmMfaOptions() {
    tau_y = 50.0;
    lrate_y = 0.5;
    tau_M = 500.0;
    lrate_M = 0.5;
    tau_N = 500.0;
    lrate_N = 0.5;
    tau_Sigma = 500.0;
    lrate_Sigma = 0.5;
    tau_mu = 500.0;
    lrate_mu = 0.5;
    tau_w = 50.0;
    lrate_w = 0.5;

    s0_thresh_ = 0.5;

    cov_min_value = 0.5;
        
    max_cond = 1.0e+05;
    epsilon = 1.0e-40;
  }

  void Register(OptionsItf *po) {
    std::string module = "EbwAmMfaOptions: ";
    po->Register("tau-y", &tau_y, module+
                 "Smoothing constant for phone vector estimation.");
    po->Register("lrate-y", &lrate_y, module+
                 "Learning rate constant for phone vector estimation.");
    po->Register("tau-m", &tau_M, module+
                 "Smoothing constant for estimation of phonetic-subspace projections (M).");
    po->Register("lrate-m", &lrate_M, module+
                 "Learning rate constant for phonetic-subspace projections.");
    po->Register("tau-n", &tau_N, module+
                 "Smoothing constant for estimation of speaker-subspace projections (N).");
    po->Register("lrate-n", &lrate_N, module+
                 "Learning rate constant for speaker-subspace projections.");
    po->Register("tau-sigma", &tau_Sigma, module+
                 "Smoothing constant for estimation of within-class covariances (Sigma)");
    po->Register("lrate-sigma", &lrate_Sigma, module+
                 "Constant that controls speed of learning for variances (larger->slower)");
    po->Register("tau-mu", &tau_mu, module+
                 "Smoothing constant for estimation of the MFA means (mu).");
    po->Register("lrate-mu", &lrate_mu, module+
                 "Learning rate constant for  the MFA means (mu).");
    po->Register("tau-w", &tau_w, module+
                 "Smoothing constant for estimation of the component weights for each state.");
    po->Register("lrate-w", &lrate_w, module+
                  "Learning rate constant for the component weights for each state.");
    po->Register("s0-thresh", &s0_thresh_, module + "the minimal count to estimate the parameters.");
    po->Register("cov-min-value", &cov_min_value, module+
                 "Minimum value that an eigenvalue of the updated covariance matrix can take, "
                 "relative to its old value (maximum is inverse of this.)");
    po->Register("max-cond", &max_cond, module+
                 "Value used in handling singular matrices during update.");
    po->Register("epsilon", &max_cond, module+
                 "Value used in handling singular matrices during update.");
  }
};


/** \class EbwAmMfaUpdater
 *  Contains the functions needed to update the MFA acoustic model parameters.
 */
class EbwAmMfaUpdater {
 public:
  explicit EbwAmMfaUpdater(const EbwAmMfaOptions &options):
      options_(options) {}
  
  void Update(const MleAmMfaAccs &num_accs,
              const MleAmMfaAccs &den_accs,
              AmMfa *model,
              AmMfaUpdateFlagsType flags,
              BaseFloat *auxf_change_out,
              BaseFloat *count_out);
    
 protected:
  // The following two classes relate to multi-core parallelization of some
  // phases of the update.

 private:
  EbwAmMfaOptions options_;

  Vector<double> gamma_j_;  ///< State occupancies

  BaseFloat Update_y(const MleAmMfaAccs &num_accs,
                  const MleAmMfaAccs &den_accs,
                  AmMfa *model) const;
                                    
  BaseFloat Update_M(const MleAmMfaAccs &num_accs,
                  const MleAmMfaAccs &den_accs,
                  const Vector<BaseFloat> &s0_i_num,
                  const Vector<BaseFloat> &s0_i_den,
                  const std::vector< SpMatrix<BaseFloat> > &Q_i_vec_num,
                  const std::vector< SpMatrix<BaseFloat> > &Q_i_vec_den,
                  const std::vector<Matrix<BaseFloat> > &Y_i_vec_num,
                  const std::vector<Matrix<BaseFloat> > &Y_i_vec_den,
                  AmMfa *model) const;
  
  BaseFloat Update_Mu(const Vector<BaseFloat> &gamma_num,
                 const Vector<BaseFloat> &gamma_den,
                 const Matrix<BaseFloat> & s1_mean,
                 AmMfa *model) const;
  
  BaseFloat Update_Sigma(const MleAmMfaAccs &num_accs,
                    const MleAmMfaAccs &den_accs,
                    const Vector<BaseFloat> &gamma_num,
                    const Vector<BaseFloat> &gamma_den,
                    const std::vector< SpMatrix<BaseFloat> > &S_means,
                    AmMfa *model) const;

  BaseFloat Update_N(const MleAmMfaAccs &num_accs,
                    const MleAmMfaAccs &den_accs,
                    const Vector<BaseFloat> &s0_i_num,
                    const Vector<BaseFloat> &s0_i_den,
                    AmMfa *model) const;

  BaseFloat Update_w(const MleAmMfaAccs &num_accs,
                     const MleAmMfaAccs &den_accs,
                     AmMfa *model) const;

  KALDI_DISALLOW_COPY_AND_ASSIGN(EbwAmMfaUpdater);
  EbwAmMfaUpdater() {}  // Prevent unconfigured updater.
};


}  // namespace kaldi


#endif  // KALDI_SGMM_ESTIMATE_AM_SGMM_EBW_H_
