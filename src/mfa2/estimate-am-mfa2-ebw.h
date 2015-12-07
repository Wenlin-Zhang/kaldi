// mfa2/estimate-am-mfa2-ebw.h

// Copyright 2015  Wen-Lin Zhang

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

#ifndef KALDI_MFA_ESTIMATE_AM_MFA2_EBW_H_
#define KALDI_MFA_ESTIMATE_AM_MFA2_EBW_H_ 1

#include <string>
#include <vector>

#include "gmm/model-common.h"
#include "itf/options-itf.h"
#include "mfa2/estimate-am-mfa2.h"

namespace kaldi {


struct EbwAmMfa2Options {
  BaseFloat tau_y;   ///<  Smoothing constant for updates of phone vectors y_{ji}
  BaseFloat lrate_y; ///< Learning rate used in updating y_{ji} -- default 0.5
  BaseFloat tau_Sigma;   ///< Tau value for smoothing covariance-matrices Sigma.
  BaseFloat lrate_Sigma; ///< Learning rate used in updating Sigma-- default 0.5
  BaseFloat tau_w;     ///<  Smoothing constant for the state specific weights
  BaseFloat lrate_w;  ///< Learning rate used in updating the state specific weights -- default 0.5
  
  BaseFloat s0_thresh_; ///< Threshold for updating parameters except for y

  BaseFloat glasso_tau_; ///< L1 Weight for sparse inverse covarianc matrix estimation with graphical lasso

  BaseFloat cov_min_value; ///< E.g. 0.5-- the maximum any eigenvalue of a covariance
  /// is allowed to change.  [this is the minimum; the maximum is the inverse of this,
  /// i.e. 2.0 in this case.  For example, 0.9 would constrain the covariance quite tightly,
  /// 0.1 would be a loose setting.
  
  BaseFloat max_cond; ///< large value used in SolveQuadraticProblem.
  BaseFloat epsilon;  ///< very small value used in SolveQuadraticProblem; workaround
  /// for an issue in some implementations of SVD.
  
  EbwAmMfa2Options() {
    tau_y = 50.0;
    lrate_y = 0.5;
    tau_Sigma = 50.0;
    lrate_Sigma = 0.5;
    tau_w = 50.0;
    lrate_w = 0.5;

    s0_thresh_ = 0.5;

    glasso_tau_ = 2.0;

    cov_min_value = 0.5;
        
    max_cond = 1.0e+05;
    epsilon = 1.0e-40;
  }

  void Register(OptionsItf *po) {
    std::string module = "EbwAmMfa2Options: ";
    po->Register("tau-y", &tau_y, module+
                 "Smoothing constant for phone vector estimation.");
    po->Register("lrate-y", &lrate_y, module+
                 "Learning rate constant for phone vector estimation.");
    po->Register("tau-sigma", &tau_Sigma, module+
                 "Smoothing constant for estimation of within-class covariances (Sigma)");
    po->Register("lrate-sigma", &lrate_Sigma, module+
                 "Constant that controls speed of learning for variances (larger->slower)");
    po->Register("tau-w", &tau_w, module+
                 "Smoothing constant for estimation of the component weights for each state.");
    po->Register("lrate-w", &lrate_w, module+
                  "Learning rate constant for the component weights for each state.");
    po->Register("s0-thresh", &s0_thresh_, module + "the minimal count to estimate the parameters.");
    po->Register("glasso_tau", &glasso_tau_, module + "L1 Weight for sparse inverse covarianc matrix estimation with graphical lasso.");
    po->Register("cov-min-value", &cov_min_value, module+
                 "Minimum value that an eigenvalue of the updated covariance matrix can take, "
                 "relative to its old value (maximum is inverse of this.)");
    po->Register("max-cond", &max_cond, module+
                 "Value used in handling singular matrices during update.");
    po->Register("epsilon", &max_cond, module+
                 "Value used in handling singular matrices during update.");
  }
};


/** \class EbwAmMfa2pdater
 *  Contains the functions needed to update the AmMfa2 acoustic model parameters.
 */
class EbwAmMfa2Updater {
 public:
  explicit EbwAmMfa2Updater(const EbwAmMfa2Options &options):
      options_(options) {}
  
  void Update(const MleAmMfa2Accs &num_accs,
              const MleAmMfa2Accs &den_accs,
              AmMfa2 *model,
              AmMfaUpdateFlagsType flags,
              BaseFloat *auxf_change_out,
              BaseFloat *count_out);
    
 protected:
  // The following two classes relate to multi-core parallelization of some
  // phases of the update.

 private:
  EbwAmMfa2Options options_;

  Vector<double> gamma_j_;  ///< State occupancies

  BaseFloat Update_y(const MleAmMfa2Accs &num_accs,
                  const MleAmMfa2Accs &den_accs,
                  AmMfa2 *model) const;
  
  BaseFloat Update_Sigma(const MleAmMfa2Accs &num_accs,
                    const MleAmMfa2Accs &den_accs,
                    AmMfa2 *model) const;

  BaseFloat Update_w(const MleAmMfa2Accs &num_accs,
                     const MleAmMfa2Accs &den_accs,
                     AmMfa2 *model) const;

  KALDI_DISALLOW_COPY_AND_ASSIGN(EbwAmMfa2Updater);
  EbwAmMfa2Updater() {}  // Prevent unconfigured updater.
};


}  // namespace kaldi


#endif
