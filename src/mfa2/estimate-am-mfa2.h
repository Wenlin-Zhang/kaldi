// mfa2/estimate-am-mfa2.h

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

#ifndef KALDI_ESTIMATE_AM_MFA2_H_
#define KALDI_ESTIMATE_AM_MFA2_H_ 1

#include <string>
#include <vector>

#include "mfa2/am-mfa2.h"
#include "mfa/estimate-am-mfa-types.h"
#include "gmm/model-common.h"
#include "itf/options-itf.h"

namespace kaldi {

/** \struct MleAmMfa2Options
 *  Configuration variables needed in the AM-MFA2 estimation process.
 */
struct MleAmMfa2Options {
  MleAmMfa2Options(): max_comp_(-1.0), min_comp_(10), use_l1_(false),
      gpsr_tau_(10.0), glasso_tau_(2.0), s0_thresh_(5.0), min_cov_ratio_(2.0),  weight_method_(2), weight_parm_(0.1)
  {  }

  void Register(OptionsItf *po) {
    std::string module = "MleAmMfa2Options: ";
    po->Register("max-comp", &max_comp_, module + "maximal component for each state.");
    po->Register("min-comp", &min_comp_, module + "minimal component for each state.");
    po->Register("use-l1", &use_l1_, module + "whether use L1 regularization for phone vector estimation.");
    po->Register("tau", &gpsr_tau_, module + "L1 weight for phone vector estimation.");
    po->Register("glasso-tau", &glasso_tau_, module + "L1 weight for sparse inverse covariance estimation with graphical lasso.");
    po->Register("s0-thresh", &s0_thresh_, module + "the minimal count to estimate the local dimension.");
    po->Register("min-cov-ratio", &min_cov_ratio_, module + "the minimal count to estimate the local covariance matrix.");
    po->Register("weight-method", &weight_method_, module + "method for weight estimation.");
    po->Register("weight-parm", &weight_parm_, module + "parameter for weight estimation.");
  }

  int32 max_comp_;      // maximal components for each state
  int32 min_comp_;      // minimal components for each state
  bool use_l1_;         // whether use L1 regularization for phone vector estimation
  BaseFloat gpsr_tau_;  // phone vector estimation L1 weight
  BaseFloat glasso_tau_; // L1 weight for sparse inverse covariance estimation with graphical lasso
  BaseFloat s0_thresh_; // the minimal component occupation for state specific parameter update
  BaseFloat min_cov_ratio_; // if the component occupation < min_cov_ratio * dim, use global covariance matrix
  int32 weight_method_;    // weight estimation method, 1,2,3 or 4
  BaseFloat weight_parm_;  // parameter for weight estimation
};

/** \class MleAmMfa2Accs
 *  Class for the accumulators associated with the AmMfa parameters
 */
class MleAmMfa2Accs {
 public:
  explicit MleAmMfa2Accs():
          total_frames_(0.0), total_like_(0.0),
          feature_dim_(0),num_states_(0)
  {}

  MleAmMfa2Accs(const AmMfa2 &model): total_frames_(0.0), total_like_(0.0)
  {
    ResizeAccumulators(model);
  }

  ~MleAmMfa2Accs()
  {}

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Checks the various accumulators for correct sizes given a model. With
  /// wrong sizes, assertion failure occurs. When the show_properties argument
  /// is set to true, dimensions and presence/absence of the various
  /// accumulators are printed. For use when accumulators are read from file.
  void Check(const AmMfa2 &model, bool show_properties = true) const;

  /// Resizes the accumulators to the correct sizes given the model. The flags
  /// argument control which accumulators to resize.
  void ResizeAccumulators(const AmMfa2 &model);

  /// Returns likelihood.
  BaseFloat Accumulate(const AmMfa2 &model,
                       const VectorBase<BaseFloat> &data,
                       int32 state_index, BaseFloat weight,
                       AmMfaUpdateFlagsType flags, const std::vector<int32>* gselect = NULL);

  /// Returns count accumulated (may differ from posteriors.Sum()
  /// due to weight pruning).
  BaseFloat AccumulateFromPosteriors(const AmMfa2 &model,
                                     const Vector<BaseFloat> &posteriors,
                                     const VectorBase<BaseFloat> &data,
                                     int32 state_index,
                                     AmMfaUpdateFlagsType flags);

  int32 FeatureDim() const { return feature_dim_; }
  int32 NumStates() const { return num_states_; }
  double TotalFrames() const { return total_frames_; }
  double TotalLike() const { return total_like_; }
  BaseFloat StateOcc(int j)const
  {
    KALDI_ASSERT(j >= 0 && j < num_states_);
    return s0_[j].Sum();
  }

  const Vector<BaseFloat>& occs(int32 j)const {
    KALDI_ASSERT(j >= 0 && j < num_states_);
    return s0_[j];
  }

  const Matrix<BaseFloat>& mean_accs(int32 j)const {
    KALDI_ASSERT(j >= 0 && j < num_states_);
    return s1_[j];
  }

 private:
    double total_frames_, total_like_;

    /// Dimensionality of various subspaces
    int32 feature_dim_;

    /// Number of state
    int32 num_states_;

    /// Zero-th order statistics; dim = [S][nFA]
    /// \gamma_{ji} = \sum_t \gamma_{ji}(t)
    std::vector<Vector<BaseFloat> > s0_;

    /// First order statistics; dim = [S][nFA][Dim]
    /// s_{ji} = \sum_t \gamma_{ji}(t) x_t)
    std::vector<Matrix<BaseFloat > > s1_;

    /// Second order statistics; dim = [S][nFA][Dim][Dim]
    /// S_i = \sum_i \sum_j \gamma_{ji}(t) x_t x_t^T
    std::vector<std::vector<SpMatrix<BaseFloat> > > s2_;

    KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmMfa2Accs);
    friend class MleAmMfa2Updater;
    friend class EbwAmMfa2Updater;
};

/** \class MleAmMfa2Updater
 *  Contains the functions needed to update the AM-MFA2 parameters.
 */
class MleAmMfa2Updater {
 public:
  explicit MleAmMfa2Updater(const MleAmMfa2Options &options)
      : update_options_(options), weight_threshold_(0.0)
  {}

  void Reconfigure(const MleAmMfa2Options &options) {
    update_options_ = options;
  }

  /// Main update function: Computes some overall stats, does parameter updates
  /// and returns the total improvement of the different auxiliary functions.
  double Update(const MleAmMfa2Accs &accs, AmMfa2 *model, AmMfaUpdateFlagsType flags);

  /// Shrink the AmMfa model, remove the zero weights and the corresponding factors for each state
  static void ShrinkAmMfa2(AmMfa2* model, BaseFloat minW = 1.0e-9);
  /// Shrink the model by MFA posteriors sum vectors
  static void ShrinkAmMfa2(AmMfa2* model, const Matrix<BaseFloat>& mfa_post_sum_mat, const int32 maxComp);

 protected:


 private:
  MleAmMfa2Options update_options_;

  /// kAmMfaPhoneVectors
  double UpdatePhoneVectors(const MleAmMfa2Accs &accs, AmMfa2 *model);

  /// kAmMfaPhoneWeights
  double UpdatePhoneWeights(const MleAmMfa2Accs &accs, AmMfa2 *model);

  /// kAmMfaCovarianceMatrix
  double UpdateCovarianceMatrix(const MleAmMfa2Accs &accs, AmMfa2 *model);

  /// weight shrink threshold
  BaseFloat weight_threshold_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmMfa2Updater);

  MleAmMfa2Updater() {}  // Prevent unconfigured updater.
};


}

#endif
