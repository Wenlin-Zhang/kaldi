// mfa/estimate-am-mfa.h

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

#ifndef KALDI_ESTIMATE_AM_MFA_H_
#define KALDI_ESTIMATE_AM_MFA_H_ 1

#include <string>
#include <vector>

#include "mfa/am-mfa.h"
#include "estimate-am-mfa-types.h"
#include "gmm/model-common.h"
#include "itf/options-itf.h"

namespace kaldi {

/** \struct MleAmMfaOptions
 *  Configuration variables needed in the AM-MFA estimation process.
 */
struct MleAmMfaOptions {
  MleAmMfaOptions(): max_comp_(-1.0), min_comp_(10), use_l1_(false),
      gpsr_tau_(10.0), s0_thresh_(5.0), weight_method_(2), weight_parm_(0.1)
  {  }

  void Register(OptionsItf *po) {
    std::string module = "MleAmMfaOptions: ";
    po->Register("max-comp", &max_comp_, module + "maximal component for each state.");
    po->Register("min-comp", &min_comp_, module + "minimal component for each state.");
    po->Register("use-l1", &use_l1_, module + "whether use L1 regularization for phone vector estimation.");
    po->Register("tau", &gpsr_tau_, module + "L1 weight for phone vector estimation.");
    po->Register("s0-thresh", &s0_thresh_, module + "the minimal count to estimate the local dimension.");
    po->Register("weight-method", &weight_method_, module + "method for weight estimation.");
    po->Register("weight-parm", &weight_parm_, module + "parameter for weight estimation.");
  }

  int32 max_comp_;      // maximal components for each state
  int32 min_comp_;      // minimal components for each state
  bool use_l1_;         // whether use L1 regularization for phone vector estimation
  BaseFloat gpsr_tau_;  // phone vector estimation L1 weight
  BaseFloat s0_thresh_; // the minimal component occupation, for y_ji, W_i and w_i computation
  int32 weight_method_;    // weight estimation method, 1,2,3 or 4
  BaseFloat weight_parm_;  // parameter for weight estimation
};

/** \class MleAmMfaAccs
 *  Class for the accumulators associated with the MFA parameters
 */
class MleAmMfaAccs {
 public:
  explicit MleAmMfaAccs():
          total_frames_(0.0), total_like_(0.0),
          feature_dim_(0),num_states_(0), num_factors_(0), spk_dim_(0)
  {}

  MleAmMfaAccs(const AmMfa &model): total_frames_(0.0), total_like_(0.0)
  {
    ResizeAccumulators(model);
  }

  ~MleAmMfaAccs()
  {}

  void Read(std::istream &in_stream, bool binary, bool add);
  void Write(std::ostream &out_stream, bool binary) const;

  /// Checks the various accumulators for correct sizes given a model. With
  /// wrong sizes, assertion failure occurs. When the show_properties argument
  /// is set to true, dimensions and presence/absence of the various
  /// accumulators are printed. For use when accumulators are read from file.
  void Check(const AmMfa &model, bool show_properties = true) const;

  /// Resizes the accumulators to the correct sizes given the model. The flags
  /// argument control which accumulators to resize.
  void ResizeAccumulators(const AmMfa &model);

  /// Begin a new speaker, clear the speaker specific temp accumulator
  void BeginSpkr(const AmMfaPerSpkDerivedVars* pVars, AmMfaUpdateFlagsType flags);

  /// Update the speaker statistics
  void CommitSpkr(const AmMfaPerSpkDerivedVars* pVars, AmMfaUpdateFlagsType flags);

  /// Returns likelihood.
  BaseFloat Accumulate(const AmMfa &model,
                       const VectorBase<BaseFloat> &data,
                       const AmMfaPerSpkDerivedVars* pVars,
                       int32 state_index, BaseFloat weight,
                       AmMfaUpdateFlagsType flags, const std::vector<int32>* gselect = NULL);

  /// Returns count accumulated (may differ from posteriors.Sum()
  /// due to weight pruning).
  BaseFloat AccumulateFromPosteriors(const AmMfa &model,
                                     const Vector<BaseFloat> &posteriors,
                                     const VectorBase<BaseFloat> &data,
                                     const AmMfaPerSpkDerivedVars* pVars,
                                     int32 state_index,
                                     AmMfaUpdateFlagsType flags);

  int32 FeatureDim() const { return feature_dim_; }
  int32 SpkrDim() const { return spk_dim_; }
  int32 NumStates() const { return num_states_; }
  int32 NumFactors() const { return num_factors_; }
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

    /// Number of factors
    int32 num_factors_;

    /// Dimensionality of the speaker subspace
    int32 spk_dim_;

    /// Zero-th order statistics; dim = [S][nFA]
    /// \gamma_{ji} = \sum_t \gamma_{ji}(t)
    std::vector<Vector<BaseFloat> > s0_;

    /// First order statistics; dim = [S][nFA][Dim]
    /// s_{ji} = \sum_t \gamma_{ji}(t) x_t)
    std::vector<Matrix<BaseFloat > > s1_;

    /// Second order statistics; dim = [I][Dim][Dim]
    /// S_i = \sum_i \sum_j \gamma_{ji}(t) x_t x_t^T
    std::vector<SpMatrix<BaseFloat> > s2_;

    /// Stat for speaker subspace estimation
    std::vector<Matrix<BaseFloat> > Z_vec_;
    std::vector<SpMatrix<BaseFloat> > R_vec_;

    KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmMfaAccs);
    friend class MleAmMfaUpdater;
    friend class EbwAmMfaUpdater;

 private:
    typedef struct SpkAccsHelper_
    {
      Matrix<BaseFloat> spk_s1_;
      Vector<BaseFloat> spk_s0_;
    }SpkAccsHelper;

    SpkAccsHelper spkAccsHelper_;
};

/** \class MleAmMfaUpdater
 *  Contains the functions needed to update the AM-MFA parameters.
 */
class MleAmMfaUpdater {
 public:
  explicit MleAmMfaUpdater(const MleAmMfaOptions &options)
      : update_options_(options), is_Y_i_computed_(false), is_Q_i_computed_(false),
        is_s0_i_s1_i_computed_(false), is_S_i_S_means_i_computed_(false),
        is_s1_means_i_computed_(false), weight_threshold_(0.0)
  {}

  void Reconfigure(const MleAmMfaOptions &options) {
    update_options_ = options;
  }

  /// Main update function: Computes some overall stats, does parameter updates
  /// and returns the total improvement of the different auxiliary functions.
  double Update(const MleAmMfaAccs &accs, AmMfa *model, AmMfaUpdateFlagsType flags);

  /// Shrink the AmMfa model, remove the zero weights and the corresponding factors for each state
  static void ShrinkAmMfa(AmMfa* model, BaseFloat minW = 1.0e-9);
  /// Shrink the model by MFA posteriors sum vectors
  static void ShrinkAmMfa(AmMfa* model, const Matrix<BaseFloat>& mfa_post_sum_mat, const int32 maxComp);

  static void Compute_Y_i_s(const MleAmMfaAccs &accs, const AmMfa& model, std::vector<Matrix<BaseFloat> >* p_Y_i_vec);
  static void Compute_Q_i_s(const MleAmMfaAccs &accs, const AmMfa& model, std::vector<SpMatrix<BaseFloat> >* p_Q_i_vec);
  static void Compute_s0_i_s1_i_s(const MleAmMfaAccs &accs, const AmMfa& model, Vector<BaseFloat>* p_s0_i, Matrix<BaseFloat>* p_s1_i);
  static void Compute_s1_means_i_s(const MleAmMfaAccs &accs, const AmMfa& model, Matrix<BaseFloat>* p_s1_means_i);
  static void Compute_S_i_S_means_i_s(const MleAmMfaAccs &accs, const AmMfa& model, const Vector<BaseFloat>& s0_i, const Matrix<BaseFloat>& s1_i,
               std::vector<SpMatrix<BaseFloat> >* p_S_i_vec, std::vector<SpMatrix<BaseFloat> >* p_S_means_i_vec);

 protected:


 private:
  MleAmMfaOptions update_options_;

  /// kAmMfaPhoneVectors
  double UpdatePhoneVectors(const MleAmMfaAccs &accs, AmMfa *model);

  /// kAmMfaPhoneProjections
  double UpdatePhoneProjections(const MleAmMfaAccs &accs, AmMfa *model);

  /// kAmMfaPhoneWeights
  double UpdatePhoneWeights(const MleAmMfaAccs &accs, AmMfa *model);

  /// kAmMfaCovarianceMatrix
  double UpdateCovarianceMatrix(const MleAmMfaAccs &accs, AmMfa *model);

  /// kAmMfaFAMeans
  double UpdateMFAMeans(const MleAmMfaAccs &accs, AmMfa *model);

  /// kAmMfaSpeakerProjections
  double UpdateSpeakerProjections(const MleAmMfaAccs &accs, AmMfa *model);

  /// Y_i statistics, shared by UpdatePhoneWeights and UpdateCovarianceMatrix
  std::vector<Matrix<BaseFloat> > Y_i_vec_;
  bool is_Y_i_computed_;
  void Compute_Y_i(const MleAmMfaAccs &accs, const AmMfa& model);
  void Free_Y_i();

  /// Q_i statistics, needed by UpdatePhoneWeights
  std::vector<SpMatrix<BaseFloat> > Q_i_vec_;
  bool is_Q_i_computed_;
  void Compute_Q_i(const MleAmMfaAccs &accs, const AmMfa& model);
  void Free_Q_i();

  /// s0_i, s1_i, need by UpdateCovarianceMatrix
  Vector<BaseFloat> s0_i_;
  Matrix<BaseFloat> s1_i_;
  bool is_s0_i_s1_i_computed_;
  void Compute_s0_i_s1_i(const MleAmMfaAccs &accs, const AmMfa& model);
  void Free_s0_i_s1_i();

  /// S_i and S_means_i, needed by UpdateCovarianceMatrix
  std::vector<SpMatrix<BaseFloat> > S_i_vec_;
  std::vector<SpMatrix<BaseFloat> > S_means_i_vec_;
  bool is_S_i_S_means_i_computed_;
  void Compute_S_i_S_means_i(const MleAmMfaAccs &accs, const AmMfa& model);
  void Free_S_i_S_means_i();

  /// mu_mean_i, needed by UpdateFaMeans
  Matrix<BaseFloat> s1_means_i_;
  bool is_s1_means_i_computed_;
  void Compute_s1_means_i(const MleAmMfaAccs &accs, const AmMfa& model);
  void Free_s1_means_i();

  /// weight shrink threshold
  BaseFloat weight_threshold_;

  KALDI_DISALLOW_COPY_AND_ASSIGN(MleAmMfaUpdater);

  MleAmMfaUpdater() {}  // Prevent unconfigured updater.
};

/** \class MleAmMfaSpeakerAccs
 *  Class for the accumulators required to update the speaker
 *  vectors v_s.
 *  Note: if you have multiple speakers you will want to initialize
 *  this just once and call Clear() after you're done with each speaker,
 *  rather than creating a new object for each speaker, since the
 *  initialization function does nontrivial work.
 */

class MleAmMfaSpeakerAccs {
 public:
  /// Initialize the object.  Error if speaker subspace not set up.
  MleAmMfaSpeakerAccs(const AmMfa &model);

  /// Clear the statistics.
  void Clear();

  /// Accumulate statistics.  Returns per-frame log-likelihood.
  BaseFloat Accumulate(const AmMfa &model,
                       const VectorBase<BaseFloat> &data,
                       const AmMfaPerSpkDerivedVars* pVars,
                       int32 state_index, BaseFloat weight, const std::vector<int32>* gselect = NULL);

  /// Accumulate statistics, given posteriors.  Returns total
  /// count accumulated, which may differ from posteriors.Sum()
  /// due to randomized pruning.
  BaseFloat AccumulateFromPosteriors(const AmMfa &model,
                                     const VectorBase<BaseFloat> &data,
                                     const AmMfaPerSpkDerivedVars* pVars,
                                     const Vector<BaseFloat> &posteriors,
                                     int32 state_index);

  /// Update speaker vector.  If v_s was empty, will assume it started as zero
  /// and will resize it to the speaker-subspace size.
  void Update(BaseFloat min_count,  // e.g. 100
              Vector<BaseFloat> *v_s,
              BaseFloat *objf_impr_out,
              BaseFloat *count_out);

 private:
  /// Statistics for speaker adaptation (vectors), stored per-speaker.
  /// Per-speaker stats for vectors, y^{(s)}. Dimension [K].
  Vector<BaseFloat> y_s_;
  /// gamma_{i}^{(s)}.  Per-speaker counts for each Gaussian. Dimension is [I]
  Vector<BaseFloat> gamma_s_;

  /// The following variable does not change per speaker.
  /// Eq. (82): H_{i}^{spk} = N_{i}^T \Sigma_{i}^{-1} N_{i}
  std::vector< SpMatrix<BaseFloat> > H_spk_;

  /// N_i^T \Sigma_{i}^{-1}. Needed for y^{(s)}
  std::vector< Matrix<BaseFloat> > NtransSigmaInv_;

};

}

#endif
