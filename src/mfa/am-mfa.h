// mfa/am-mfa.h

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

#ifndef KALDI_AM_MFA_H_
#define KALDI_AM_MFA_H_ 1

#include <vector>
#include <queue>
#include <map>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "gmm/model-common.h"
#include "gmm/diag-gmm.h"
#include "gmm/full-gmm.h"
#include "mfa/mfa.h"
#include "util/table-types.h"

namespace kaldi {

class AmDiagGmm;

struct AmMfaPerSpkDerivedVars {
  // To set this up, call ComputePerSpkDerivedVars from the AmMfa object.
  void Clear() {
    v_s.Resize(0);
    o_s.Resize(0, 0);
  }
  Vector<BaseFloat> v_s;  ///< Speaker adaptation vector v_^{(s)}. Dim is [T]
  Matrix<BaseFloat> o_s;  ///< Per-speaker offsets o_{i}. Dimension is [I][D]
};

struct AmMfaGselectConfig {
  /// Number of highest-scoring full-covariance Gaussians per frame.
  int32 full_gmm_nbest;
  /// Number of highest-scoring diagonal-covariance Gaussians per frame.
  int32 diag_gmm_nbest;

  AmMfaGselectConfig() {
    full_gmm_nbest = 15;
    diag_gmm_nbest = 50;
  }

  AmMfaGselectConfig(const AmMfaGselectConfig& opt) {
    full_gmm_nbest = opt.full_gmm_nbest;
    diag_gmm_nbest = opt.diag_gmm_nbest;
  }

  void Register(OptionsItf *po) {
    po->Register("full-gmm-nbest", &full_gmm_nbest, "Number of highest-scoring"
        " full-covariance Gaussians selected per frame.");
    po->Register("diag-gmm-nbest", &diag_gmm_nbest, "Number of highest-scoring"
        " diagonal-covariance Gaussians selected per frame.");
  }
};

struct AmMfaGselectDirectConfig {
  int32 isPreSelectGaussDirect_;
  int32 pruneCompCnt_;
  int32 maxCompCnt_;

  AmMfaGselectDirectConfig(): isPreSelectGaussDirect_(false), pruneCompCnt_(40), maxCompCnt_(20) {
  }

  void Register(OptionsItf* po) {
    po->Register("ammfa-gselect-direct", &isPreSelectGaussDirect_, "Prune Gaussians of each state using direct method");
    po->Register("ammfa-prune-comp", &pruneCompCnt_, "Prune threshold for AmMfa likelihood computation");
    po->Register("ammfa-max-comp", &maxCompCnt_, "Maximum Gaussian component count for AmMfa likelihood computation");
  }
};

struct AmMfaDecodingMFACache
{
  AmMfaDecodingMFACache(): previous_frame_(-1)
  {
  }

  ~AmMfaDecodingMFACache()
  {
    ResetCache();
  }

  /// Must be called every frame
  void ResetCache()
  {
    decoding_cache_.clear();
  }
  void SetFrame(int32 current_frame)
  {
    if (current_frame != previous_frame_)
    {
      if (current_frame < previous_frame_)
        KALDI_WARN << "AmMfaDecodingMFACache::SetFrame: current = " << current_frame
                   << " previous = " << previous_frame_ << ".";
      ResetCache();
      previous_frame_ = current_frame;
    }
  }
  bool FindCache(int32 m, BaseFloat* llk)
  {
    std::map<int32, BaseFloat>::const_iterator it = decoding_cache_.find(m);
    if (it == decoding_cache_.end())
      return false;
    else
    {
      *llk = it->second;
      return true;
    }
  }
  void InsertCache(int32 i, BaseFloat llk)
  {
    decoding_cache_.insert(std::make_pair(i, llk));
  }

  // Following is the decoding cache
  std::map<int32, BaseFloat> decoding_cache_;
  int32 previous_frame_;
};

/** \class AmMfa
 *  Class for definition of the MFA GMM acoustic model
 */
class AmMfa {
public:
  AmMfa();

  void Read(std::istream &rIn, bool binary);
  void Write(std::ostream &out, bool binary) const;

  /// Checks the various components for correct sizes. With wrong sizes,
  /// assertion failure occurs. When the argument is set to true, dimensions of
  /// the various components are printed.
  void Check(bool show_properties = true);

  /// Initializes the model parameters from an MFA UBM.
  void InitializeFromMfa(const MFA &mfa, int32 num_states, int32 spk_subspace_dim);

  /// Copy from another AmMfa model
  void CopyFromAmMfa(const AmMfa& am_mfa);

  /// Initializes the speaker space
  void InitSpkSpace(int32 spk_subspace_dim);

  /// Precompute for likelihood computation.
  void PreCompute();

  bool IsPreCompute()const { return isPreCompute_; }

  // Unprecompute for likelihood computation.
  void UnPreCompute();

  /// whether using gselect
  bool IsPreSelGauss()const { return isPreSelectGauss_; }

  /// Set the gselect flag.
  /// Note that the AmMfaGselectConfig is stored in the AmMfa object,
  /// So during decoding, there is no need to pass it to the GaussianSelection call.
  /// We can simply set the bPreSelGauss to be true to turn on the gselect option.
  /// If we leave this flag to false, and don't give the gselect rspecifier,
  /// the gselect option will be turned off.
  void SetPreSelGauss(bool bPreSelGauss = true, const AmMfaGselectConfig* pOpt = NULL);

  /// Computes the top-scoring Gaussian indices (used for pruning of later
  /// stages of computation). Returns frame log-likelihood given selected
  /// Gaussians from full UBM.
  BaseFloat GaussianSelection(const VectorBase<BaseFloat> &data,
                              std::vector<int32> *gselect) const;

  /// Computes the top-scoring Gaussian indices (used for pruning of later
  /// stages of computation). Returns frame log-likelihood given selected
  /// Gaussians from full UBM.
  BaseFloat GaussianSelectionForState(const int32 state, const VectorBase<BaseFloat> &data,
                                std::vector<int32> *gselect) const;

  /// whether using directly gselect
  bool IsPreSelGaussDirect()const { return gsDirectOpt_.isPreSelectGaussDirect_; }

  /// Set the gselect direct flag.
  void SetPreSelGaussDirect(const AmMfaGselectDirectConfig* pOpt = NULL) {
	  if (pOpt == NULL)
		  return ;
    gsDirectOpt_.isPreSelectGaussDirect_ = pOpt->isPreSelectGaussDirect_;
    gsDirectOpt_.pruneCompCnt_ = pOpt->pruneCompCnt_;
    gsDirectOpt_.maxCompCnt_ = pOpt->maxCompCnt_;
  }

  /// Computes the top-scoring Gaussian indices (used for pruning of later
  /// stages of computation).
  void GaussianSelectionForStateDirect(const int32 state, const VectorBase<BaseFloat> &data,
                                std::vector<int32> *gselect) const;

  // Convert the covariance matrix to diagonal matrix
  void ConvertToDiagCov()
  {
    if (mfa_.CovarianceType() != DIAG)
    {
      mfa_.ConvertToDiagCov();
      if (isPreCompute_ == true)
      {
        UnPreCompute();
        PreCompute();
      }
    }
  }

  // Convert the covariance matrix to full matrix
  void ConvertToFullCov()
  {
    if (mfa_.CovarianceType() != FULL)
    {
      mfa_.ConvertToFullCov();
      if (isPreCompute_ == true)
      {
        UnPreCompute();
        PreCompute();
      }
    }
  }

  // Convert the MFA acoustic model to a normal diagonal GMM acoustic model
  void ConvertToAmDiagGmm(AmDiagGmm* pAmDiagGmm);

  // Get the SD Model, which changes the precomputed means
  void GenSDModel(const Matrix<BaseFloat>& L, const Matrix<BaseFloat>& ep_mat);

  /// Computes the per-speaker derived vars; assumes vars->v_s is already set up.
  void ComputePerSpkDerivedVars(AmMfaPerSpkDerivedVars *vars) const;

  // Calclulate the log likelihood of one state, returns the posterior optionally
  BaseFloat LogLikelihood(const int32 state, const VectorBase<BaseFloat> &data,
                          const AmMfaPerSpkDerivedVars *vars = NULL,
                          Vector<BaseFloat> *post = NULL,
                          const std::vector<int32>* gselect = NULL,
                          AmMfaDecodingMFACache* pDecodingMFACache = NULL) const;

  /// Various model dimensions.
  int32 NumStates() const
  {
	  return num_states_;
  }
  int32 FeatureDim()const
  {
	  return dim_;
  }
  int32 NumSubspace()const
  {
    return mfa_.NumComps();
  }
  int32 SpkSpaceDim() const
  {
    return (N_.size() > 0) ? N_[0].NumCols() : 0;
  }
  bool HasSpeakerSpace() const
  {
    return SpkSpaceDim() != 0;
  }
  void RemoveSpeakerSpace()
  {
    N_.clear();
  }
  /// Get the number of components for a specified state
  int32 NumComps(int32 state)const
  {
    KALDI_ASSERT(state >= 0 && state < num_states_);
    return sFaIndex_[state].size();
  }
  // Get the total number of components
  int32 TotalNumComps()const
  {
    int32 n = 0;
    for(int32 j = 0; j < num_states_; ++ j){
      n += sFaIndex_[j].size();
    }
    return n;
  }
  // Get the weight vector of a specified state
  const Vector<BaseFloat>& GetWeights(int32 state)const
  {
    KALDI_ASSERT(state >= 0 && state < num_states_);
    return sFaWeight_[state];
  }
  // Get the mean matrix of a specified state
  void GetMeans(int32 state, Matrix<BaseFloat>* pMeans)const;
  // Get the covariance matrix of a specified state and component
  void GetCov(int32 state, int32 comp, SpMatrix<BaseFloat>* pCov)const;

  /// Accessors
  const MFA & GetMFA() const { return mfa_; }

  const SpMatrix<BaseFloat>& GetSigmaInvForSubspace(int32 i)const {
    KALDI_ASSERT(IsPreCompute());
    return invSigma_[i];
  }

  int32 GetStateParmCnt(int32 state)
  {
    int32 sd = 0;
    for(int32 g = 0; g < sFaIndex_[state].size(); ++ g)
    {
      int32 i = sFaIndex_[state][g];
      sd += mfa_.GetLocalDim(i);
    }
    return sd;
  }

protected:
  KALDI_DISALLOW_COPY_AND_ASSIGN(AmMfa);

  /// These contain the "background" model associated with the subspace GMM.
  MFA mfa_;

  /// Speaker-subspace projections. Dimension is [I][D][T]
  std::vector< Matrix<BaseFloat> > N_;

  std::size_t dim_;
  std::size_t num_states_;

  /// factor index for each state; dim is [S][nFA]
  std::vector<std::vector<int32> > sFaIndex_;

  /// local factor locations for each state. Dimension is [S][nFA][LocalDim]
  std::vector<std::vector<Vector<BaseFloat> > > sFaLocation_;

  /// Weight vectors. Dimension is [S][nFA]
  std::vector<Vector<BaseFloat> > sFaWeight_;

  /// Following is the precompute data
  bool isPreCompute_;
  /// Inverse within-class covariances; dim is [I][D][D].
  std::vector<SpMatrix<BaseFloat> > invSigma_;
  /// GConst for each FA; dim is [I]
  Vector<BaseFloat> gconst_;
  /// local means for each state; dim is [S][nFA][D], not include the local center !!!!
  std::vector<Matrix<BaseFloat> > means_;
  /// GConst for each state and gaussian; dim is [S][nFA]
  std::vector<Vector<BaseFloat> > gconst_ji_;
  /// vector of \Sigma_i^{-1} (\mu_i + M y); dim is [S][nFA][D]
  std::vector<Matrix<BaseFloat> > invSigma_mu_ji_;

  /// Direct Gaussian Pruning option: exceed pruneCnt_ count, do direct Gaussian Pruning which prune the number of Gaussian components
  AmMfaGselectDirectConfig gsDirectOpt_;
  /// invSigma_d is a [I][D] matrix, where each row is the inverse of the diagnoanl of the shared covaraince matrix
  Matrix<BaseFloat> invSigma_d_;
  /// vector of \Sigma_i^{-1} (\mu_i + M y); dim is [S][nFA][D]
  //std::vector<Matrix<BaseFloat> > invSigma_mu_ji_d_;
  /// Gconst for each Gaussian Component = log(w_ji) - 0.5 \mu_ji^{T} \Sigma_i^{-1} \mu_ji; dim is [S][nFA]
  //std::vector<Vector<BaseFloat> > gconst_ji_d_;

  /// Following is for gmm pre-selection
  bool isPreSelectGauss_;
  /// Pre-selection options
  AmMfaGselectConfig gsOpt_;
  /// The background FullGmm
  FullGmm fullBgGmm_;
  /// The background DiagGmm
  DiagGmm diagBgGmm_;
  /// The gconst for DiagGmm
  Vector<BaseFloat> diagBgGmmGConst_;

  friend class MleAmMfaAccs;
  friend class MleAmMfaUpdater;
  friend class MleAmMfaSpeakerAccs;
  friend class EpAmMfaAccs;
  friend class EbwAmMfaUpdater;
  friend class AmMfa2;
};

}  // namespace kaldi


#endif  // KALDI_AM_MFA_H_

