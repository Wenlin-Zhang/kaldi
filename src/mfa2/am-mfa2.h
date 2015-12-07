// mfa2/am-mfa2.h

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

#ifndef KALDI_AM_MFA2_H_
#define KALDI_AM_MFA2_H_ 1

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
class AmMfa;

struct AmMfa2GselectDirectConfig {
  int32 isPreSelectGaussDirect_;
  int32 pruneCompCnt_;
  int32 maxCompCnt_;

  AmMfa2GselectDirectConfig(): isPreSelectGaussDirect_(false), pruneCompCnt_(40), maxCompCnt_(20) {
  }

  void Register(OptionsItf* po) {
    po->Register("ammfa2-gselect-direct", &isPreSelectGaussDirect_, "Prune Gaussians of each state using direct method");
    po->Register("ammfa2-prune-comp", &pruneCompCnt_, "Prune threshold for AmMfa likelihood computation");
    po->Register("ammfa2-max-comp", &maxCompCnt_, "Maximum Gaussian component count for AmMfa likelihood computation");
  }
};

/** \class AmMfa2
 *  Class for definition of the MFA2 GMM acoustic model with sparse inverse covariance matrix for each state component
 */
class AmMfa2{
public:
  AmMfa2();

  void Read(std::istream &rIn, bool binary);
  void Write(std::ostream &out, bool binary) const;

  /// Checks the various components for correct sizes. With wrong sizes,
  /// assertion failure occurs. When the argument is set to true, dimensions of
  /// the various components are printed.
  void Check(bool show_properties = true);

  /// Copy from another AmMfa model
  void CopyFromAmMfa(const AmMfa& am_mfa);

  /// Precompute for likelihood computation.
  void PreCompute();

  bool IsPreCompute()const { return isPreCompute_; }

  // Unprecompute for likelihood computation.
  void UnPreCompute();

  /// whether using directly gselect
  bool IsPreSelGaussDirect()const { return gsDirectOpt_.isPreSelectGaussDirect_; }

  /// Set the gselect direct flag.
  void SetPreSelGaussDirect(const AmMfa2GselectDirectConfig* pOpt = NULL) {
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

  // Calclulate the log likelihood of one state, returns the posterior optionally
  BaseFloat LogLikelihood(const int32 state, const VectorBase<BaseFloat> &data,
                          Vector<BaseFloat> *post = NULL,  const std::vector<int32>* gselect = NULL) const;

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
  const SpMatrix<BaseFloat>& GetInvCov(int32 state, int32 comp)const
  {
	  return sFaInvSigma_[state][comp];
  }

  /// Accessors
  const MFA & GetMFA() const { return mfa_; }

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
  KALDI_DISALLOW_COPY_AND_ASSIGN(AmMfa2);

  /// These contain the "background" model associated with the subspace GMM.
  MFA mfa_;

  std::size_t dim_;
  std::size_t num_states_;

  /// factor index for each state; dim is [S][nFA]
  std::vector<std::vector<int32> > sFaIndex_;

  /// local factor locations for each state. Dimension is [S][nFA][LocalDim]
  std::vector<std::vector<Vector<BaseFloat> > > sFaLocation_;

  /// Inverse within-class covariances; dim is [S][nFA][D][D].
  std::vector<std::vector<SpMatrix<BaseFloat> > > sFaInvSigma_;

  /// Weight vectors. Dimension is [S][nFA]
  std::vector<Vector<BaseFloat> > sFaWeight_;

  /// Following is the precompute data
  bool isPreCompute_;
  /// local means for each state; dim is [S][nFA][D], not include the local center !!!!
  std::vector<Matrix<BaseFloat> > means_;
  /// GConst for each state and gaussian; dim is [S][nFA]
  std::vector<Vector<BaseFloat> > gconst_ji_;
  /// vector of \Sigma_{ji}^{-1} (\mu_i + M y); dim is [S][nFA][D]
  std::vector<Matrix<BaseFloat> > invSigma_mu_ji_;

  /// Direct Gaussian Pruning option: exceed pruneCnt_ count, do direct Gaussian Pruning which prune the number of Gaussian components
  AmMfa2GselectDirectConfig gsDirectOpt_;
  /// invSigma_d is a [S][nFA][D] matrix, where each row is the inverse of the diagnoanl of the shared covaraince matrix
  std::vector<Matrix<BaseFloat> >  invSigma_d_;


  friend class MleAmMfa2Accs;
  friend class MleAmMfa2Updater;
  friend class EbwAmMfa2Updater;
};

}  // namespace kaldi


#endif  // KALDI_AM_MFA_H_

