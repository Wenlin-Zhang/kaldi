// mfa/am-mfa.cc

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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "gmm/full-gmm-normal.h"
#include "gmm/full-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "mfa/mfa.h"
#include "mfa/am-mfa.h"

namespace kaldi {

AmMfa::AmMfa(): dim_(0), num_states_(0),isPreCompute_(false), isPreSelectGauss_(false)
{}

void AmMfa::Read(std::istream &in_stream, bool binary)
{
  std::string token;

  ExpectToken(in_stream, binary, "<AMMFA>");
  ExpectToken(in_stream, binary, "<DIMENSION>");
  ReadBasicType(in_stream, binary, &dim_);
  ExpectToken(in_stream, binary, "<NUMSTATES>");
  ReadBasicType(in_stream, binary, &num_states_);

  mfa_.Read(in_stream, binary);

  sFaIndex_.resize(num_states_);
  sFaLocation_.resize(num_states_);
  sFaWeight_.resize(num_states_);
  N_.clear();

  int i = 0, num_comp = 0;
  ReadToken(in_stream, binary, &token);
  while (token != "</AMMFA>") {
    if (token == "<STATE>") {
      ExpectToken(in_stream, binary, "<NUMCOMPS>");
      ReadBasicType(in_stream, binary, &num_comp);

      ReadToken(in_stream, binary, &token);
      while (token != "</STATE>") {
        if (token == "<COMP_INDEXES>") {
          ReadIntegerVector(in_stream, binary, &(sFaIndex_[i]));
        } else if (token == "<COMP_WEIGHTS>") {
          sFaWeight_[i].Read(in_stream, binary);
        } else if (token == "<COMP_LOCATIONS>") {
          sFaLocation_[i].resize(num_comp);
          for (int j = 0; j < num_comp; ++j) {
            sFaLocation_[i][j].Read(in_stream, binary);
          }
        } else {
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        }
        ReadToken(in_stream, binary, &token);
      }
      ReadToken(in_stream, binary, &token);
      ++i;
    } else if (token == "<SPKSPACE>") {
      int32 nSpace = 0;
      ReadBasicType(in_stream, binary, &nSpace);
      N_.resize(nSpace);
      for (int k = 0; k < nSpace; ++k) {
        N_[k].Read(in_stream, binary, false);
      }
      ExpectToken(in_stream, binary, "</SPKSPACE>");
      ReadToken(in_stream, binary, &token);
    }
  }

  // Check the state numbers
  if (i != num_states_)
      KALDI_WARN << "AmMfa::Read, find fewer state than expected, " << i << ", vs " << num_states_ << ".";

  // Read the speaker subspace
  if (HasSpeakerSpace() == false)
  {
    KALDI_WARN << "AmMfa::Read, no speaker subspace found!";
  }

  // Check the model consistence
  Check(false);
}

void AmMfa::Write(std::ostream &out_stream, bool binary) const
{
  WriteToken(out_stream, binary, "<AMMFA>");
  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<DIMENSION>");
  WriteBasicType(out_stream, binary, dim_);
  if (!binary) out_stream << "\n";
  WriteToken(out_stream, binary, "<NUMSTATES>");
  WriteBasicType(out_stream, binary, num_states_);
  if (!binary) out_stream << "\n";

  mfa_.Write(out_stream, binary);

  for(int i = 0; i < num_states_; ++ i)
  {
	  WriteToken(out_stream, binary, "<STATE>");
	  if (!binary) out_stream << "\n";

	  int num_comp = sFaIndex_[i].size();
	  WriteToken(out_stream, binary, "<NUMCOMPS>");
	  WriteBasicType(out_stream, binary, num_comp);
	  if (!binary) out_stream << "\n";

	  WriteToken(out_stream, binary, "<COMP_INDEXES>");
	  WriteIntegerVector(out_stream, binary, sFaIndex_[i]);
	  if (!binary) out_stream << "\n";

	  WriteToken(out_stream, binary, "<COMP_WEIGHTS>");
	  sFaWeight_[i].Write(out_stream, binary);
	  if (!binary) out_stream << "\n";

	  WriteToken(out_stream, binary, "<COMP_LOCATIONS>");
	  for(int j = 0; j < num_comp; ++ j)
	  {
		  sFaLocation_[i][j].Write(out_stream, binary);
		  if (!binary) out_stream << "\n";
	  }

	  WriteToken(out_stream, binary, "</STATE>");
	  if (!binary) out_stream << "\n";
  }

  // Write out the speaker subspace
  WriteToken(out_stream, binary, "<SPKSPACE>");
  if (!binary) out_stream << "\n";
  int32 nSpace = N_.size();
  WriteBasicType(out_stream, binary, nSpace);
  if (!binary) out_stream << "\n";
  for (int k = 0; k < nSpace; ++k) {
    N_[k].Write(out_stream, binary);
    if (!binary) out_stream << "\n";
  }
  WriteToken(out_stream, binary, "</SPKSPACE>");
  if (!binary) out_stream << "\n";

  WriteToken(out_stream, binary, "</AMMFA>");

}

/// Checks the various components for correct sizes. With wrong sizes,
/// assertion failure occurs. When the argument is set to true, dimensions of
/// the various components are printed.
void AmMfa::Check(bool show_properties/* = true */)
{
  KALDI_ASSERT(dim_ == mfa_.Dim());
  KALDI_ASSERT(num_states_ == sFaIndex_.size());
  KALDI_ASSERT(num_states_ == sFaWeight_.size());
  KALDI_ASSERT(num_states_ == sFaLocation_.size());
  if (show_properties) {
    KALDI_LOG << "AmMfa: #states = " << num_states_ << ", feature dim = "
              << dim_;
  }
  for (int i = 0; i < num_states_; ++i) {
    KALDI_ASSERT(sFaIndex_[i].size() == sFaWeight_[i].Dim());
    KALDI_ASSERT(sFaIndex_[i].size() == sFaLocation_[i].size());
    if (show_properties) {
      KALDI_LOG << "#s" << i << ":" << sFaIndex_[i].size();
    }
    for (int j = 0; j < sFaIndex_[i].size(); ++j) {
      KALDI_ASSERT(sFaIndex_[i][j] >= 0 && sFaIndex_[i][j] < mfa_.NumComps());
      KALDI_ASSERT(sFaWeight_[i](j) >= 0 && sFaWeight_[i](j) <= 1);
      //if (!(sFaWeight_[i](j) >= 0 && sFaWeight_[i](j) <= 1))
      //{
      //  KALDI_ASSERT(0);
      //}
      KALDI_ASSERT(sFaLocation_[i][j].Dim() == mfa_.GetLocalDim(sFaIndex_[i][j]));
    }
  }
  if (HasSpeakerSpace() == true)
  {
     KALDI_ASSERT(N_.size() == mfa_.NumComps());
     int spkSpaceDim = this->SpkSpaceDim();
     for(int i = 0; i < N_.size(); ++ i)
     {
       KALDI_ASSERT(N_[i].NumRows() == mfa_.Dim());
       if (i > 0)
         KALDI_ASSERT(N_[i].NumCols() == spkSpaceDim);
     }
  }
}

/// Initializes the speaker space
void AmMfa::InitSpkSpace(int32 spk_subspace_dim)
 {
  Matrix<BaseFloat> norm_xform;
  if (spk_subspace_dim > 0) {
    KALDI_ASSERT(spk_subspace_dim <= dim_);
    N_.resize(mfa_.NumComps());
    mfa_.ComputeFeatureNormalizer(&norm_xform);
    for (int32 i = 0; i < mfa_.NumComps(); ++i) {
      N_[i].Resize(dim_, spk_subspace_dim);
      N_[i].CopyFromMat(norm_xform.Range(0, dim_, 0, spk_subspace_dim));
    }
  } else
    N_.resize(0);
}

/// Initializes the model parameters from an MFA UBM.
void AmMfa::InitializeFromMfa(const MFA &mfa, int32 num_state, int32 spk_subspace_dim)
{
	dim_ = mfa.Dim();
	num_states_ = num_state;

	this->mfa_.CopyFromMFA(mfa);

	sFaIndex_.resize(num_states_);
	sFaLocation_.resize(num_states_);
	sFaWeight_.resize(num_states_);

	isPreCompute_ = false;
	invSigma_.resize(0);
	gconst_.Resize(0);
	means_.resize(0);

	int num_fa = mfa_.NumComps();
	const std::vector<std::size_t>& k_vec = mfa_.LocalDims();
	float weight = 1.0 / (float)num_fa;
	std::vector<int> indVec(num_fa);
	for(int i = 0; i < num_fa; ++ i)
		indVec[i] = i;


	for(int i = 0; i < num_states_; ++ i)
	{
		sFaIndex_[i] = indVec;

		sFaWeight_[i].Resize(num_fa);
		sFaWeight_[i].Set(weight);

		sFaLocation_[i].resize(num_fa);
		for(int j = 0; j < num_fa; ++ j)
			sFaLocation_[i][j].Resize(k_vec[j], kSetZero);
	}

	InitSpkSpace(spk_subspace_dim);
}

/// Copy from another AmMfa model
void AmMfa::CopyFromAmMfa(const AmMfa& am_mfa)
{
	 if (this->IsPreCompute())
		 UnPreCompute();
	 if (this->IsPreSelGauss())
		 SetPreSelGauss(false);

  // copy mfa
	mfa_.CopyFromMFA(am_mfa.mfa_);

	// copy feature dim and state number
	dim_ = am_mfa.dim_;
	num_states_ = am_mfa.num_states_;

	// copy  speaker-subspace projections
	N_ = am_mfa.N_;

	// copy state-dependent parameters
	sFaIndex_ = am_mfa.sFaIndex_;
	sFaLocation_ = am_mfa.sFaLocation_;
	sFaWeight_ = am_mfa.sFaWeight_;

}

/// Precompute for likelihood computation.
void AmMfa::PreCompute()
{
  int I = mfa_.NumComps(), D = mfa_.Dim();

  // PreCompute at the MFA level
  gconst_.Resize(I);
  gconst_.Set(- D * M_LOG_2PI / 2);
  invSigma_.resize(I);
  switch(mfa_.CovarianceType())
  {
    case DIAG:
      for(int i = 0; i < I; ++ i)
      {
        const Vector<BaseFloat>& var = mfa_.GetLocalVar(i).diagCov_;
        gconst_(i) -= var.SumLog() / 2;

        if (mfa_.IsPreCompute() == true)
          invSigma_[i] = mfa_.pre_data_vec_[i]->inv_Sigma_;
        else
        {
          invSigma_[i].Resize(dim_);
          invSigma_[i].AddDiagVec(1.0, var);
          invSigma_[i].Invert();
        }
      }
      break;
    case FULL:
      for(int i = 0; i < I; ++ i)
      {
        const SpMatrix<BaseFloat>& var = mfa_.GetLocalVar(i).fullCov_;
        gconst_(i) -= var.LogDet() / 2;

        if (mfa_.IsPreCompute() == true)
          invSigma_[i] = mfa_.pre_data_vec_[i]->inv_Sigma_;
        else
        {
          invSigma_[i] = var;
          invSigma_[i].Invert();
        }
      }
      break;
    default:
      KALDI_ERR << "Invalid Covariance type.";
      break;
  }

  // PreCompute at the state level
  means_.resize(num_states_);
  invSigma_mu_ji_.resize(num_states_);
  gconst_ji_.resize(num_states_);
  Vector<BaseFloat> mu_ji(D);
  for(int j = 0; j < num_states_; ++ j)
  {
    const std::vector<int>& faIndex = sFaIndex_[j];
    int M = faIndex.size();
    Matrix<BaseFloat>& matMean = means_[j];
    Matrix<BaseFloat>& invSigma_mu_ji = invSigma_mu_ji_[j];
    Vector<BaseFloat>& gconst_ji = gconst_ji_[j];
    matMean.Resize(M, D);
    invSigma_mu_ji.Resize(M, D);
    gconst_ji.Resize(M);
    for (int m = 0; m < M; ++ m)
    {
      int i = faIndex[m];
      mu_ji.AddMatVec(1.0, mfa_.GetLocalBases(i), kNoTrans, sFaLocation_[j][m], 0.0);
      matMean.Row(m).CopyFromVec(mu_ji);
      mu_ji.AddVec(1.0, mfa_.GetLocalCenter(i));
      invSigma_mu_ji.Row(m).AddSpVec(1.0, invSigma_[i], mu_ji, 0.0);
      gconst_ji(m) = log(sFaWeight_[j](m)) - 0.5 * VecVec(mu_ji, invSigma_mu_ji.Row(m));
    }
  }

  // PreCompute For diagonal background model
  if (gsDirectOpt_.isPreSelectGaussDirect_ == true)
  {
    invSigma_d_.Resize(I, D);
    switch(mfa_.CovarianceType())
    {
      case DIAG:
        for(int i = 0; i < I; ++ i)
        {
          kaldi::SubVector<BaseFloat> sv = invSigma_d_.Row(i);
          const Vector<BaseFloat>& var = mfa_.GetLocalVar(i).diagCov_;
          sv.CopyFromVec(var);
          sv.InvertElements();
        }
        break;
      case FULL:
        for(int i = 0; i < I; ++ i)
        {
          kaldi::SubVector<BaseFloat> sv = invSigma_d_.Row(i);
          const SpMatrix<BaseFloat>& var = mfa_.GetLocalVar(i).fullCov_;
          sv.CopyDiagFromSp(var);
          sv.InvertElements();
        }
        break;
      default:
        KALDI_ERR << "Invalid Covariance type.";
        break;
    }
  }
  /*
  invSigma_mu_ji_d_.resize(num_states_);
  gconst_ji_d_.resize(num_states_);
  for(int j = 0; j < num_states_; ++ j)
  {
	  const std::vector<int>& faIndex = sFaIndex_[j];
	  int M = faIndex.size();
	  Matrix<BaseFloat>& matMean = means_[j];  // Get the local means
	  Matrix<BaseFloat>& invSigma_mu_ji_d = invSigma_mu_ji_d_[j];
	  Vector<BaseFloat>& gconst_ji_d = gconst_ji_d_[j];
	  invSigma_mu_ji_d.Resize(M, D);
	  gconst_ji_d.Resize(M);
	  for (int m = 0; m < M; ++ m)
	  {
	    int i = faIndex[m];
	    mu_ji.CopyFromVec(matMean.Row(m));
		  mu_ji.AddVec(1.0, mfa_.GetLocalCenter(i));
		  invSigma_mu_ji_d.Row(m).AddVecVec(1.0, invSigma_d_.Row(i), mu_ji, 0.0);
		  gconst_ji_d(m) = - D * M_LOG_2PI / 2 + invSigma_d_.Row(i).SumLog() / 2
		      + log(sFaWeight_[j](m)) - 0.5 * VecVec(mu_ji, invSigma_mu_ji_d.Row(m));
	  }
  }
  */
  isPreCompute_ = true;
}

// Unprecompute for likelihood computation.
void AmMfa::UnPreCompute()
{
  if (isPreCompute_ == false)
    return ;

  // MFA level
  invSigma_.resize(0);
  gconst_.Resize(0);
  means_.resize(0);

  // State level
  gconst_ji_.resize(0);
  invSigma_mu_ji_.resize(0);

  // direct precompute level
  invSigma_d_.Resize(0, 0);
  //invSigma_mu_ji_d_.resize(0);
  //gconst_ji_d_.resize(0);

  isPreCompute_ = false;
}

/// Set the gselect flag.
/// Note that the AmMfaGselectConfig is stored in the AmMfa object,
/// So during decoding, there is no need to pass it to the GaussianSelection call.
/// We can simply set the bPreSelGauss to be true to turn on the gselect option.
/// If we leave this flag to false, and don't give the gselect rspecifier,
/// the gselect option will be turned off.
void AmMfa::SetPreSelGauss(bool bPreSelGauss/* = true */, const AmMfaGselectConfig* pOpt/* = NULL*/)
{
  if (bPreSelGauss == true)
  {
    if (isPreSelectGauss_ == true)
      return ;
    fullBgGmm_.Resize(mfa_.NumComps(), mfa_.Dim());
    diagBgGmm_.Resize(mfa_.NumComps(), mfa_.Dim());
    mfa_.ConvertToFullGmm(&fullBgGmm_, false);
    diagBgGmm_.CopyFromFullGmm(fullBgGmm_);

    // 计算对角矩阵的GConst常数（除去权重）
    diagBgGmmGConst_.Resize(mfa_.NumComps());
    int32 D = mfa_.Dim();
    for(size_t i = 0; i < mfa_.NumComps(); ++ i)
    {
      const Covariance& cov = mfa_.GetLocalVar(i);
      if (mfa_.CovarianceType() == DIAG)
      {
        diagBgGmmGConst_(i) = - D * M_LOG_2PI / 2 - cov.diagCov_.SumLog() / 2;
      }
      else
      {
        double tmp = 0.0;
        for(size_t d = 0; d < D; ++d)
          tmp += log(double(cov.fullCov_(d, d)));
        diagBgGmmGConst_(i) = - D * M_LOG_2PI / 2 - tmp / 2;
      }
    }

    if (pOpt != NULL)
    {
      KALDI_ASSERT(pOpt->diag_gmm_nbest > 0 &&
                   pOpt->full_gmm_nbest > 0 &&
                   pOpt->full_gmm_nbest < pOpt->diag_gmm_nbest);
      gsOpt_ = *pOpt;
    }
    isPreSelectGauss_ = true;
  }
  else
  {
    if (isPreSelectGauss_ == false)
      return ;
    else
    {
      fullBgGmm_.Resize(0, 0);
      diagBgGmm_.Resize(0, 0);
      isPreSelectGauss_ = false;
    }
  }
}

/// Computes the top-scoring Gaussian indices (used for pruning of later
/// stages of computation). Returns frame log-likelihood given selected
/// Gaussians from full UBM.
BaseFloat AmMfa::GaussianSelection(const VectorBase<BaseFloat> &data,
                                std::vector<int32> *gselect) const
{
  KALDI_ASSERT(isPreSelectGauss_ == true);
  KALDI_ASSERT(diagBgGmm_.NumGauss() != 0 &&
               diagBgGmm_.NumGauss() == fullBgGmm_.NumGauss() &&
               diagBgGmm_.Dim() == data.Dim());

  int32 num_gauss = diagBgGmm_.NumGauss();

  std::vector<std::pair<BaseFloat, int32> > pruned_pairs;
  if (gsOpt_.diag_gmm_nbest < num_gauss) {
    Vector<BaseFloat> loglikes(num_gauss);
    diagBgGmm_.LogLikelihoods(data, &loglikes);
    Vector<BaseFloat> loglikes_copy(loglikes);
    BaseFloat *ptr = loglikes_copy.Data();
    std::nth_element(ptr, ptr + num_gauss - gsOpt_.diag_gmm_nbest,
                     ptr + num_gauss);
    BaseFloat thresh = ptr[num_gauss - gsOpt_.diag_gmm_nbest];
    for (int32 g = 0; g < num_gauss; g++)
      if (loglikes(g) >= thresh)  // met threshold for diagonal phase.
        pruned_pairs.push_back(
            std::make_pair(fullBgGmm_.ComponentLogLikelihood(data, g), g));
  } else {
    Vector<BaseFloat> loglikes(num_gauss);
    fullBgGmm_.LogLikelihoods(data, &loglikes);
    for (int32 g = 0; g < num_gauss; g++)
      pruned_pairs.push_back(std::make_pair(loglikes(g), g));
  }
  KALDI_ASSERT(!pruned_pairs.empty());
  if (pruned_pairs.size() > static_cast<size_t>(gsOpt_.full_gmm_nbest)) {
    std::nth_element(pruned_pairs.begin(),
                     pruned_pairs.end() - gsOpt_.full_gmm_nbest,
                     pruned_pairs.end());
    pruned_pairs.erase(pruned_pairs.begin(),
                       pruned_pairs.end() - gsOpt_.full_gmm_nbest);
  }
  Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
  KALDI_ASSERT(gselect != NULL);
  gselect->resize(pruned_pairs.size());
  // Make sure pruned Gaussians appear from best to worst.
  std::sort(pruned_pairs.begin(), pruned_pairs.end(),
            std::greater<std::pair<BaseFloat, int32> >());
  for (size_t i = 0; i < pruned_pairs.size(); i++) {
    loglikes_tmp(i) = pruned_pairs[i].first;
    (*gselect)[i] = pruned_pairs[i].second;
  }
  return loglikes_tmp.LogSumExp();
}

/// Computes the top-scoring Gaussian indices (used for pruning of later
/// stages of computation). Returns frame log-likelihood given selected
/// Gaussians from full UBM.
BaseFloat AmMfa::GaussianSelectionForState(const int32 state, const VectorBase<BaseFloat> &data,
                                std::vector<int32> *gselect) const
{
  KALDI_ASSERT(isPreSelectGauss_ == true);
  KALDI_ASSERT(diagBgGmm_.NumGauss() != 0 &&
               diagBgGmm_.NumGauss() == fullBgGmm_.NumGauss() &&
               diagBgGmm_.Dim() == data.Dim());

  const std::vector<int32>& faIndex = sFaIndex_[state];
  int nFA = faIndex.size();

  std::vector<std::pair<BaseFloat, int32> > pruned_pairs;
  if (gsOpt_.diag_gmm_nbest < nFA) {
    Vector<BaseFloat> loglikes(nFA);
    diagBgGmm_.LogLikelihoodsPreselect(data, faIndex, &loglikes);
    Vector<BaseFloat> loglikes_copy(loglikes);
    BaseFloat *ptr = loglikes_copy.Data();
    std::nth_element(ptr, ptr + nFA - gsOpt_.diag_gmm_nbest,
                     ptr + nFA);
    BaseFloat thresh = ptr[nFA - gsOpt_.diag_gmm_nbest];
    for (int32 g = 0; g < nFA; g++)
      if (loglikes(g) >= thresh)  // met threshold for diagonal phase.
        pruned_pairs.push_back(
            std::make_pair(fullBgGmm_.ComponentLogLikelihood(data, faIndex[g]), faIndex[g]));
  } else {
    Vector<BaseFloat> loglikes(nFA);
    fullBgGmm_.LogLikelihoodsPreselect(data, faIndex, &loglikes);
    for (int32 g = 0; g < nFA; g++)
      pruned_pairs.push_back(std::make_pair(loglikes(g), faIndex[g]));
  }
  KALDI_ASSERT(!pruned_pairs.empty());
  if (pruned_pairs.size() > static_cast<size_t>(gsOpt_.full_gmm_nbest)) {
    std::nth_element(pruned_pairs.begin(),
                     pruned_pairs.end() - gsOpt_.full_gmm_nbest,
                     pruned_pairs.end());
    pruned_pairs.erase(pruned_pairs.begin(),
                       pruned_pairs.end() - gsOpt_.full_gmm_nbest);
  }
  Vector<BaseFloat> loglikes_tmp(pruned_pairs.size());  // for return value.
  KALDI_ASSERT(gselect != NULL);
  gselect->resize(pruned_pairs.size());
  // Make sure pruned Gaussians appear from best to worst.
  std::sort(pruned_pairs.begin(), pruned_pairs.end(),
            std::greater<std::pair<BaseFloat, int32> >());
  for (size_t i = 0; i < pruned_pairs.size(); i++) {
    loglikes_tmp(i) = pruned_pairs[i].first;
    (*gselect)[i] = pruned_pairs[i].second;
  }
  return loglikes_tmp.LogSumExp();
}


/// Computes the top-scoring Gaussian indices (used for pruning of later
/// stages of computation).
void AmMfa::GaussianSelectionForStateDirect(const int32 state, const VectorBase<BaseFloat> &data,
                              std::vector<int32> *gselect) const
{
  KALDI_ASSERT(this->IsPreSelGaussDirect() == true);
  KALDI_ASSERT(gsDirectOpt_.maxCompCnt_ > 0 && gsDirectOpt_.pruneCompCnt_ >= gsDirectOpt_.maxCompCnt_);

  const std::vector<int32>& faIndex = sFaIndex_[state];
  int nFA = faIndex.size();

  Vector<BaseFloat> mu_ji(dim_);
  Vector<BaseFloat> invSigma_mu(dim_);
  const Matrix<BaseFloat>& matMean = means_[state];  // Get the local means
  const Vector<BaseFloat>& faWeights = sFaWeight_[state];
  Vector<BaseFloat> data2(data); // squared data
  data2.ApplyPow(2.0);
  if (nFA > gsDirectOpt_.pruneCompCnt_) {
    Vector<BaseFloat> loglikes(nFA);
    for (int32 i = 0; i < nFA; ++i) {
      int m = faIndex[i];
      SubVector<BaseFloat> invSigma = invSigma_d_.Row(m);
      mu_ji.CopyFromVec(matMean.Row(i));
      mu_ji.AddVec(1.0, mfa_.GetLocalCenter(m));
      invSigma_mu.AddVecVec(1.0, invSigma, mu_ji, 0.0);
      loglikes(i) = /*-D * M_LOG_2PI / 2 + */ invSigma.SumLog() / 2
          + log(faWeights(i)) - 0.5 * VecVec(mu_ji, invSigma_mu)
          - 0.5 * VecVec(data2, invSigma)
          + VecVec(invSigma_mu, data);
    }
    Vector<BaseFloat> loglikes_copy(loglikes);
    BaseFloat *ptr = loglikes_copy.Data();
    std::nth_element(ptr, ptr + nFA - gsDirectOpt_.maxCompCnt_, ptr + nFA);
    BaseFloat thresh = ptr[nFA - gsDirectOpt_.maxCompCnt_];
    gselect->resize(0);
    for (int32 g = 0; g < nFA; g++)
      if (loglikes(g) >= thresh)  // met threshold for diagonal phase.
        gselect->push_back(g);
  } else {
    //gselect->resize(nFA);
    //for (int32 g = 0; g < nFA; g++)
    //  gselect[g] = g;
    gselect->resize(0);
  }
  return;
}

void AmMfa::ConvertToAmDiagGmm(AmDiagGmm* pAmDiagGmm)
{
  // pAmDiagGmm->CopyFromAmMfa(*this);
  if (pAmDiagGmm->densities_.size() != 0) {
    DeletePointers(&pAmDiagGmm->densities_);
  }
  pAmDiagGmm->densities_.resize(this->NumStates(), NULL);

  int32 dim = this->FeatureDim();
  for (int32 j = 0, end = pAmDiagGmm->densities_.size(); j < end; j++) {
    DiagGmm* pDiagGmm = new DiagGmm();
    {
      int32 num_comp = this->NumComps(j);
      pDiagGmm->Resize(num_comp, dim);
      pDiagGmm->weights_.CopyFromVec(this->GetWeights(j));
      Matrix<BaseFloat> means(num_comp, dim);
      this->GetMeans(j, &means);
      for (int32 mix = 0; mix < num_comp; mix++) {
        SpMatrix<BaseFloat> covar(dim);
        this->GetCov(j, mix, &covar);
        Vector<BaseFloat> diag(dim);
        diag.CopyDiagFromPacked(covar);
        diag.InvertElements();
        pDiagGmm->inv_vars_.Row(mix).CopyFromVec(diag);
      }
      pDiagGmm->means_invvars_.CopyFromMat(means);
      pDiagGmm->means_invvars_.MulElements(pDiagGmm->inv_vars_);

      pDiagGmm->gconsts_.Resize(num_comp);
      pDiagGmm->valid_gconsts_ = false;
      pDiagGmm->ComputeGconsts();
    }
    pAmDiagGmm->densities_[j] = pDiagGmm;
  }
}

// Get the mean matrix of a specified state
void AmMfa::GetMeans(int32 state, Matrix<BaseFloat>* pMeans)const
{
  KALDI_ASSERT(state >= 0 && state < num_states_);
  const std::vector<int>& faIndex = sFaIndex_[state];
  pMeans->Resize(faIndex.size(), dim_);
  for (int i = 0; i < faIndex.size(); ++ i)
  {
    int m = faIndex[i];
    pMeans->Row(i).CopyFromVec(mfa_.GetLocalCenter(m));
    pMeans->Row(i).AddMatVec(1.0, mfa_.GetLocalBases(m), kNoTrans, sFaLocation_[state][i], 1.0);
  }
}

// Get the diagonal covariance of a specified state and component
void AmMfa::GetCov(int32 state, int32 comp, SpMatrix<BaseFloat>* pCov)const
{
  KALDI_ASSERT(state >= 0 && state < num_states_);
  KALDI_ASSERT(comp >= 0 && comp < sFaIndex_[state].size());
  KALDI_ASSERT(pCov != NULL);
  if (pCov->NumRows() != dim_)
    pCov->Resize(dim_);

  int32 m = sFaIndex_[state][comp];
  const Covariance& cov = mfa_.GetLocalVar(m);
  switch(mfa_.CovarianceType())
  {
    case DIAG:
      pCov->SetZero();
      pCov->AddDiagVec(1.0, cov.diagCov_);
      break;
    case FULL:
      pCov->CopyFromSp(cov.fullCov_);
      break;
    default:
      KALDI_ERR << "Invalid covariance type.";
      break;
  }
}

// Get the SD Model, which changes the precomputed means
void AmMfa::GenSDModel(const Matrix<BaseFloat>& L, const Matrix<BaseFloat>& ep_mat)
{
  KALDI_ASSERT(isPreCompute_ == true);
  KALDI_ASSERT(L.NumCols() == ep_mat.NumRows());
  int32 mix = 0;
  for(int i = 0; i < num_states_; ++ i)
  {
    const std::vector<int>& faIndex = sFaIndex_[i];
    Matrix<BaseFloat>& matMean = means_[i];
    for (int j = 0; j < faIndex.size(); ++ j)
    {
      matMean.Row(j).AddMatVec(1.0, mfa_.GetLocalBases(faIndex[j]), kNoTrans, sFaLocation_[i][j], 0.0);
    }
    SubMatrix<BaseFloat> L_mat = L.Range(mix, faIndex.size(), 0, L.NumCols());
    matMean.AddMatMat(1.0, L_mat, kNoTrans,
                      ep_mat, kNoTrans, 1.0);
    mix += faIndex.size();
  }
  KALDI_ASSERT(mix == L.NumRows());
}

/// Computes the per-speaker derived vars; assumes vars->v_s is already set up.
void AmMfa::ComputePerSpkDerivedVars(AmMfaPerSpkDerivedVars *vars) const {
  KALDI_ASSERT(vars != NULL);
  if (vars->v_s.Dim() != 0) {
    KALDI_ASSERT(vars->v_s.Dim() == SpkSpaceDim());
    int32 num_comps = mfa_.NumComps();
    vars->o_s.Resize(num_comps, FeatureDim());

    for (int32 i = 0; i < num_comps; i++) {
      // Eqn. (32): o_i^{(s)} = N_i v^{(s)}
      vars->o_s.Row(i).AddMatVec(1.0, N_[i], kNoTrans, vars->v_s, 0.0);
    }
  } else {
    vars->o_s.Resize(0, 0);
  }
}

// Calculate the log likelihood of one state
BaseFloat AmMfa::LogLikelihood(const int32 state, const VectorBase<BaseFloat> &data,
                               const AmMfaPerSpkDerivedVars *vars/* = NULL */,
                               Vector<BaseFloat> *post/* = NULL */,
                               const std::vector<int32>* gselect/* = NULL */,
                               AmMfaDecodingMFACache* pDecodingMFACache/* = NULL*/) const
{
  KALDI_ASSERT(isPreCompute_ == true);

  int nFA = sFaIndex_[state].size();

  std::vector<int32> gselect_tmp;

  // Doing Gaussian preslection at the decoding time
  if (this->IsPreSelGauss() == true)
  {
    GaussianSelectionForState(state, data, &gselect_tmp);
    gselect = &gselect_tmp;
  }

  // Doing Gaussian preselection using a direct method
  if (this->IsPreSelGaussDirect() == true)
  {
    GaussianSelectionForStateDirect(state, data, &gselect_tmp);
    if (gselect_tmp.size() > 0)
    {
      gselect = &gselect_tmp;
      //KALDI_LOG << "Prune Gaussian from " << nFA << " to " << gselect_tmp.size() << ".";
    }
  }

  Vector<BaseFloat> llk(nFA);
  Vector<BaseFloat> x(data);
  int32 pruned_cnt = 0;
  for(int i = 0; i < nFA; ++ i)
  {
    int m = sFaIndex_[state][i];

    // perform gselect
    if (gselect != NULL && std::find(gselect->begin(), gselect->end(), i) == gselect->end())
    {
      llk(i) = kLogZeroBaseFloat;
      ++ pruned_cnt;
      continue;
    }

    x = data;
    if (vars != NULL && vars->o_s.NumRows() > 0)  // speaker-dependent shift
      x.AddVec(-1, vars->o_s.Row(m));
    llk(i) += gconst_(m);
    llk(i) += gconst_ji_[state](i);
    llk(i) += VecVec(x, invSigma_mu_ji_[state].Row(i));
    if (pDecodingMFACache != NULL)
    {
      // find the cache
      BaseFloat mfa_llk = 0.0;
      if (pDecodingMFACache->FindCache(m, &mfa_llk))
      {
        llk(i) += mfa_llk;
      }
      else
      {
        mfa_llk = -0.5 * VecSpVec(x, invSigma_[m], x);
        llk(i) += mfa_llk;
        pDecodingMFACache->InsertCache(m, mfa_llk);
      }
    }
    else
      llk(i) += -0.5 * VecSpVec(x, invSigma_[m], x);
  }

  if (post == NULL)
  {
    if (pruned_cnt == nFA)
      return -1.0e10;
    else
      return llk.LogSumExp();
  }
  else
  {
    post->Resize(nFA);
    if (pruned_cnt == nFA)
    {
      post->SetZero();
      return -1.0e10;
    }
    else
    {
      BaseFloat totLLK = llk.ApplySoftMax();
      post->CopyFromVec(llk);
      return totLLK;
    }
  }
}

}
