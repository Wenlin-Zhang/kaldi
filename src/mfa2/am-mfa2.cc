// mfa2/am-mfa2.cc

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

#include <algorithm>
#include <limits>
#include <string>
#include <vector>

#include "gmm/full-gmm-normal.h"
#include "gmm/full-gmm.h"
#include "gmm/am-diag-gmm.h"
#include "mfa/mfa.h"
#include "mfa/am-mfa.h"
#include "mfa2/am-mfa2.h"

namespace kaldi {

AmMfa2::AmMfa2(): dim_(0), num_states_(0),isPreCompute_(false)
{}

void AmMfa2::Read(std::istream &in_stream, bool binary)
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
  sFaInvSigma_.resize(num_states_);

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
        }else if(token == "<COMP_INV_COVS>") {
        	  sFaInvSigma_[i].resize(num_comp);
        	  for(int j = 0; j < num_comp; ++ j) {
        		  sFaInvSigma_[i][j].Read(in_stream, binary);
        	  }
          }else {
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        }
        ReadToken(in_stream, binary, &token);
      }
      ReadToken(in_stream, binary, &token);
      ++i;
    }
  }

  // Check the state numbers
  if (i != num_states_)
      KALDI_WARN << "AmMfa::Read, find fewer state than expected, " << i << ", vs " << num_states_ << ".";

  // Check the model consistence
  Check(false);
}

void AmMfa2::Write(std::ostream &out_stream, bool binary) const
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

	  WriteToken(out_stream, binary, "<COMP_INV_COVS>");
	  for(int j = 0; j < num_comp; ++ j)
	  {
		  sFaInvSigma_[i][j].Write(out_stream, binary);
	  		  if (!binary) out_stream << "\n";
	   }

	  WriteToken(out_stream, binary, "</STATE>");
	  if (!binary) out_stream << "\n";
  }

  WriteToken(out_stream, binary, "</AMMFA>");

}

/// Checks the various components for correct sizes. With wrong sizes,
/// assertion failure occurs. When the argument is set to true, dimensions of
/// the various components are printed.
void AmMfa2::Check(bool show_properties/* = true */)
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
    KALDI_ASSERT(sFaIndex_[i].size() == sFaInvSigma_[i].size());
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
}

/// Copy from another AmMfa model
void AmMfa2::CopyFromAmMfa(const AmMfa& am_mfa)
{
	 if (this->IsPreCompute())
		 UnPreCompute();

  // copy mfa
	mfa_.CopyFromMFA(am_mfa.mfa_);

	// copy feature dim and state number
	dim_ = am_mfa.dim_;
	num_states_ = am_mfa.num_states_;

	// copy state-dependent parameters
	sFaIndex_ = am_mfa.sFaIndex_;
	sFaLocation_ = am_mfa.sFaLocation_;
	sFaWeight_ = am_mfa.sFaWeight_;

	// copy the inverse covariance matrix
	mfa_.PreCompute();
	sFaInvSigma_.resize(num_states_);
	for(int i = 0; i < num_states_; ++ i)
	{
		int num_comp = sFaIndex_[i].size();
		sFaInvSigma_[i].resize(num_comp);
		for(int j = 0; j < num_comp; ++ j)
		{
			mfa_.GetInvCovarianceMatrix(sFaIndex_[i][j],  sFaInvSigma_[i][j]);
		}
	}
}

/// Precompute for likelihood computation.
void AmMfa2::PreCompute()
{
  int D = mfa_.Dim();

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
    Vector<BaseFloat>& gconst_j = gconst_ji_[j];
    matMean.Resize(M, D);
    invSigma_mu_ji.Resize(M, D);
    gconst_j.Resize(M);
    gconst_j.Set(- D * M_LOG_2PI / 2);
    for (int m = 0; m < M; ++ m)
    {
      int i = faIndex[m];
      mu_ji.AddMatVec(1.0, mfa_.GetLocalBases(i), kNoTrans, sFaLocation_[j][m], 0.0);
      matMean.Row(m).CopyFromVec(mu_ji);
      mu_ji.AddVec(1.0, mfa_.GetLocalCenter(i));
      invSigma_mu_ji.Row(m).AddSpVec(1.0, sFaInvSigma_[j][m], mu_ji, 0.0);
      gconst_j(m) +=  (sFaInvSigma_[j][m].LogDet() / 2 + log(sFaWeight_[j](m)) - 0.5 * VecVec(mu_ji, invSigma_mu_ji.Row(m)));
    }
  }

  // PreCompute For diagonal background model
	if (gsDirectOpt_.isPreSelectGaussDirect_ == true) {
		invSigma_d_.resize(num_states_);
		SpMatrix<BaseFloat> fullCov(D);
		for (int i = 0; i < num_states_; ++i) {
			const std::vector<int>& faIndex = sFaIndex_[i];
			int M = faIndex.size();
			invSigma_d_[i].Resize(M, D);
			for (int j = 0; j < M; ++j) {
				GetCov(i, j, &fullCov);
				kaldi::SubVector<BaseFloat> sv = invSigma_d_[i].Row(j);
				sv.CopyDiagFromSp(fullCov);
				sv.InvertElements();
			}
		}
	}

  isPreCompute_ = true;
}

// Unprecompute for likelihood computation.
void AmMfa2::UnPreCompute()
{
  if (isPreCompute_ == false)
    return ;

  // State level
  means_.resize(0);
  gconst_ji_.resize(0);
  invSigma_mu_ji_.resize(0);

  // direct precompute level
  invSigma_d_.resize(0);

  isPreCompute_ = false;
}


/// Computes the top-scoring Gaussian indices (used for pruning of later
/// stages of computation).
void AmMfa2::GaussianSelectionForStateDirect(const int32 state, const VectorBase<BaseFloat> &data,
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
      SubVector<BaseFloat> invSigma = invSigma_d_[state].Row(i);
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

// Get the mean matrix of a specified state
void AmMfa2::GetMeans(int32 state, Matrix<BaseFloat>* pMeans)const
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
void AmMfa2::GetCov(int32 state, int32 comp, SpMatrix<BaseFloat>* pCov)const
{
  KALDI_ASSERT(state >= 0 && state < num_states_);
  KALDI_ASSERT(comp >= 0 && comp < sFaIndex_[state].size());
  KALDI_ASSERT(pCov != NULL);
  if (pCov->NumRows() != dim_)
    pCov->Resize(dim_);
  pCov->CopyFromSp(sFaInvSigma_[state][comp]);
  pCov->Invert();
}

// Calculate the log likelihood of one state
BaseFloat AmMfa2::LogLikelihood(const int32 state, const VectorBase<BaseFloat> &data,
                               Vector<BaseFloat> *post/* = NULL */,  const std::vector<int32>* gselect/* = NULL */) const
{
  KALDI_ASSERT(isPreCompute_ == true);

  int nFA = sFaIndex_[state].size();

  // Doing Gaussian preselection using a direct method
  std::vector<int32> gselect_tmp;
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

	 // perform gselect
    if (gselect != NULL && std::find(gselect->begin(), gselect->end(), i) == gselect->end())
    {
      llk(i) = kLogZeroBaseFloat;
      ++ pruned_cnt;
      continue;
    }

    x = data;
    llk(i) = gconst_ji_[state](i);
    llk(i) += VecVec(x, invSigma_mu_ji_[state].Row(i));
    llk(i) += -0.5 * VecSpVec(x, sFaInvSigma_[state][i], x);
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
