// mfa2/estimate-am-mfa2.cc

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
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <iostream>

#include "matrix/kaldi-gpsr.h"
#include "matrix/kaldi-graphical-lasso.h"
#include "mfa2/am-mfa2.h"
#include "mfa2/estimate-am-mfa2.h"

namespace kaldi {

void MleAmMfa2Accs::Read(std::istream &in_stream, bool binary, bool add)
{
  int32 feature_dim, num_states;
  double total_frames, total_like;
  std::string token;

  ExpectToken(in_stream, binary, "<AMMFA2ACCS>");
  ExpectToken(in_stream, binary, "<FEADIM>");
  ReadBasicType(in_stream, binary, &feature_dim);
  ExpectToken(in_stream, binary, "<NUMSTATES>");
  ReadBasicType(in_stream, binary, &num_states);

  ExpectToken(in_stream, binary, "<TOTFRAMES>");
  ReadBasicType(in_stream, binary, &total_frames);
  ExpectToken(in_stream, binary, "<TOTLIKE>");
  ReadBasicType(in_stream, binary, &total_like);

  if (add == false)
  {
    feature_dim_ = feature_dim;
    num_states_ = num_states;
    total_frames_ = total_frames;
    total_like_ = total_like;
    s0_.resize(num_states_);
    s1_.resize(num_states_);
    s2_.resize(num_states_);
  }
  else
  {
    // match check
    if (num_states_ != 0 || feature_dim_ != 0 )
    {
      if (num_states != num_states_ || feature_dim != feature_dim_)
        KALDI_ERR << "MleAmMfaAccs::Read, num_states or feature_dim mismatch, "
                  << num_states_ << ", " << feature_dim_
                  << " vs. "
                  << num_states << ", " << feature_dim
                  << " (mixing accs from different models?)";
    }
    else
    {
      feature_dim_ = feature_dim;
      num_states_ = num_states;
      total_frames_ = total_frames;
      total_like_ = total_like;
      s0_.resize(num_states_);
      s1_.resize(num_states_);
      s2_.resize(num_states_);
    }

      // add total frame count and llk
    total_frames_ += total_frames;
    total_like_ += total_like;
  }

  int32 j = 0;
  ReadToken(in_stream, binary, &token);
  while (token != "</AMMFA2ACCS>") {
    if (token == "<STATEACCS>") {
      ReadToken(in_stream, binary, &token);
      while(token != "</STATEACCS>") {
        if (token == "<S0>") {
          s0_[j].Read(in_stream, binary, add);
        } else if (token == "<S1>") {
          s1_[j].Read(in_stream, binary, add);
        } else if (token == "<S2>"){
        	int32 num_comp = 0;
        	ExpectToken(in_stream, binary, "<NUMCOMP>");
        	ReadBasicType(in_stream, binary, &num_comp);
        	s2_[j].resize(num_comp);
        	for(int m = 0; m < num_comp; ++ m)
        		s2_[j][m].Read(in_stream, binary, add);
        } else {
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        }
        ReadToken(in_stream, binary, &token);
      }
      ++ j;
      ReadToken(in_stream, binary, &token);
    }
  }

  if (j != num_states_)
    KALDI_WARN << "MleAmMfaAccs::Read, find fewer states than expected, " << j << ", vs " << num_states_ << ".";

}

void MleAmMfa2Accs::Write(std::ostream &out_stream, bool binary) const
{
  // write basic information
  WriteToken(out_stream, binary, "<AMMFA2ACCS>");
  WriteToken(out_stream, binary, "<FEADIM>");
  WriteBasicType(out_stream, binary, feature_dim_);
  WriteToken(out_stream, binary, "<NUMSTATES>");
  WriteBasicType(out_stream, binary, num_states_);

  WriteToken(out_stream, binary, "<TOTFRAMES>");
  WriteBasicType(out_stream, binary, total_frames_);
  WriteToken(out_stream, binary, "<TOTLIKE>");
  WriteBasicType(out_stream, binary, total_like_);

  for(int i = 0; i < num_states_; ++ i)
  {
    WriteToken(out_stream, binary, "<STATEACCS>");
    WriteToken(out_stream, binary, "<S0>");
    s0_[i].Write(out_stream, binary);
    WriteToken(out_stream, binary, "<S1>");
    s1_[i].Write(out_stream, binary);
    WriteToken(out_stream, binary, "<S2>");
    int32 num_comp = s2_[i].size();
    WriteToken(out_stream, binary, "<NUMCOMP>");
    WriteBasicType(out_stream, binary, num_comp);
    for(int j = 0; j < num_comp; ++ j)
    	s2_[i][j].Write(out_stream, binary);
    WriteToken(out_stream, binary, "</STATEACCS>");
  }

  WriteToken(out_stream, binary, "</AMMFA2ACCS>");
}

/// Checks the various accumulators for correct sizes given a model. With
/// wrong sizes, assertion failure occurs. When the show_properties argument
/// is set to true, dimensions and presence/absence of the various
/// accumulators are printed. For use when accumulators are read from file.
void MleAmMfa2Accs::Check(const AmMfa2 &model, bool show_properties/* = true */) const
{
  KALDI_ASSERT(num_states_ == model.NumStates());
  KALDI_ASSERT(feature_dim_ == model.FeatureDim());

  KALDI_ASSERT(s0_.size() == num_states_);
  KALDI_ASSERT(s1_.size() == num_states_);
  KALDI_ASSERT(s2_.size() == num_states_);

  for(int32 i = 0; i < num_states_; ++ i)
  {
    KALDI_ASSERT(s0_[i].Dim() == model.NumComps(i));
    KALDI_ASSERT(s1_[i].NumRows() == model.NumComps(i));
    KALDI_ASSERT(s1_[i].NumCols() == feature_dim_);
  }

  if (show_properties == true)
  {
    KALDI_LOG << "MleAmMfaAccs: num_states = " << num_states_
        << ", feature_dim = " << feature_dim_;
  }

}

/// Resizes the accumulators to the correct sizes given the model. The flags
/// argument control which accumulators to resize.
void MleAmMfa2Accs::ResizeAccumulators(const AmMfa2 &model)
{
  feature_dim_ = model.FeatureDim();
  num_states_ = model.NumStates();

  s0_.resize(num_states_);
  s1_.resize(num_states_);
  s2_.resize(num_states_);
  for(int i = 0; i < num_states_; ++ i)
  {
    int nFA = model.NumComps(i);
    s0_[i].Resize(nFA);
    s1_[i].Resize(nFA, feature_dim_);
    int num_comp = model.NumComps(i);
    s2_[i].resize(num_comp);
    for(int j = 0; j < num_comp; ++ j)
    	s2_[i][j].Resize(feature_dim_);
  }

}

BaseFloat MleAmMfa2Accs::Accumulate(const AmMfa2 &model,
                                    const VectorBase<BaseFloat> &data,
                                    int32 state_index,
                                    BaseFloat weight,
                                    AmMfaUpdateFlagsType flags, const std::vector<int32>* gselect/* = NULL*/)
{
  // Calculate Gaussian posteriors and collect statistics
  Vector<BaseFloat> posteriors;
  BaseFloat log_like = model.LogLikelihood(state_index, data, &posteriors, gselect);
  posteriors.Scale(weight);
  BaseFloat count = AccumulateFromPosteriors(model, posteriors, data, state_index, flags);
  total_like_ += count * log_like;
  total_frames_ += count;
  return log_like;
}

BaseFloat MleAmMfa2Accs::AccumulateFromPosteriors(const AmMfa2 &model,
                                     const Vector<BaseFloat> &posteriors,
                                     const VectorBase<BaseFloat> &data,
                                     int32 state_index,
                                     AmMfaUpdateFlagsType flags)
{
  // Do some check
  KALDI_ASSERT(state_index >= 0 && state_index < this->NumStates());
  KALDI_ASSERT(data.Dim() == this->FeatureDim());

  // gamma_ is needed for all parameters
  s0_[state_index].AddVec(1.0, posteriors);

  // first order statistics is needed for y, M and Sigama
  if (flags & kAmMfaPhoneVectors )
  {
      s1_[state_index].AddVecVec(1.0, posteriors, data);
  }

  // second order statistics is needed for Sigma
	if (flags & kAmMfaCovarianceMatrix) {
		const std::vector<int32>& faIndex = model.sFaIndex_[state_index];
		Vector<BaseFloat> data2;
		Matrix<BaseFloat> means;
		model.GetMeans(state_index, &means);
		for (int k = 0; k < faIndex.size(); ++k) {
			data2 = data;
			data2.AddVec(-1, means.Row(k));
			s2_[state_index][k].AddVec2(posteriors(k), data2);
		}
	}

  return posteriors.Sum();
}

double MleAmMfa2Updater::Update(const MleAmMfa2Accs &accs, AmMfa2 *model, AmMfaUpdateFlagsType flags)
{
  accs.Check(*model, false);

  double tot_impr = 0.0;

  /// update phone vectors
  if (flags & kAmMfaPhoneVectors)
  {
    std::cout << "Update phone vectors (y_ji).\n";
    tot_impr += UpdatePhoneVectors(accs, model);
  }
  /// update phone weights
  if (flags & kAmMfaPhoneWeights)
  {
    std::cout << "Update phone weights (pi_ji).\n";
    tot_impr += UpdatePhoneWeights(accs, model);
  }
  /// update phone projections
  if (flags & kAmMfaPhoneProjections)
  {
    KALDI_ERR << "The AmMfa2 acoustic model doesn't support to update the phone projections.\n "
    		<< "The phone projection matrix must be copied from a normal AmMfa acoustic model.";
  }

  /// update covariance matrices
  if (flags & kAmMfaCovarianceMatrix)
  {
    std::cout << "Update MFA covariance matrix (Sigma_i).\n";
    tot_impr += UpdateCovarianceMatrix(accs, model);
  }

 /// update means
  if (flags & kAmMfaFAMeans)
  {
	  KALDI_ERR << "The AmMfa2 acoustic model doesn't support to update the MFA mean vectors.\n "
	      		<< "The  MFA mean vectors must be copied from a normal AmMfa acoustic model.";
  }

  // if update the weights, then shrink the model
  if (flags & kAmMfaPhoneWeights)
    ShrinkAmMfa2(model);

  return tot_impr;
}

/// Shrink the AmMfa model, remove the zero weights and the corresponding factors for each state
void MleAmMfa2Updater::ShrinkAmMfa2(AmMfa2* model, BaseFloat minW/* = 1.0e-9 */)
{
  int S = model->NumStates();
  std::vector<int32> reservedVec;

  for(int j = 0; j < S; ++ j)
  {
    // record the reserved index
    reservedVec.clear();
    const std::vector<int32>& faIndex = model->sFaIndex_[j];
    const Vector<BaseFloat>& weights = model->sFaWeight_[j];
    for(int k = 0; k < faIndex.size(); ++ k)
    {
      // reserve the weights which are larger than the tol
      if (weights(k) >= minW)
        reservedVec.push_back(k);
      else
        continue;
    }

    // if the size of the weights is unchanged, not shrink
    if (reservedVec.size() == faIndex.size())
      continue;

    KALDI_LOG << "Reduce #Gaussian from " << faIndex.size() << " to "
            << reservedVec.size() << " for #pdf " << j;

    // record the new index, weights and locations
    int newNum = reservedVec.size();
    std::vector<int32> newIndex(newNum);
    Vector<BaseFloat> newWeights(newNum);
    std::vector<Vector<BaseFloat> > newLocations(newNum);
    for(int k = 0; k < newNum; ++ k)
    {
      int l = reservedVec[k];
      newIndex[k] = faIndex[l];
      newWeights(k) = weights(l);
      newLocations[k] = model->sFaLocation_[j][l];
    }

    // copy the new index, weights and locations
    model->sFaIndex_[j] = newIndex;
    model->sFaWeight_[j] = newWeights;
    model->sFaLocation_[j] = newLocations;
  }

}

/// Shrink the model by MFA posteriors sum vectors
void MleAmMfa2Updater::ShrinkAmMfa2(AmMfa2* model, const Matrix<BaseFloat>& mfa_post_sum_mat, const int32 maxComp)
{
  int S = model->NumStates(), M = model->GetMFA().NumComps();
  KALDI_ASSERT(mfa_post_sum_mat.NumRows() == model->NumStates());
  KALDI_ASSERT(mfa_post_sum_mat.NumCols() == M);
  std::vector<int32> reservedVec;
  std::vector<BaseFloat> reservedCnt;
  for(int j = 0; j < S; ++ j)
  {
    const std::vector<int32>& faIndex = model->sFaIndex_[j];
    //const Vector<BaseFloat>& weights = model->sFaWeight_[j];

    const SubVector<BaseFloat> post_sum_vec = mfa_post_sum_mat.Row(j);

    Vector<BaseFloat> post_sum_vec_copy(post_sum_vec);
    BaseFloat *ptr = post_sum_vec_copy.Data();
    std::nth_element(ptr, ptr + M - maxComp, ptr + M);
    BaseFloat thresh = ptr[M - maxComp];
    if (thresh == 0.0)
      thresh = (*std::min(ptr, ptr+M));

    /// record the index of the reserved component
    reservedVec.clear();
    reservedCnt.clear();
    for (int32 i = 0; i < faIndex.size(); i++)
    {
      int32 m = faIndex[i];
      if (post_sum_vec(m) >= thresh)  // met threshold for save phase.
      {
        reservedVec.push_back(i);
        reservedCnt.push_back(post_sum_vec(m));
      }
    }

    // if the size of the weights is unchanged, not shrink
    if (reservedVec.size() == faIndex.size())
      continue;
    if (reservedVec.size() == 0)
    {
      KALDI_WARN << "For state " << j << ", the count of reserved components become zero, unchange it.";
      continue;
    }

    KALDI_LOG << "Reduce #Gaussian from " << faIndex.size() << " to "
            << reservedVec.size() << " for #pdf " << j;

    // record the new index, weights and locations
    int newNum = reservedVec.size();
    std::vector<int32> newIndex(newNum);
    Vector<BaseFloat> newWeights(newNum);
    std::vector<Vector<BaseFloat> > newLocations(newNum);
    BaseFloat totalCnt = 0.0;
    for(int k = 0; k < newNum; ++ k)
    {
      int l = reservedVec[k];
      newIndex[k] = faIndex[l];
      newWeights(k) = reservedCnt[k]; // 1.0 / newNum; // Change the weight
      totalCnt += reservedCnt[k];
      newLocations[k] = model->sFaLocation_[j][l];
    }
    newWeights.Scale(1.0 / totalCnt);

    // copy the new index, weights and locations
    model->sFaIndex_[j] = newIndex;
    model->sFaWeight_[j] = newWeights;
    model->sFaLocation_[j] = newLocations;
  }

}

/// kAmMfaPhoneVectors, y_ji
double MleAmMfa2Updater::UpdatePhoneVectors(const MleAmMfa2Accs &accs, AmMfa2 *model)
{
  const MFA& mfa = model->GetMFA();
  int32 dim = model->FeatureDim();

  GpsrConfig gpsrConfig;
  gpsrConfig.gpsr_tau = 10.0;

  for(int j = 0; j < model->NumStates(); ++ j)
  {
    const std::vector<int32>& faIndex = model->sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      BaseFloat gamma_ji = accs.s0_[j](k);

      /// Check the count, if count < s0_thresh, set the local location to be zero.
      if (gamma_ji <= update_options_.s0_thresh_)
        model->sFaLocation_[j][k].Set(0.0);
      else
      {
        Vector<BaseFloat> s_ji(accs.s1_[j].Row(k));
        s_ji.AddVec(-gamma_ji, mfa.fa_info_vec_[i]->mu_);

        int32 localDim = mfa.GetLocalDim(i);
        const Matrix<BaseFloat>& W = mfa.GetLocalBases(i);
        Matrix<BaseFloat> InvSigmaW(dim, localDim);
        InvSigmaW.AddSpMat(1.0, model->sFaInvSigma_[j][k], W, kNoTrans, 0.0);
        Vector<BaseFloat> b(mfa.k_vec_[i]);
        b.AddMatVec(1.0, InvSigmaW, kTrans, s_ji, 0.0);

        SpMatrix<BaseFloat> A(localDim);
        A.SetUnit();
        A.AddMat2Sp(1.0, W, kTrans, model->sFaInvSigma_[j][k], 1.0);
        A.Scale(gamma_ji);

        if (update_options_.use_l1_ == true)
          Gpsr(gpsrConfig, A, b, &(model->sFaLocation_[j][k]), "Estimate phone vector.");
        else
          SolveQuadraticProblem(A, b, SolverOptions(), &(model->sFaLocation_[j][k]));
      }
    }
  }

  return 0;
}

double SolveWeightsShrinkStatHard(const Vector<BaseFloat>& s0, Vector<BaseFloat>* pWeights, int32 max_comp, BaseFloat prune_s0_threshold/* = -1.0 */)
{
  *pWeights = s0;

  // prune the weight by s0 statistics
  int prune_count = 0;
  // this is a test
  if (prune_s0_threshold > 0)
  {
    for(int i = 0; i < pWeights->Dim(); ++ i)
    {
      if ((*pWeights)(i) < prune_s0_threshold)
      {
        (*pWeights)(i) = 0.0;
        ++ prune_count;
      }
    }
  }
  if (prune_count > 0)
    KALDI_LOG << "Prune " << prune_count << " component for one state using s0 threshold.";

  // prune the weight by max component count
  int reserved_dim = pWeights->Dim() - prune_count;
  if (max_comp > 0 && reserved_dim > max_comp)
  {
    std::vector<BaseFloat> vec(pWeights->Dim());
    for(int i = 0; i < pWeights->Dim(); ++ i)
      vec[i] = (*pWeights)(i);
    std::partial_sort(vec.begin(), vec.begin() + max_comp, vec.end(),std::greater<BaseFloat>());
    BaseFloat thresh = vec[max_comp - 1];
    for(int i = 0; i < pWeights->Dim(); ++ i)
      if ((*pWeights)(i) < thresh)
        (*pWeights)(i) = -1.0;//0.0;
    KALDI_LOG << "Prune weight components from " << pWeights->Dim()
              << " to " << max_comp << "(max comp) for one state.";
  }

  // normalize the weight
  BaseFloat wSum = 0.0;
  for(int i = 0; i < pWeights->Dim(); ++ i)
	  if ((*pWeights)(i) > 0.0)
		  wSum += (*pWeights)(i);
  pWeights->Scale(1.0 / wSum);

  return 0.0;
}

double SolveWeightsShrinkThreshold(Vector<BaseFloat>* pWeights, int32 min_comp, int32 max_comp, BaseFloat threshold)
{
  // prune the weight by s0 statistics
  int prune_count = 0;

  // calculate the thresholded weight count
  if (threshold > 0)
  {
    for(int i = 0; i < pWeights->Dim(); ++ i)
    {
      if ((*pWeights)(i) < threshold)
      {
        ++ prune_count;
      }
    }
  }

  // calculate the real weight count
  int reserved_dim = pWeights->Dim() - prune_count;
  if (max_comp > 0 && reserved_dim > max_comp)
    reserved_dim = max_comp;
  else if (min_comp > 0 && reserved_dim < min_comp)
    reserved_dim = min_comp;

  // prune the weight by reserved_dim
  if (reserved_dim < pWeights->Dim())
  {
    std::vector<BaseFloat> vec(pWeights->Dim());
    for(int i = 0; i < pWeights->Dim(); ++ i)
      vec[i] = (*pWeights)(i);
    std::sort(vec.begin(), vec.end(),std::greater<BaseFloat>());
    BaseFloat thresh = vec[reserved_dim - 1];
    int prune_cnt = 0;
    for(int i = 0; i < pWeights->Dim(); ++ i)
    {
      if ((*pWeights)(i) < thresh)
      {
        (*pWeights)(i) = -1.0;//0.0;
        prune_cnt ++;
      }
    }
    // normalize the weight
    BaseFloat wSum = 0.0;
    for(int i = 0; i < pWeights->Dim(); ++ i)
  	  if ((*pWeights)(i) > 0.0)
  		  wSum += (*pWeights)(i);
    pWeights->Scale(1.0 / wSum);

    KALDI_LOG << "Prune weight components from " << pWeights->Dim()
              << " to " << reserved_dim << "  for one state "
              << "(prune " << prune_cnt << "components).";
  }

  return 0.0;
}

double SolveWeightsShrinkFactor(const Vector<BaseFloat>& s0, Vector<BaseFloat>* pWeights, int32 min_comp, int32 max_comp, BaseFloat factor)
{
  *pWeights = s0;

  // calculate the real weight count
  int reserved_dim = pWeights->Dim();
  reserved_dim *= factor;
  if (max_comp > 0 && reserved_dim > max_comp)
    reserved_dim = max_comp;
  if (min_comp > 0 && reserved_dim < min_comp)
    reserved_dim = min_comp;

  // prune the weight by max component count
  if (reserved_dim < pWeights->Dim())
  {
    std::vector<BaseFloat> vec(pWeights->Dim());
    for(int i = 0; i < pWeights->Dim(); ++ i)
      vec[i] = (*pWeights)(i);
    std::sort(vec.begin(), vec.end(),std::greater<BaseFloat>());
    BaseFloat thresh = vec[reserved_dim - 1];
    for(int i = 0; i < pWeights->Dim(); ++ i)
      if ((*pWeights)(i) < thresh)
        (*pWeights)(i) = -1.0;//0.0;
    KALDI_LOG << "Prune weight components from " << pWeights->Dim()
              << " to " << reserved_dim << " for one state.";
  }

  // normalize the weight
  BaseFloat wSum = 0.0;
  for(int i = 0; i < pWeights->Dim(); ++ i)
	  if ((*pWeights)(i) > 0.0)
		  wSum += (*pWeights)(i);
  pWeights->Scale(1.0 / wSum);

  return 0.0;
}

double SolveWeightsFloor(const Vector<BaseFloat>& s0, Vector<BaseFloat>* pWeights, int32 max_comp, BaseFloat floor/* = -1.0 */)
{
  *pWeights = s0;

  // prune the weight count
  int real_comp = pWeights->Dim();
  if (max_comp > 0 && pWeights->Dim() > max_comp)
  {
    std::vector<BaseFloat> vec(pWeights->Dim());
    for(int i = 0; i < pWeights->Dim(); ++ i)
      vec[i] = (*pWeights)(i);
    std::partial_sort(vec.begin(), vec.begin() + max_comp, vec.end(),std::greater<BaseFloat>());
    BaseFloat thresh = vec[max_comp - 1];
    for(int i = 0; i < pWeights->Dim(); ++ i)
      if ((*pWeights)(i) < thresh)
        (*pWeights)(i) = 0.0;
    KALDI_LOG << "Prune weight components from " << pWeights->Dim()
              << " to " << max_comp << "(max comp) for one state.";
    real_comp = max_comp;
  }

  // renormalize and floor the weight
  pWeights->Scale((1.0 - real_comp * floor) / pWeights->Sum());
  for(int i = 0; i < pWeights->Dim(); ++ i)
    if (ApproxEqual((*pWeights)(i), 0.0) == false)
      (*pWeights)(i) += floor;

  return 0.0;
}

BaseFloat CalcThresholdByPercent(const VectorBase<BaseFloat>& v, BaseFloat percentage)
{
  KALDI_ASSERT(percentage >= 0.0 && percentage <= 1.0);
  std::vector<BaseFloat> v_sort;
  for(int32 i = 0; i < v.Dim(); ++ i)
    v_sort.push_back(v(i));
  std::sort(v_sort.begin(), v_sort.end(), std::greater<BaseFloat>());
  BaseFloat tot = v.Sum(), part = 0.0, threshold = 0.0;
  for(int32 i = 0; i < v_sort.size(); ++ i)
  {
    part += v_sort[i];
    if (part / tot >= percentage)
    {
      threshold = v_sort[i];
    }
  }
  return threshold;
}

/// kAmMfaPhoneWeights, w_j
double MleAmMfa2Updater::UpdatePhoneWeights(const MleAmMfa2Accs &accs, AmMfa2 *model)
{
  double tot_impr = 0.0;
  int S = accs.NumStates();
  for(int j = 0; j < S; ++ j)
  {
    const Vector<BaseFloat>& s0_j = accs.s0_[j];
    if (s0_j.Sum() < update_options_.s0_thresh_)
    {
      KALDI_LOG << "state " << j << ": low state occupation detect, this state will not be updated. ";
      continue;
    }
    switch (update_options_.weight_method_)
    {
      case kAmMfaWeightShrinkHard:
        KALDI_LOG << "state " << j << ": solving weight using hard (s0 stat) threshold: " << update_options_.s0_thresh_
            << ", min comp = " << update_options_.min_comp_
            << ", max comp = " << update_options_.max_comp_;
        tot_impr += SolveWeightsShrinkStatHard(s0_j, &(model->sFaWeight_[j]), update_options_.max_comp_, update_options_.s0_thresh_);
        break;
      case kAmMfaWeightShrinkAver:
        {
          KALDI_LOG << "state " << j << ": solving weight using average threshold: " << update_options_.weight_parm_
              << ", min comp = " << update_options_.min_comp_
              << ", max comp = " << update_options_.max_comp_;
          // normalize the weight
          model->sFaWeight_[j] = s0_j;
          model->sFaWeight_[j].Scale(1.0 / s0_j.Sum());
          BaseFloat w_thresh = update_options_.weight_parm_ / (BaseFloat)(model->GetMFA().NumComps());
          KALDI_LOG << "Real threshold = " << w_thresh;
          tot_impr += SolveWeightsShrinkThreshold(&(model->sFaWeight_[j]), update_options_.min_comp_, update_options_.max_comp_, w_thresh);
        }
        break;
      case kAmMfaWeightShrinkSoft:
        {
          KALDI_LOG << "state " << j << ": solving weight using relative threshold: " << update_options_.weight_parm_
              << ", min comp = " << update_options_.min_comp_
              << ", max comp = " << update_options_.max_comp_;
          // normalize the weight
          model->sFaWeight_[j] = s0_j;
          model->sFaWeight_[j].Scale(1.0 / s0_j.Sum());
          BaseFloat w_thresh = CalcThresholdByPercent(model->sFaWeight_[j], update_options_.weight_parm_);
          //w_thresh += 1.0e-10;  // may change this value
          KALDI_LOG << "Real threshold = " << w_thresh;
          tot_impr += SolveWeightsShrinkThreshold(&(model->sFaWeight_[j]), update_options_.min_comp_, update_options_.max_comp_, w_thresh);
        }
        break;
      case kAmMfaWeightFloor:
        {
          KALDI_LOG << "state " << j << ": solving weight using relative weight floor: " << update_options_.weight_parm_ << ", max comp = " << update_options_.max_comp_;
          BaseFloat w_floor = update_options_.weight_parm_ / (BaseFloat)(model->GetMFA().NumComps());
          KALDI_LOG << "Real floor = " << w_floor;
          tot_impr += SolveWeightsFloor(s0_j, &(model->sFaWeight_[j]), update_options_.max_comp_, w_floor);
        }
        break;
      case kAmMfaWeightDirect:
        KALDI_LOG << "state " << j << ": solving weight using direct estimation method, max comp = " << update_options_.max_comp_;
        tot_impr += SolveWeightsShrinkStatHard(s0_j, &(model->sFaWeight_[j]), update_options_.max_comp_, -1.0);
        break;
      case kAmMfaWeightFactor:
        KALDI_LOG << "state " << j << ": solving weight using a factor of " << update_options_.weight_parm_
                  << ", min comp = " << update_options_.min_comp_
                  << ", max comp = " << update_options_.max_comp_;
        tot_impr += SolveWeightsShrinkFactor(s0_j, &(model->sFaWeight_[j]), update_options_.min_comp_, update_options_.max_comp_, update_options_.weight_parm_);
        break;
      case kAmMfaWeightShrinkAver2:
      {
        KALDI_LOG << "state " << j << ": solving weight using real average threshold: " << update_options_.weight_parm_
            << ", min comp = " << update_options_.min_comp_
            << ", max comp = " << update_options_.max_comp_;
        // normalize the weight
        model->sFaWeight_[j] = s0_j;
        model->sFaWeight_[j].Scale(1.0 / s0_j.Sum());
        BaseFloat w_thresh = update_options_.weight_parm_ / (BaseFloat)(model->NumComps(j));
        KALDI_LOG << "Real threshold = " << w_thresh;
        tot_impr += SolveWeightsShrinkThreshold(&(model->sFaWeight_[j]), update_options_.min_comp_, update_options_.max_comp_, w_thresh);
      }
      break;
      case kAmMfaWeightShrinkThreshold: {
        KALDI_LOG << "state " << j
            << ": solving weight using fixed threshold: "
            << update_options_.weight_parm_ << ", min comp = "
            << update_options_.min_comp_ << ", max comp = "
            << update_options_.max_comp_;
        // normalize the weight
        model->sFaWeight_[j] = s0_j;
        model->sFaWeight_[j].Scale(1.0 / s0_j.Sum());
        tot_impr += SolveWeightsShrinkThreshold(&(model->sFaWeight_[j]),
                                                update_options_.min_comp_,
                                                update_options_.max_comp_,
                                                update_options_.weight_parm_);
      }
        break;
    }
  }

  return 0.0;
}

/// kAmMfaCovarianceMatrix, Sigma_i
double MleAmMfa2Updater::UpdateCovarianceMatrix(const MleAmMfa2Accs &accs,
		AmMfa2 *model) {
	int dim = accs.FeatureDim();

	SpMatrix<BaseFloat> Sigma(dim);
	GraphicLassoConfig opt;
	opt.graphlasso_tau = update_options_.glasso_tau_;
	for (int i = 0; i < model->NumStates(); ++i) {
		for (int j = 0; j < model->NumComps(i); ++j) {
			if (accs.s0_[i](j) < update_options_.min_cov_ratio_ * model->FeatureDim()) {
				KALDI_LOG<< "The " << i << "th state's " << j << "th component's covariance matrix is not updated due to small occupation.";
				continue;
			}
			SpMatrix<BaseFloat> origSigma = accs.s2_[i][j];
			origSigma.Scale(1.0 / accs.s0_[i](j));
			 if (origSigma.IsPosDef() == false)
			 {
			        KALDI_WARN << "origSigma is not positive definite!" << "occ = " << accs.s0_[i](j);
			        KALDI_LOG<< "The " << i << "th state's " << j << "th component's covariance matrix is not updated due to small occupation.";
			        continue;
			 }
			GraphicalLasso(origSigma, &Sigma, &(model->sFaInvSigma_[i][j]), opt);
		}
	}

	return 0.0;
}


}
