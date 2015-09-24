// mfa/estimate-am-mfa.cc

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
#include <string>
using std::string;
#include <vector>
using std::vector;
#include <iostream>

#include "matrix/kaldi-gpsr.h"
#include "mfa/am-mfa.h"
#include "mfa/estimate-am-mfa.h"

namespace kaldi {

void MleAmMfaAccs::Read(std::istream &in_stream, bool binary, bool add)
{
  int32 feature_dim, num_states, num_factors, spk_dim;
  double total_frames, total_like;
  std::string token;

  ExpectToken(in_stream, binary, "<AMMFAACCS>");
  ExpectToken(in_stream, binary, "<FEADIM>");
  ReadBasicType(in_stream, binary, &feature_dim);
  ExpectToken(in_stream, binary, "<SPKDIM>");
  ReadBasicType(in_stream, binary, &spk_dim);
  ExpectToken(in_stream, binary, "<NUMSTATES>");
  ReadBasicType(in_stream, binary, &num_states);
  ExpectToken(in_stream, binary, "<NUMFACTORS>");
  ReadBasicType(in_stream, binary, &num_factors);

  ExpectToken(in_stream, binary, "<TOTFRAMES>");
  ReadBasicType(in_stream, binary, &total_frames);
  ExpectToken(in_stream, binary, "<TOTLIKE>");
  ReadBasicType(in_stream, binary, &total_like);

  if (add == false)
  {
    feature_dim_ = feature_dim;
    spk_dim_ = spk_dim;
    num_states_ = num_states;
    num_factors_ = num_factors;
    total_frames_ = total_frames;
    total_like_ = total_like;
    s0_.resize(num_states_);
    s1_.resize(num_states_);
    s2_.resize(num_factors_);
    Z_vec_.resize(num_factors_);
    R_vec_.resize(num_factors_);
  }
  else
  {
    // match check
    if (num_states_ != 0 || feature_dim_ != 0 || num_factors_ != 0 || spk_dim_ != 0)
    {
      if (num_states != num_states_ || feature_dim != feature_dim_ || num_factors != num_factors_ || spk_dim != spk_dim_)
        KALDI_ERR << "MleAmMfaAccs::Read, num_states or feature_dim mismatch, "
                  << num_states_ << ", " << feature_dim_ << ", " << num_factors_
                  << spk_dim_ << " vs. " << num_states << ", "
                  << feature_dim << ", " << num_factors << spk_dim
                  << " (mixing accs from different models?)";
    }
    else
    {
      feature_dim_ = feature_dim;
      spk_dim_ = spk_dim;
      num_states_ = num_states;
      num_factors_ = num_factors;
      total_frames_ = total_frames;
      total_like_ = total_like;
      s0_.resize(num_states_);
      s1_.resize(num_states_);
      s2_.resize(num_factors_);
      Z_vec_.resize(num_factors_);
      R_vec_.resize(num_factors_);
    }

      // add total frame count and llk
    total_frames_ += total_frames;
    total_like_ += total_like;
  }

  int32 j = 0;
  ReadToken(in_stream, binary, &token);
  while (token != "</AMMFAACCS>") {
    if (token == "<STATEACCS>") {
      ReadToken(in_stream, binary, &token);
      while(token != "</STATEACCS>") {
        if (token == "<S0>") {
          s0_[j].Read(in_stream, binary, add);
        } else if (token == "<S1>") {
          s1_[j].Read(in_stream, binary, add);
        } else {
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        }
        ReadToken(in_stream, binary, &token);
      }
      ++ j;
      ReadToken(in_stream, binary, &token);
    }
    else if (token == "<MFAACCS>") {
      ReadToken(in_stream, binary, &token);
      int i = 0;
      while(token != "</MFAACCS>") {
        if (token == "<S2>") {
          s2_[i].Read(in_stream, binary, add);
        } else {
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        }
        ReadToken(in_stream, binary, &token);
        ++ i;
      }
      ReadToken(in_stream, binary, &token);
      if (i != num_factors_)
        KALDI_WARN << "MleAmMfaAccs::Read, find fewer factors than expected, " << i << ", vs " << num_factors_ << ".";
    }
    else if (token == "<SPKSUBSPACEACCS>") {
      for(int i = 0; i < Z_vec_.size(); ++ i)
      {
        Z_vec_[i].Read(in_stream, binary, add);
        R_vec_[i].Read(in_stream, binary, add);
      }
      ExpectToken(in_stream, binary, "</SPKSUBSPACEACCS>");
      ReadToken(in_stream, binary, &token);
    }
  }

  if (j != num_states_)
    KALDI_WARN << "MleAmMfaAccs::Read, find fewer states than expected, " << j << ", vs " << num_states_ << ".";

}

void MleAmMfaAccs::Write(std::ostream &out_stream, bool binary) const
{
  // write basic information
  WriteToken(out_stream, binary, "<AMMFAACCS>");
  WriteToken(out_stream, binary, "<FEADIM>");
  WriteBasicType(out_stream, binary, feature_dim_);
  WriteToken(out_stream, binary, "<SPKDIM>");
  WriteBasicType(out_stream, binary, spk_dim_);
  WriteToken(out_stream, binary, "<NUMSTATES>");
  WriteBasicType(out_stream, binary, num_states_);
  WriteToken(out_stream, binary, "<NUMFACTORS>");
  WriteBasicType(out_stream, binary, num_factors_);

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
    WriteToken(out_stream, binary, "</STATEACCS>");
  }

  WriteToken(out_stream, binary, "<MFAACCS>");
  for(int i = 0; i < num_factors_; ++ i)
  {
    WriteToken(out_stream, binary, "<S2>");
    s2_[i].Write(out_stream, binary);
  }
  WriteToken(out_stream, binary, "</MFAACCS>");

  WriteToken(out_stream, binary, "<SPKSUBSPACEACCS>");
  for(int i = 0; i < Z_vec_.size(); ++ i)
  {
    Z_vec_[i].Write(out_stream, binary);
    R_vec_[i].Write(out_stream, binary);
  }
  WriteToken(out_stream, binary, "</SPKSUBSPACEACCS>");

  WriteToken(out_stream, binary, "</AMMFAACCS>");
}

/// Checks the various accumulators for correct sizes given a model. With
/// wrong sizes, assertion failure occurs. When the show_properties argument
/// is set to true, dimensions and presence/absence of the various
/// accumulators are printed. For use when accumulators are read from file.
void MleAmMfaAccs::Check(const AmMfa &model, bool show_properties/* = true */) const
{
  KALDI_ASSERT(num_states_ == model.NumStates());
  KALDI_ASSERT(feature_dim_ == model.FeatureDim());
  KALDI_ASSERT(num_factors_ == model.GetMFA().NumComps());
  KALDI_ASSERT(num_factors_ == Z_vec_.size());
  KALDI_ASSERT(num_factors_ == R_vec_.size());

  KALDI_ASSERT(s0_.size() == num_states_);
  KALDI_ASSERT(s1_.size() == num_states_);
  KALDI_ASSERT(s2_.size() == num_factors_);

  for(int32 i = 0; i < num_states_; ++ i)
  {
    KALDI_ASSERT(s0_[i].Dim() == model.NumComps(i));
    KALDI_ASSERT(s1_[i].NumRows() == model.NumComps(i));
    KALDI_ASSERT(s1_[i].NumCols() == feature_dim_);
  }
  for(int32 i = 0; i < num_factors_; ++ i)
  {
    KALDI_ASSERT(s2_[i].NumRows() == feature_dim_);
    if (spk_dim_ > 0) {
      KALDI_ASSERT(Z_vec_[i].NumRows() == feature_dim_);
      KALDI_ASSERT(Z_vec_[i].NumCols() == spk_dim_);
      KALDI_ASSERT(R_vec_[i].NumRows() == spk_dim_);
    }
  }

  if (show_properties == true)
  {
    KALDI_LOG << "MleAmMfaAccs: num_states = " << num_states_
        << ", feature_dim = " << feature_dim_
        << ", spk_dim = " << spk_dim_
        << ", num_factors = " << num_factors_;
  }

}

/// Resizes the accumulators to the correct sizes given the model. The flags
/// argument control which accumulators to resize.
void MleAmMfaAccs::ResizeAccumulators(const AmMfa &model)
{
  feature_dim_ = model.FeatureDim();
  num_states_ = model.NumStates();
  num_factors_ = model.GetMFA().NumComps();
  spk_dim_ = model.SpkSpaceDim();

  s0_.resize(num_states_);
  s1_.resize(num_states_);
  s2_.resize(num_factors_);
  for(int i = 0; i < num_states_; ++ i)
  {
    int nFA = model.NumComps(i);
    s0_[i].Resize(nFA);
    s1_[i].Resize(nFA, feature_dim_);
  }
  for(int i = 0; i < num_factors_; ++ i)
    s2_[i].Resize(feature_dim_);


  Z_vec_.resize(num_factors_);
  R_vec_.resize(num_factors_);
  if (spk_dim_ > 0) {
    for (int32 i = 0; i < num_factors_; ++i) {
      Z_vec_[i].Resize(feature_dim_, spk_dim_);
      R_vec_[i].Resize(spk_dim_);
    }
    spkAccsHelper_.spk_s1_.Resize(num_factors_, feature_dim_);
    spkAccsHelper_.spk_s0_.Resize(num_factors_);
  }
}

/// Begin a new speaker, clear the speaker specific temp accumulator
void MleAmMfaAccs::BeginSpkr(const AmMfaPerSpkDerivedVars* pVars, AmMfaUpdateFlagsType flags)
{
  if ((flags & kAmMfaSpeakerProjections) && pVars != NULL && pVars->v_s.Dim() > 0)
  {
    spkAccsHelper_.spk_s1_.SetZero();
    spkAccsHelper_.spk_s0_.SetZero();
  }
}

/// Update the speaker statistics
void MleAmMfaAccs::CommitSpkr(const AmMfaPerSpkDerivedVars* pVars, AmMfaUpdateFlagsType flags)
{
  if ((flags & kAmMfaSpeakerProjections) && pVars != NULL && pVars->v_s.Dim() > 0)
  {
    for(int32 i = 0; i < num_factors_; ++ i)
    {
      Z_vec_[i].AddVecVec(1.0, spkAccsHelper_.spk_s1_.Row(i), pVars->v_s);
      R_vec_[i].AddVec2(spkAccsHelper_.spk_s0_(i), pVars->v_s);
    }
  }
}

BaseFloat MleAmMfaAccs::Accumulate(const AmMfa &model,
                                    const VectorBase<BaseFloat> &data,
                                    const AmMfaPerSpkDerivedVars* pVars,
                                    int32 state_index,
                                    BaseFloat weight,
                                    AmMfaUpdateFlagsType flags, const std::vector<int32>* gselect/* = NULL*/)
{
  // Do some check
  if (flags & (kAmMfaRemoveSpkrProjections & kAmMfaSpeakerProjections))
    KALDI_ERR << "Cannot both remove speaker projection and update speaker projection.";

  // Calculate Gaussian posteriors and collect statistics
  Vector<BaseFloat> posteriors;
  BaseFloat log_like = model.LogLikelihood(state_index, data, pVars, &posteriors, gselect);
  posteriors.Scale(weight);
  BaseFloat count = 0.0;
  if (flags & kAmMfaRemoveSpkrProjections)
    count = AccumulateFromPosteriors(model, posteriors, data, NULL, state_index, flags);
  else
    count = AccumulateFromPosteriors(model, posteriors, data, pVars, state_index, flags);
  total_like_ += count * log_like;
  total_frames_ += count;
  return log_like;
}

BaseFloat MleAmMfaAccs::AccumulateFromPosteriors(const AmMfa &model,
                                     const Vector<BaseFloat> &posteriors,
                                     const VectorBase<BaseFloat> &data,
                                     const AmMfaPerSpkDerivedVars* pVars,
                                     int32 state_index,
                                     AmMfaUpdateFlagsType flags)
{
  // Do some check
  if (flags & (kAmMfaRemoveSpkrProjections & kAmMfaSpeakerProjections))
    KALDI_ERR << "Cannot both remove speaker projection and update speaker projection.";
  KALDI_ASSERT(state_index >= 0 && state_index < this->NumStates());
  KALDI_ASSERT(data.Dim() == this->FeatureDim());

  // gamma_ is needed for all parameters
  s0_[state_index].AddVec(1.0, posteriors);

  // speaker dependent observations
  Matrix<BaseFloat> data_s;
  if (!(flags & kAmMfaRemoveSpkrProjections) && pVars != NULL && pVars->v_s.Dim() > 0)
  {
    const std::vector<int32>& faIndex = model.sFaIndex_[state_index];
    data_s.Resize(faIndex.size(), data.Dim());
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int32 i = faIndex[k];
      data_s.Row(k).AddVec(1.0, data);
      data_s.Row(k).AddVec(-1.0, pVars->o_s.Row(i));
    }
  }

  // first order statistics is needed for y, M and Sigama
  if (flags & (kAmMfaPhoneVectors | kAmMfaPhoneProjections | kAmMfaCovarianceMatrix | kAmMfaSpeakerProjections))
  {
    if (!(flags & kAmMfaRemoveSpkrProjections) && pVars != NULL && pVars->v_s.Dim() > 0)
    {
      Matrix<BaseFloat> data_s2 = data_s;
      data_s2.MulRowsVec(posteriors);
      s1_[state_index].AddMat(1.0, data_s2, kNoTrans);
    }
    else
      s1_[state_index].AddVecVec(1.0, posteriors, data);
  }

  // second order statistics is needed for Sigamma
  if (flags & kAmMfaCovarianceMatrix)
  {
    if (!(flags & kAmMfaRemoveSpkrProjections) && pVars != NULL && pVars->v_s.Dim() > 0)
    {
      const std::vector<int32>& faIndex = model.sFaIndex_[state_index];
      for (int k = 0; k < faIndex.size(); ++ k)
      {
        s2_[faIndex[k]].AddVec2(posteriors(k), data_s.Row(k));
      }
    }
    else
    {
      const std::vector<int32>& faIndex = model.sFaIndex_[state_index];
      for (int k = 0; k < faIndex.size(); ++ k)
      {
        s2_[faIndex[k]].AddVec2(posteriors(k), data);
      }
    }
  }

  if ((flags & kAmMfaSpeakerProjections) && pVars != NULL && pVars->v_s.Dim() > 0)
  {
    Matrix<BaseFloat> means;
    model.GetMeans(state_index, &means);
    means.AddVecToRows(-1.0, data);
    means.MulRowsVec(posteriors);

    const std::vector<int32>& faIndex = model.sFaIndex_[state_index];

    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int32 i = faIndex[k];
      spkAccsHelper_.spk_s0_(i) += posteriors(k);
      spkAccsHelper_.spk_s1_.Row(i).AddVec(-1.0, means.Row(k));
    }
  }

  return posteriors.Sum();
}

double MleAmMfaUpdater::Update(const MleAmMfaAccs &accs, AmMfa *model, AmMfaUpdateFlagsType flags)
{
  accs.Check(*model, false);

  model->mfa_.PreCompute();
  model->PreCompute();

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
 /*
 if (flags & (kAmMfaPhoneProjections | kAmMfaCovarianceMatrix))
  {
    Compute_Y_i(accs, *model);
  }
  if (flags & kAmMfaPhoneProjections)
  {
    Compute_Q_i(accs, *model);
  } */
  if (flags & kAmMfaPhoneProjections)
  {
    std::cout << "Update phone projections (M_i).\n";
    Compute_s0_i_s1_i(accs, *model);
    Compute_Y_i(accs, *model);
   	Compute_Q_i(accs, *model);
    tot_impr += UpdatePhoneProjections(accs, model);
  }

  /// update covariance matrices
  /*if (flags & (kAmMfaPhoneProjections | kAmMfaCovarianceMatrix | kAmMfaFAMeans | kAmMfaSpeakerProjections))
  {
    Compute_s0_i_s1_i(accs, *model);
  }
  if (flags & kAmMfaCovarianceMatrix)
  {
    Compute_S_i_S_means_i(accs, *model);
  }*/
  if (flags & kAmMfaCovarianceMatrix)
  {
    std::cout << "Update MFA covariance matrix (Sigma_i).\n";
    Compute_Y_i(accs, *model);
	Compute_s0_i_s1_i(accs, *model);
	Compute_S_i_S_means_i(accs, *model);
    tot_impr += UpdateCovarianceMatrix(accs, model);
  }

 /// update means
 /* if (flags & kAmMfaFAMeans)
  {
    Compute_s1_means_i(accs, *model);
  }*/
  if (flags & kAmMfaFAMeans)
  {
    std::cout << "Update MFA means(mu_i).\n";
    Compute_s0_i_s1_i(accs, *model);
  	Compute_s1_means_i(accs, *model);
    tot_impr += UpdateMFAMeans(accs, model);
  }

  /// update speaker projections
  if (flags & kAmMfaSpeakerProjections)
  {
    std::cout << "Update Speaker projections (N_i).\n";
    Compute_s0_i_s1_i(accs, *model);
    tot_impr += UpdateSpeakerProjections(accs, model);
  }

  // if update the weights, then shrink the model
  if (flags & kAmMfaPhoneWeights)
    ShrinkAmMfa(model);

  // clear accumulator memory
  if (flags & (kAmMfaPhoneProjections | kAmMfaCovarianceMatrix))
  {
    Free_Y_i();
  }
  if (flags & kAmMfaPhoneProjections)
  {
    Free_Q_i();
  }
  if (flags & (kAmMfaCovarianceMatrix | kAmMfaFAMeans))
  {
    Free_s0_i_s1_i();
  }
  if (flags & kAmMfaCovarianceMatrix)
  {
    Free_S_i_S_means_i();
  }
  if (flags & kAmMfaFAMeans)
  {
    Free_s1_means_i();
  }

  return tot_impr;
}

/// Shrink the AmMfa model, remove the zero weights and the corresponding factors for each state
void MleAmMfaUpdater::ShrinkAmMfa(AmMfa* model, BaseFloat minW/* = 1.0e-9 */)
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
void MleAmMfaUpdater::ShrinkAmMfa(AmMfa* model, const Matrix<BaseFloat>& mfa_post_sum_mat, const int32 maxComp)
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
double MleAmMfaUpdater::UpdatePhoneVectors(const MleAmMfaAccs &accs, AmMfa *model)
{
  const MFA& mfa = model->GetMFA();

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

        Vector<BaseFloat> b(mfa.k_vec_[i]);
        b.AddMatVec(1.0, mfa.pre_data_vec_[i]->Sigma1W_, kTrans, s_ji, 0.0);

        SpMatrix<BaseFloat> A(mfa.pre_data_vec_[i]->M_);
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
              << " to " << reserved_dim << " for one state "
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
double MleAmMfaUpdater::UpdatePhoneWeights(const MleAmMfaAccs &accs, AmMfa *model)
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

void MleAmMfaUpdater::Compute_Y_i_s(const MleAmMfaAccs &accs, const AmMfa& model, std::vector<Matrix<BaseFloat> >* p_Y_i_vec)
{
	KALDI_ASSERT(p_Y_i_vec != NULL);
	std::vector<Matrix<BaseFloat> >& Y_i_vec = *p_Y_i_vec;
	
	int dim = accs.FeatureDim();
  int I = accs.NumFactors();
  int S = accs.NumStates();
  const MFA& mfa = model.GetMFA();

  // allocate Y_i_vec
  Y_i_vec.resize(I);
  for(int i = 0; i < I; ++ i)
  {
    int lDim = mfa.GetLocalDim(i);
    Y_i_vec[i].Resize(dim, lDim);
  }

  // calculate Y_i
  for(int j = 0; j < S; ++ j)
  {
    const std::vector<int32>& faIndex = model.sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      BaseFloat gamma_ji = accs.s0_[j](k);
      Vector<BaseFloat> s_ji(accs.s1_[j].Row(k));
      s_ji.AddVec(-gamma_ji, mfa.GetLocalCenter(i));
      Y_i_vec[i].AddVecVec(1.0, s_ji, model.sFaLocation_[j][k]);
    }
  }
}

void MleAmMfaUpdater::Compute_Y_i(const MleAmMfaAccs &accs, const AmMfa& model)
{
  if (is_Y_i_computed_ == true)
    return ;

  MleAmMfaUpdater::Compute_Y_i_s(accs, model, &Y_i_vec_);

  is_Y_i_computed_ = true;
}

void MleAmMfaUpdater::Free_Y_i()
{
  Y_i_vec_.clear();
  is_Y_i_computed_ = false;
}

void MleAmMfaUpdater::Compute_Q_i_s(const MleAmMfaAccs &accs, const AmMfa& model, std::vector<SpMatrix<BaseFloat> >* p_Q_i_vec)
{
	KALDI_ASSERT(p_Q_i_vec != NULL);
	std::vector<SpMatrix<BaseFloat> >& Q_i_vec = *p_Q_i_vec;
	
	int I = accs.NumFactors();
  int S = accs.NumStates();
  const MFA& mfa = model.GetMFA();

  Q_i_vec.resize(I);
  for(int i = 0; i < I; ++ i)
  {
    int lDim = mfa.GetLocalDim(i);
    Q_i_vec[i].Resize(lDim);
  }

  // calculate Q_i
  for(int j = 0; j < S; ++ j)
  {
    const std::vector<int32>& faIndex = model.sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      BaseFloat gamma_ji = accs.s0_[j](k);
      const Vector<BaseFloat>& y_ji = model.sFaLocation_[j][k];
      Q_i_vec[i].AddVec2(gamma_ji, y_ji);
    }
  }
}

void MleAmMfaUpdater::Compute_Q_i(const MleAmMfaAccs &accs, const AmMfa& model)
{
  if (is_Q_i_computed_ == true)
    return ;

  MleAmMfaUpdater::Compute_Q_i_s(accs, model, &Q_i_vec_);

  is_Q_i_computed_ = true;

}

void MleAmMfaUpdater::Free_Q_i()
{
  Q_i_vec_.clear();
  is_Q_i_computed_ = false;
}

void MleAmMfaUpdater::Compute_s0_i_s1_i_s(const MleAmMfaAccs &accs, const AmMfa& model, Vector<BaseFloat>* p_s0_i, Matrix<BaseFloat>* p_s1_i)
{
	KALDI_ASSERT(p_s0_i != NULL && p_s1_i != NULL);
	Vector<BaseFloat>& s0_i = *p_s0_i;
	Matrix<BaseFloat>& s1_i = *p_s1_i;
	
	int dim = accs.FeatureDim();
  int I = accs.NumFactors();
  int S = accs.NumStates();

  s0_i.Resize(I);
  s1_i.Resize(I, dim);
  // calculate s0_i, s1_i
  for(int j = 0; j < S; ++ j)
  {
    const std::vector<int32>& faIndex = model.sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      s0_i(i) += accs.s0_[j](k);
      s1_i.Row(i).AddVec(1.0, accs.s1_[j].Row(k));
    }
  }
}

void MleAmMfaUpdater::Compute_s0_i_s1_i(const MleAmMfaAccs &accs, const AmMfa& model)
{
  if (is_s0_i_s1_i_computed_ == true)
    return ;
  
  MleAmMfaUpdater::Compute_s0_i_s1_i_s(accs, model, &s0_i_, &s1_i_);
  
  is_s0_i_s1_i_computed_ = true;
}

void MleAmMfaUpdater::Free_s0_i_s1_i()
{
  s0_i_.Resize(0);
  s1_i_.Resize(0, 0);
  is_s0_i_s1_i_computed_ = false;
}

void MleAmMfaUpdater::Compute_S_i_S_means_i_s(const MleAmMfaAccs &accs, const AmMfa& model, const Vector<BaseFloat>& s0_i, const Matrix<BaseFloat>& s1_i,
               std::vector<SpMatrix<BaseFloat> >* p_S_i_vec, std::vector<SpMatrix<BaseFloat> >* p_S_means_i_vec)
{
	KALDI_ASSERT(p_S_i_vec != NULL && p_S_means_i_vec != NULL);
	std::vector<SpMatrix<BaseFloat> >& S_i_vec = *p_S_i_vec;
	std::vector<SpMatrix<BaseFloat> >& S_means_i_vec = *p_S_means_i_vec;
	
	int dim = accs.FeatureDim();
  int I = accs.NumFactors();
  int S = accs.NumStates();
  const MFA& mfa = model.GetMFA();

  S_i_vec.resize(I);
  S_means_i_vec.resize(I);
  for(int i = 0; i < I; ++ i)
  {
    S_i_vec[i].Resize(dim);
    S_means_i_vec[i].Resize(dim);
  }

  // calculate S_means_i_vec_
  for(int j = 0; j < S; ++ j)
  {
    const std::vector<int32>& faIndex = model.sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      S_means_i_vec[i].AddVec2(accs.s0_[j](k), model.means_[j].Row(k));
    }
  }

  // compute S_i_vec_
  for(int i = 0; i < I; ++ i)
  {
    const Vector<BaseFloat>& mu_i = mfa.GetLocalCenter(i);
    S_i_vec[i].CopyFromSp(accs.s2_[i]);
    S_i_vec[i].AddVecVec(-1.0, mu_i, s1_i.Row(i));
    S_i_vec[i].AddVec2(s0_i(i), mu_i);
  }
	
}

void MleAmMfaUpdater::Compute_S_i_S_means_i(const MleAmMfaAccs &accs, const AmMfa& model)
{
  if (is_S_i_S_means_i_computed_ == true)
      return ;
  if (is_s0_i_s1_i_computed_ == false)
    Compute_s0_i_s1_i(accs, model);
  
  MleAmMfaUpdater::Compute_S_i_S_means_i_s(accs, model, s0_i_, s1_i_, &S_i_vec_, &S_means_i_vec_);

  is_S_i_S_means_i_computed_ = true;
}

void MleAmMfaUpdater::Free_S_i_S_means_i()
{
  S_i_vec_.clear();
  S_means_i_vec_.clear();
  is_S_i_S_means_i_computed_ = false;

}

void MleAmMfaUpdater::Compute_s1_means_i_s(const MleAmMfaAccs &accs, const AmMfa& model, Matrix<BaseFloat>* p_s1_means_i)
{
	KALDI_ASSERT(p_s1_means_i != NULL);
	Matrix<BaseFloat>& s1_means_i = *p_s1_means_i;
	
	int dim = accs.FeatureDim();
  int I = accs.NumFactors();
  int S = accs.NumStates();

  s1_means_i.Resize(I, dim);
  // calculate mu_means_i_vec_
  for(int j = 0; j < S; ++ j)
  {
    const std::vector<int32>& faIndex = model.sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      s1_means_i.Row(i).AddVec(accs.s0_[j](k), model.means_[j].Row(k));
    }
  }
}

void MleAmMfaUpdater::Compute_s1_means_i(const MleAmMfaAccs &accs, const AmMfa& model)
{
  if (is_s1_means_i_computed_ == true)
        return ;

  MleAmMfaUpdater::Compute_s1_means_i_s(accs, model, &s1_means_i_);

  is_s1_means_i_computed_ = true;
}

void MleAmMfaUpdater::Free_s1_means_i()
{
  s1_means_i_.Resize(0, 0);
  is_s1_means_i_computed_ = false;
}

/// kAmMfaPhoneProjections, M_i
double MleAmMfaUpdater::UpdatePhoneProjections(const MleAmMfaAccs &accs, AmMfa *model)
{
  int I = accs.NumFactors();
  MFA* pMfa = &(model->mfa_);

  // Check the AmMfa model is precomputed
  KALDI_ASSERT(model->isPreCompute_ == true);

  // Check Y_i is computed
  KALDI_ASSERT(is_Y_i_computed_ == true);

  // Check Q_i is computed
  KALDI_ASSERT(is_Q_i_computed_ == true);

  // update M_i
  for(int i = 0; i < I; ++ i)
  {
    // if s0 is very small, skip the updation
    if (s0_i_(i) < update_options_.s0_thresh_)
      continue;
    SolveQuadraticMatrixProblem(Q_i_vec_[i], Y_i_vec_[i], model->invSigma_[i], SolverOptions(), &(pMfa->fa_info_vec_[i]->W_));
  }

  return 0.0;
}


/// kAmMfaCovarianceMatrix, Sigma_i
double MleAmMfaUpdater::UpdateCovarianceMatrix(const MleAmMfaAccs &accs, AmMfa *model)
{
  int dim = accs.FeatureDim();
  int I = accs.NumFactors();
  MFA* pMfa = &(model->mfa_);

  // Check Y_i is computed
  KALDI_ASSERT(is_Y_i_computed_ == true);

  // Check S_i and S_means_i are computed
  KALDI_ASSERT(is_S_i_S_means_i_computed_ == true);

  Covariance floorCov;
  switch(pMfa->covType_)
  {
    case DIAG:
      floorCov.diagCov_.Resize(dim);
      break;
    case FULL:
      floorCov.fullCov_.Resize(dim);
      break;
    default:
      KALDI_ERR << "Invalid covariance type.";
      break;
  }

  // update Sigma_i
  for(int i = 0; i < I; ++ i)
  {
    // if s0 is very small, skip the updation
    if (s0_i_(i) < update_options_.s0_thresh_)
    {
      KALDI_LOG << "The " << i << "th factor model (Sigma) is not updated due to small occupation.";
      continue;
    }

    // accumulate floor
    if (pMfa->covType_ == DIAG)
      floorCov.diagCov_.AddVec(s0_i_(i), pMfa->fa_info_vec_[i]->sigma_.diagCov_);
    else
      floorCov.fullCov_.AddSp(s0_i_(i), pMfa->fa_info_vec_[i]->sigma_.fullCov_);

    Matrix<BaseFloat> Sigmma_i(dim, dim);
    Sigmma_i.CopyFromSp(S_i_vec_[i]);
    //tpVec.CopyDiagFromMat(Sigmma_i);

    Sigmma_i.AddSp(1.0, S_means_i_vec_[i]);
    //tpVec.CopyDiagFromMat(Sigmma_i);

    Sigmma_i.AddMatMat(-1.0, pMfa->fa_info_vec_[i]->W_, kNoTrans, Y_i_vec_[i], kTrans, 1.0);
    Sigmma_i.AddMatMat(-1.0, Y_i_vec_[i], kNoTrans, pMfa->fa_info_vec_[i]->W_, kTrans, 1.0);

    Sigmma_i.Scale(1.0 / s0_i_(i));

    // copy the diagonal elements to MFA model
    if (pMfa->covType_ == DIAG)
      pMfa->fa_info_vec_[i]->sigma_.diagCov_.CopyDiagFromMat(Sigmma_i);
    else
      pMfa->fa_info_vec_[i]->sigma_.fullCov_.CopyFromMat(Sigmma_i);
  }

  /// Floor the covariance matrix, 0.2 is a magic number !!!
  if (pMfa->covType_ == DIAG)
  {
    floorCov.diagCov_.Scale(0.2 / s0_i_.Sum());
    for(int i = 0; i < I; ++ i)
    {
      pMfa->fa_info_vec_[i]->sigma_.diagCov_.ApplyFloor(floorCov.diagCov_);
    }
  }
  else
  {
    floorCov.fullCov_.Scale(0.2 / s0_i_.Sum());
    for(int i = 0; i < I; ++ i)
    {
      pMfa->fa_info_vec_[i]->sigma_.fullCov_.ApplyFloor(floorCov.fullCov_);
    }
  }

  return 0.0;
}

/// kAmMfaFAMeans
double MleAmMfaUpdater::UpdateMFAMeans(const MleAmMfaAccs &accs, AmMfa *model)
{
  int dim = accs.FeatureDim();
  int I = accs.NumFactors();

  // Check s0_i and s1_i are computed
  KALDI_ASSERT(is_s0_i_s1_i_computed_ == true);

  // Check s1_means_i is computed
  KALDI_ASSERT(is_s1_means_i_computed_ == true);

  // update means of the MFA components
  for(int i = 0; i < I; ++ i)
  {
    // if s0 is very small, skip the updation
    if (s0_i_(i) < update_options_.s0_thresh_)
    {
      KALDI_LOG << "The " << i << "th factor model (mu) is not updated due to small occupation.";
      continue;
    }

    Vector<BaseFloat> mu(dim);
    mu.CopyFromVec(s1_i_.Row(i));
    mu.AddVec(-1, s1_means_i_.Row(i));
    mu.Scale(1.0 / s0_i_(i));
    model->mfa_.fa_info_vec_[i]->mu_.CopyFromVec(mu);
  }

  return 0.0;
}

/// kAmMfaSpeakerProjections
double MleAmMfaUpdater::UpdateSpeakerProjections(const MleAmMfaAccs &accs, AmMfa *model)
{
  int I = accs.NumFactors();
  // Check the AmMfa model is precomputed
  KALDI_ASSERT(model->isPreCompute_ == true);

  // update N_i
  for(int i = 0; i < I; ++ i)
  {
    // if s0 is very small, skip the updation
    if (s0_i_(i) < update_options_.s0_thresh_)
      continue;
    SolveQuadraticMatrixProblem(accs.R_vec_[i], accs.Z_vec_[i], model->invSigma_[i], SolverOptions(), &model->N_[i]);
  }

  return 0.0;
}

MleAmMfaSpeakerAccs::MleAmMfaSpeakerAccs(const AmMfa &model)
{
  int I = model.NumSubspace();
  int D = model.FeatureDim();
  int K = model.SpkSpaceDim();

  KALDI_ASSERT(K != 0);

  H_spk_.resize(I);
  NtransSigmaInv_.resize(I);
  for (int32 i = 0; i < I; i++) {
    // Eq. (82): H_{i}^{spk} = N_{i}^T \Sigma_{i}^{-1} N_{i}
    H_spk_[i].Resize(K);
    H_spk_[i].AddMat2Sp(1.0, model.N_[i],
                        kTrans, model.GetSigmaInvForSubspace(i), 0.0);

    NtransSigmaInv_[i].Resize(K, D);
    NtransSigmaInv_[i].AddMatSp(1.0, model.N_[i], kTrans, model.GetSigmaInvForSubspace(i), 0.0);
  }

  gamma_s_.Resize(I);
  y_s_.Resize(K);
}

void MleAmMfaSpeakerAccs::Clear() {
  y_s_.SetZero();
  gamma_s_.SetZero();
}


BaseFloat
MleAmMfaSpeakerAccs::Accumulate(const AmMfa &model,
                                const VectorBase<BaseFloat> &data,
                                const AmMfaPerSpkDerivedVars* pVars,
                               int32 state_index,
                               BaseFloat weight, const std::vector<int32>* gselect/* = NULL*/) {
  // Calculate Gaussian posteriors and collect statistics
  Vector<BaseFloat> posteriors;
  BaseFloat log_like = model.LogLikelihood(state_index, data, pVars, &posteriors, gselect);
  posteriors.Scale(weight);
  AccumulateFromPosteriors(model, data, pVars, posteriors, state_index);
  return log_like;
}

#include <stdio.h>

BaseFloat
MleAmMfaSpeakerAccs::AccumulateFromPosteriors(const AmMfa &model,
                                              const VectorBase<BaseFloat> &data,
                                              const AmMfaPerSpkDerivedVars* pVars,
                                             const Vector<BaseFloat> &posteriors,
                                             int32 state_index) {
  int32 spk_space_dim = model.SpkSpaceDim();
  KALDI_ASSERT(spk_space_dim != 0);

  const std::vector<int32>& faIndex = model.sFaIndex_[state_index];
  Matrix<BaseFloat> meanMat;
  model.GetMeans(state_index, &meanMat);
  meanMat.AddVecToRows(-1.0, data);
  if (posteriors.Dim() != meanMat.NumRows())
  {
    std::cout << "Posteriors Dim = " << posteriors.Dim() << ", meanMat Dim = " << meanMat.NumRows() << std::endl;
  }
  meanMat.MulRowsVec(posteriors);
  meanMat.Scale(-1.0);

  for(int32 j = 0; j < faIndex.size(); ++ j)
  {
    int32 i = faIndex[j];
    gamma_s_(i) += posteriors(j);

    y_s_.AddMatVec(1.0, NtransSigmaInv_[i], kNoTrans, meanMat.Row(j), 1.0);
  }


  return posteriors.Sum();
}

void MleAmMfaSpeakerAccs::Update(BaseFloat min_count,
                                Vector<BaseFloat> *v_s,
                                BaseFloat *objf_impr_out,
                                BaseFloat *count_out) {
  double tot_gamma = gamma_s_.Sum();
  KALDI_ASSERT(y_s_.Dim() != 0);
  int32 T = y_s_.Dim();  // speaker-subspace dim.
  int32 num_gauss = gamma_s_.Dim();
  if (v_s->Dim() != T) v_s->Resize(T);  // will set it to zero.

  if (tot_gamma < min_count) {
    KALDI_WARN << "Updating speaker vectors, count is " << tot_gamma
               << " < " << min_count << "not updating.";
    if (objf_impr_out) *objf_impr_out = 0.0;
    if (count_out) *count_out = 0.0;
    return;
  }

  // Eq. (84): H^{(s)} = \sum_{i} \gamma_{i}(s) H_{i}^{spk}
  SpMatrix<BaseFloat> H_s(T);

  for (int32 i = 0; i < num_gauss; i++)
    H_s.AddSp(gamma_s_(i), H_spk_[i]);

  // Don't make these options to SolveQuadraticProblem configurable...
  // they really don't make a difference at all unless the matrix in
  // question is singular, which wouldn't happen in this case.
  Vector<BaseFloat> v_s_dbl(*v_s);
  double tot_objf_impr =
      SolveQuadraticProblem(H_s, y_s_, SolverOptions(), &v_s_dbl);
  v_s->CopyFromVec(v_s_dbl);

  KALDI_LOG << "*Objf impr for speaker vector is " << (tot_objf_impr / tot_gamma)
            << " over " << (tot_gamma) << " frames.";

  if (objf_impr_out) *objf_impr_out = tot_objf_impr;
  if (count_out) *count_out = tot_gamma;
}


}
