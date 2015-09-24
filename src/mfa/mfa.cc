// mfa/mfa.cc

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
#include "mfa/mfa.h"

namespace kaldi {

void MFA::CopyFromMFA(const MFA& mfa)
{
	std::size_t dim = mfa.Dim(), num_comp = mfa.NumComps();
  const std::vector<std::size_t>& k_vec = mfa.LocalDims();
  CovType covType = mfa.CovarianceType();

  if (this->IsPreCompute())
      this->UnPreCompute();
  if (this->IsPreSelGauss())
    this->SetPreSelGauss(false);

  Resize(dim, num_comp, k_vec, covType);
  pi_vec_ = mfa.Weights();
  for(std::size_t i = 0; i < num_comps_; ++ i)
  {
    fa_info* pInfo = fa_info_vec_[i];
    pInfo->W_ = mfa.GetLocalBases(i);
    pInfo->mu_ = mfa.GetLocalCenter(i);
    switch(covType_)
    {
      case DIAG:
        pInfo->sigma_.diagCov_ = mfa.GetLocalVar(i).diagCov_;
        break;
      case FULL:
        pInfo->sigma_.fullCov_ = mfa.GetLocalVar(i).fullCov_;
        break;
      default:
        KALDI_ERR << "Invalid covariance matrix type.";
        break;
    }
  }
}

/// Init MFA by performing PCA on the covariance matrix of FullGmm,
/// the variance is init using the formula of PPCA
void MFA::Init(const FullGmm& fgmm, const InitMFAOptions& opts)
{
  FullGmmNormal ngmm(fgmm);
  std::size_t dim, num_comp;
  dim = fgmm.Dim();
  num_comp = fgmm.NumGauss();
  std::vector<std::size_t> k_vec(num_comp, 1);

  KALDI_LOG << "Init the MFA by FullGmm, phn_space_dim = " << opts.phn_space_dim_
      << ", feature dim = " << dim;

  if (opts.phn_space_dim_ != -1)
  {
    KALDI_ASSERT(opts.phn_space_dim_ <= dim);
  }

  this->Resize(dim, num_comp, k_vec, DIAG);
  pi_vec_.CopyFromVec(ngmm.weights_);

  Vector<double> vecLambda(dim);
  Matrix<double> matP(dim, dim);
  Vector<double> vecSigma(dim);
  for(std::size_t i = 0; i < num_comp; ++ i)
  {
    ngmm.vars_[i].Eig(&vecLambda, &matP);
    SortSvd<double>(&vecLambda, &matP, NULL, true);
    double partialSum = 0.0, sum = vecLambda.Sum();
    for(std::size_t j = 0; j < dim; ++ j)
    {
      partialSum += vecLambda(j);
      if ((opts.phn_space_dim_ > 0 && j == opts.phn_space_dim_ - 1) ||
          (partialSum / sum >= opts.lambda_percentage_) )
      {
        double theta = (sum - partialSum) / (dim - j - 1);
        vecSigma.Set(theta);
        SubVector<double> subVecLambda = vecLambda.Range(0, j + 1);
        subVecLambda.Add(-theta);
        subVecLambda.ApplyPow(0.5);
        SubMatrix<double> subMatP = matP.Range(0, dim, 0, j + 1);
        subMatP.MulColsVec(subVecLambda);
        SetFA(i, j + 1, subMatP, ngmm.means_.Row(i), vecSigma);
        break;
      }
    }
  }
}

/// Convert the covariance matrix to diagonal type
void MFA::ConvertToDiagCov()
{
  if (covType_ == DIAG)
    return ;

  Vector<BaseFloat> var(dim_);
  for(std::size_t i = 0; i < num_comps_; ++ i)
  {
    fa_info* pFaInfo = fa_info_vec_[i];
    var.CopyDiagFromSp(pFaInfo->sigma_.fullCov_);
    pFaInfo->sigma_.diagCov_ = var;
    pFaInfo->sigma_.fullCov_.Resize(0);
  }
  covType_ = DIAG;

  if (this->isPreComputed_ == true)
  {
    UnPreCompute();
    PreCompute();
  }
}

/// Convert the covariance matrix to full type
void MFA::ConvertToFullCov()
{
  if (covType_ == FULL)
      return ;

  SpMatrix<BaseFloat> var(dim_);
  for(std::size_t i = 0; i < num_comps_; ++ i)
  {
    fa_info* pFaInfo = fa_info_vec_[i];
    var.SetZero();
    var.AddDiagVec(1.0, pFaInfo->sigma_.diagCov_);
    pFaInfo->sigma_.fullCov_ = var;
    pFaInfo->sigma_.diagCov_.Resize(0);
  }
  covType_ = FULL;

  if (this->isPreComputed_ == true)
  {
    UnPreCompute();
    PreCompute();
  }
}

/// Convert the model to a full covariance GMM,
//  the FullGmm must be resized before calling this function
void MFA::ConvertToFullGmm(FullGmm* pFGmm, bool cpw/* = true*/)const
{
  FullGmmNormal fullGmmNormal;
  fullGmmNormal.Resize((int)num_comps_, (int)dim_);
  if (cpw == true)
    fullGmmNormal.weights_.CopyFromVec(pi_vec_);
  else
    fullGmmNormal.weights_.Set(1.0 / num_comps_);

  fa_info* pFaInfo = NULL;
  SpMatrix<BaseFloat> cov(dim_);
  for(int32 i = 0; i < num_comps_; ++ i)
  {
    pFaInfo = fa_info_vec_[i];
    fullGmmNormal.means_.Row(i).CopyFromVec(pFaInfo->mu_);
    cov.AddMat2(1.0, pFaInfo->W_, kNoTrans, 0.0);
    if (covType_ == DIAG)
      cov.AddDiagVec(1.0, pFaInfo->sigma_.diagCov_);
    else
      cov.AddSp(1.0, pFaInfo->sigma_.fullCov_);
    fullGmmNormal.vars_[i].CopyFromSp(cov);
  }

  fullGmmNormal.CopyToFullGmm(pFGmm);
  pFGmm->ComputeGconsts();
}

/// Delete components
void MFA::DeleteComponents(const std::vector<int32> indexVec)
{
  if (indexVec.size() == 0)
    return ;

  bool bPreCompute = this->isPreComputed_;
  if (bPreCompute)
    this->UnPreCompute();
  bool bIsPreComputeGauss = this->isPreSelectGauss_;
  MfaGselectConfig gselOpt = gsOpt_;
  if (bIsPreComputeGauss)
    this->SetPreSelGauss(false);

  Vector<BaseFloat> pi_vec(num_comps_ - indexVec.size());
  std::vector<std::size_t> k_vec;
  std::vector<fa_info*> fa_info_vec;

  int32 j = 0;
  for(int32 i = 0; i < num_comps_; ++ i)
  {
    if (std::find(indexVec.begin(), indexVec.end(), i) == indexVec.end())
    {
      pi_vec(j) = pi_vec_(i);
      k_vec.push_back(k_vec_[i]);
      fa_info_vec.push_back(fa_info_vec_[i]);
      ++ j;
    }
    else
    {
      delete fa_info_vec_[i];
      fa_info_vec_[i] = NULL;
    }
  }

  pi_vec_ = pi_vec;
  k_vec_ = k_vec;
  fa_info_vec_ = fa_info_vec;
  num_comps_ = pi_vec_.Dim();

  if (bPreCompute)
      this->PreCompute();
  if (bIsPreComputeGauss)
      this->SetPreSelGauss(true, &gselOpt);

}

/// Set the gselect flag.
/// Note that the MfaGselectConfig is stored in the MFA object,
/// So during decoding, there is no need to pass it to the GaussianSelection call.
/// We can simply set the bPreSelGauss to be true to turn on the gselect option.
/// If we leave this flag to false, and don't give the gselect rspecifier,
/// the gselect option will be turned off.
void MFA::SetPreSelGauss(bool bPreSelGauss/* = true*/,
                         const MfaGselectConfig* pOpt/* =
                          NULL*/) {
  FullGmm fullBgGmm;
  if (bPreSelGauss == true) {
    if (isPreSelectGauss_ == true)
      return;
    fullBgGmm.Resize(this->NumComps(), this->Dim());
    this->ConvertToFullGmm(&fullBgGmm, true);
    diagBgGmm_.Resize(this->NumComps(), this->Dim());
    diagBgGmm_.CopyFromFullGmm(fullBgGmm);
    if (pOpt != NULL) {
      KALDI_ASSERT(pOpt->diag_gmm_nbest > 0);
      gsOpt_ = *pOpt;
    }
    isPreSelectGauss_ = true;
  } else {
    if (isPreSelectGauss_ == false)
      return;
    else {
      diagBgGmm_.Resize(0, 0);
      isPreSelectGauss_ = false;
    }
  }
}

/// Computes the top-scoring Gaussian indices (used for pruning of later
/// stages of computation). Returns frame log-likelihood given selected
/// Gaussians from full UBM.
void MFA::GaussianSelection(const VectorBase<BaseFloat> &data,
                                 std::vector<int32> *gselect) const {
  KALDI_ASSERT(isPreSelectGauss_ == true);
  KALDI_ASSERT(diagBgGmm_.NumGauss() != 0 && diagBgGmm_.Dim() == data.Dim());

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
    gselect->clear();
    for (int32 g = 0; g < num_gauss; g++)
    {
      if (loglikes(g) >= thresh)  // met threshold for diagonal phase.
        gselect->push_back(g);
    }
  }
  else
  {
    for (int32 g = 0; g < num_gauss; g++) {
      gselect->push_back(g);
    }
  }
}

/// Returns the log-likelihood of a data point (vector) given the MFA
BaseFloat MFA::LogLikelihood(const VectorBase<BaseFloat> &data, std::vector<int32> *gselect/* = NULL*/)
{
  if (isPreComputed_ == false)
    PreCompute();
  BaseFloat llk = 0.0;
  for(std::size_t i = 0; i < num_comps_; ++ i)
  {
    if (gselect != NULL && std::find(gselect->begin(), gselect->end(), i) == gselect->end())
      continue;
    llk += log((double)pi_vec_(i));
    llk += LogLikelihood(data, i);
  }
  return llk;
}

// Returns the log-likelihood of a data point (vector) given the mixture index i
BaseFloat MFA::LogLikelihood(const VectorBase<BaseFloat> &data, std::size_t i,
                             VectorBase<BaseFloat>* q1/* = NULL */, VectorBase<BaseFloat>* expected_x1/* = NULL */)
{
  if (isPreComputed_ == false)
      PreCompute();
  std::size_t k = k_vec_[i];
  fa_info* pFaInfo = fa_info_vec_[i];
  pre_compute_data* pData = pre_data_vec_[i];

  // compute q = (y - \mu)^T \Sigma^{-1} W
  Vector<BaseFloat> q(k);
  Vector<BaseFloat> dataS(data);
  dataS.AddVec(-1, pFaInfo->mu_);
  q.AddMatVec(1.0, pData->Sigma1W_, kTrans, dataS, 0.0);
  if (q1 != NULL)
    q1->CopyFromVec(q);

  // Now compute a = (q * M_inv * q^T).  Start by computing M_inv * q^T.
  Vector<BaseFloat> q_temp(k);
  q_temp.AddSpVec(1.0, pData->inv_M_, q, 0.0);
  if (expected_x1 != NULL)
    expected_x1->CopyFromVec(q_temp);
  float a = VecVec(q, q_temp);

  // Now compute b = (y - \mu)^T \Sigma^{-1} (y - \mu)
  Vector<BaseFloat> data_temp(dim_);
  data_temp.AddSpVec(1.0, pData->inv_Sigma_, dataS, 0.0);
  float b = VecVec(dataS, data_temp);

  // compute log_det_Sigma
  float log_det_Sigma = 0.0;
  switch(covType_)
  {
    case DIAG:
      log_det_Sigma = pFaInfo->sigma_.diagCov_.SumLog();
      break;
    case FULL:
      log_det_Sigma = pFaInfo->sigma_.fullCov_.LogDet();
      break;
    default:
      KALDI_ERR << "Invalid covariance type.";
      break;
  }

  // calculate the log determinant of M
  float log_det_M = pData->M_.LogDet();

  // calculate the log likelihood, note that log(det(W W^T + \Sigma) = log_det_Sigma + log_det_M
  float llk = -(b - a) / 2.0 - 0.5 * (dim_ * M_LOG_2PI + log_det_Sigma + log_det_M);

  return llk;
}

/// precompute the data
void MFA::PreCompute()
{
  if (isPreComputed_ == true)
    UnPreCompute();

  isPreComputed_ = true;
  for (std::size_t i = 0; i < num_comps_; ++ i)
  {
    // get the i-th factor model
    fa_info* pFaInfo = fa_info_vec_[i];
    std::size_t k = k_vec_[i];

    // allocate the memory
    pre_compute_data* pData = new pre_compute_data();
    pData->M_.Resize(k);
    pData->inv_M_.Resize(k);
    pData->Sigma1W_.Resize(dim_, k);
    pData->inv_Sigma_.Resize(dim_);

    // calculate Sigma^{-1}
    switch (covType_)
    {
      case DIAG:
        pData->inv_Sigma_.AddDiagVec(1.0, pFaInfo->sigma_.diagCov_);
        break;
      case FULL:
        pData->inv_Sigma_.CopyFromSp(pFaInfo->sigma_.fullCov_);
        break;
      default:
        KALDI_ERR << "Invalid covariance matrix type.";
        break;
    }
    pData->inv_Sigma_.Invert();

    // firstly calculate Sigma1W = \Sigma^{-1} W
    pData->Sigma1W_.AddSpMat(1.0, pData->inv_Sigma_, pFaInfo->W_, kNoTrans, 0.0);

    // secondly calculate M = I + W^T \Sigma^{-1} W
    pData->M_.SetUnit();
    pData->M_.AddMat2Sp(1.0, pFaInfo->W_, kTrans, pData->inv_Sigma_, 1.0);

    // thirdly calculate the inverse of M
    pData->inv_M_ = pData->M_;
    pData->inv_M_.Invert();

    // save the precomputed data
    pre_data_vec_.push_back(pData);
  }
}


/// release the precomputed data
void MFA::UnPreCompute()
{
  if (isPreComputed_ == false)
    return ;
  for(std::size_t i = 0; i < pre_data_vec_.size(); ++ i)
  {
    delete pre_data_vec_[i];
    pre_data_vec_[i] = NULL;
  }
  pre_data_vec_.clear();
  isPreComputed_ = false;
}

CovType CharToCovType(char strCovType)
{
  if (strCovType == 'd')
    return DIAG;
  else if (strCovType == 'f')
    return FULL;
  else
  {
    KALDI_ERR << "Invalid covariance type string: " << strCovType;
    return UNKN;
  }
}

char CovTypeToChar(CovType type)
{
  switch(type)
  {
    case DIAG:
      return 'd';
    case FULL:
      return 'f';
    default:
      KALDI_ERR << "Invalid covariance type.";
      break;
  }
  return '\0';
}

void MFA::Write(std::ostream &out_stream, bool binary) const
{
  WriteToken(out_stream, binary, "<MFA>");
  WriteToken(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteToken(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comps_);
  WriteToken(out_stream, binary, "<WEIGHTS>");
  pi_vec_.Write(out_stream, binary);
  WriteToken(out_stream, binary, "<LOCALDIMENSIONS>");
  WriteIntegerVector(out_stream, binary, k_vec_);
  char cType = CovTypeToChar(covType_);
  WriteToken(out_stream, binary, "<COVTYPE>");
  WriteBasicType(out_stream, binary, cType);

  // write component accumulators
  for (std::size_t i = 0; i < num_comps_; ++ i)
  {
    fa_info* pInfo = fa_info_vec_[i];
    WriteToken(out_stream, binary, "<FAInfo>");

    WriteToken(out_stream, binary, "<W>");
    pInfo->W_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<mu>");
    pInfo->mu_.Write(out_stream, binary);
    switch(covType_)
    {
      case DIAG:
        WriteToken(out_stream, binary, "<DiagCov>");
        pInfo->sigma_.diagCov_.Write(out_stream, binary);
        break;
      case FULL:
        WriteToken(out_stream, binary, "<FullCov>");
        pInfo->sigma_.fullCov_.Write(out_stream, binary);
        break;
      default:
        KALDI_ERR << "Invalid Covariance Type.";
        break;
    }
    WriteToken(out_stream, binary, "</FAInfo>");
  }

  WriteToken(out_stream, binary, "</MFA>");
}

void MFA::Read(std::istream &in_stream, bool binary)
{
  std::size_t dim, num_comp, i;
  std::vector<std::size_t> k_vec;
  Vector<BaseFloat> pi_vec;
  std::string token;
  char cType;
  CovType covType;

  ExpectToken(in_stream, binary, "<MFA>");
  ExpectToken(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dim);
  ExpectToken(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_comp);

  pi_vec.Resize(num_comp);
  k_vec.resize(num_comp);

  ExpectToken(in_stream, binary, "<WEIGHTS>");
  pi_vec.Read(in_stream, binary);
  for(i = 0; i < pi_vec.Dim(); ++ i)
    if (pi_vec(i) < 0 || pi_vec(i) > 1)
      KALDI_ERR << "MFA::Read: the weight of " << i << "th component is invalid (" << pi_vec(i) << ").";

  ExpectToken(in_stream, binary, "<LOCALDIMENSIONS>");
  ReadIntegerVector(in_stream, binary, &k_vec);

  // for compatibility to an oder version of MFA
  ReadToken(in_stream, binary, &token);
  if (token == "<COVTYPE>")
  {
    ReadBasicType(in_stream, binary, &cType);
    covType = CharToCovType(cType);
    ReadToken(in_stream, binary, &token);
  }
  else
    covType = DIAG;

  Resize(dim, num_comp, k_vec, covType);
  pi_vec_ = pi_vec;

  i = 0;
  while (token != "</MFA>") {
    if (token == "<FAInfo>") {

      fa_info* pInfo = fa_info_vec_[i];
      ReadToken(in_stream, binary, &token);
      while(token != "</FAInfo>") {
        if (token == "<W>") {
          pInfo->W_.Read(in_stream, binary);
        } else if (token == "<mu>") {
          pInfo->mu_.Read(in_stream, binary);
        } else if (token == "<DiagCov>") {
          KALDI_ASSERT(covType_ == DIAG);
          pInfo->sigma_.diagCov_.Read(in_stream, binary);
        } else if (token == "<FullCov>")
        {
          KALDI_ASSERT(covType_ == FULL);
          pInfo->sigma_.fullCov_.Read(in_stream, binary);
        } else if (token == "<sigma>") // old model version
        {
          KALDI_ASSERT(covType_ == DIAG);
          pInfo->sigma_.diagCov_.Read(in_stream, binary);
        } else
        {
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        }
        ReadToken(in_stream, binary, &token);
      }
      ++ i;
      ReadToken(in_stream, binary, &token);
    }
  }

   if (i != num_comps_)
    KALDI_WARN << "MFA::Read, find fewer component than expected, " << i << ", vs " << num_comps_ << ".";
}

void MFA::ComputeFeatureNormalizer(Matrix<BaseFloat>* xform)const
{
  int32 dim = this->Dim();
  int32 num_gauss = this->NumComps();
  SpMatrix<BaseFloat> within_class_covar(dim);
  SpMatrix<BaseFloat> between_class_covar(dim);
  Vector<BaseFloat> global_mean(dim);

  // Accumulate LDA statistics from the GMM parameters.
  {
    BaseFloat total_weight = 0.0;
    Vector<BaseFloat> tmp_weight(num_gauss);
    Matrix<BaseFloat> tmp_means;
    std::vector< SpMatrix<BaseFloat> > tmp_covars;
    tmp_weight.CopyFromVec(pi_vec_);
    tmp_covars.resize(num_gauss);
    tmp_means.Resize(num_gauss, dim);
    for(int32 i = 0; i < num_gauss; ++ i)
    {
      tmp_covars[i].Resize(dim);
      if (covType_ == FULL)
        tmp_covars[i].AddSp(1.0, fa_info_vec_[i]->sigma_.fullCov_);
      else
        tmp_covars[i].AddDiagVec(1.0, fa_info_vec_[i]->sigma_.diagCov_);
      tmp_covars[i].AddMat2(1.0, fa_info_vec_[i]->W_, kNoTrans, 1.0);

      tmp_means.Row(i).CopyFromVec(fa_info_vec_[i]->mu_);
    }

    for (int32 i = 0; i < num_gauss; i++) {
      BaseFloat w_i = tmp_weight(i);
      total_weight += w_i;
      within_class_covar.AddSp(w_i, tmp_covars[i]);
      between_class_covar.AddVec2(w_i, tmp_means.Row(i));
      global_mean.AddVec(w_i, tmp_means.Row(i));
    }
    KALDI_ASSERT(total_weight > 0);
    if (fabs(total_weight - 1.0) > 0.001) {
      KALDI_WARN << "Total weight across the GMMs is " << (total_weight)
                 << ", renormalizing.";
      global_mean.Scale(1.0 / total_weight);
      within_class_covar.Scale(1.0 / total_weight);
      between_class_covar.Scale(1.0 / total_weight);
    }
    between_class_covar.AddVec2(-1.0, global_mean);
  }

  TpMatrix<BaseFloat> chol(dim);
  chol.Cholesky(within_class_covar);  // Sigma_W = L L^T
  TpMatrix<BaseFloat> chol_inv(chol);
  chol_inv.InvertDouble();
  Matrix<BaseFloat> chol_full(dim, dim);
  chol_full.CopyFromTp(chol_inv);
  SpMatrix<BaseFloat> LBL(dim);
  // LBL = L^{-1} \Sigma_B L^{-T}
  LBL.AddMat2Sp(1.0, chol_full, kNoTrans, between_class_covar, 0.0);
  Vector<BaseFloat> Dvec(dim);
  Matrix<BaseFloat> U(dim, dim);
  LBL.Eig(&Dvec, &U);
  SortSvd(&Dvec, &U);

  xform->Resize(dim, dim);
  chol_full.CopyFromTp(chol);
  // T := L U, eq (23)
  xform->AddMatMat(1.0, chol_full, kNoTrans, U, kNoTrans, 0.0);
}

}
