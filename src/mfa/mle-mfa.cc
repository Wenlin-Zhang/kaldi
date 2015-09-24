// mfa/mle-mfa.cc

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

#include "mfa/mfa.h"
#include "mfa/mle-mfa.h"

namespace kaldi {

AccumMFA::AccumMFA(): dim_(0), num_comp_(0), count_(0.0), llk_(0.0)
{
}

AccumMFA::AccumMFA(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec)
{
  Resize(dim, num_comp, k_vec);
}

AccumMFA::AccumMFA(const MFA& mfa)
{
  // this class only supports normal MFA training
  KALDI_ASSERT(mfa.CovarianceType() == DIAG);
  Resize(mfa.Dim(), mfa.NumComps(), mfa.LocalDims());
}

AccumMFA::~AccumMFA()
{
  Clear();
}

void AccumMFA::Resize(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec)
{
  if (fa_stat_vec_.size() > 0)
    Clear();

  dim_ = dim;
  num_comp_ = num_comp;
  k_vec_ = k_vec;
  for(std::size_t i = 0; i < num_comp_; ++ i)
  {
    fa_stat* pStat = new fa_stat(dim_, k_vec_[i]);
    fa_stat_vec_.push_back(pStat);
  }
  count_ = 0.0;
  llk_ = 0.0;
}

void AccumMFA::Clear()
{
  for(std::size_t i = 0; i < fa_stat_vec_.size(); ++ i)
  {
    delete fa_stat_vec_[i];
    fa_stat_vec_[i] = NULL;
  }
  fa_stat_vec_.clear();

  dim_ = 0;
  num_comp_ = 0;
  k_vec_.clear();
  count_ = 0;
  llk_ = 0.0;
}

void AccumMFA::Read(std::istream &in_stream, bool binary, bool add)
{
  std::size_t dim, num_comp, i;
  std::vector<std::size_t> k_vec;
  std::string token;
  BaseFloat count, llk;

  ExpectToken(in_stream, binary, "<MFAACCS>");
  ExpectToken(in_stream, binary, "<VECSIZE>");
  ReadBasicType(in_stream, binary, &dim);
  ExpectToken(in_stream, binary, "<NUMCOMPONENTS>");
  ReadBasicType(in_stream, binary, &num_comp);
  k_vec.resize(num_comp);
  ExpectToken(in_stream, binary, "<LOCALDIMENSIONS>");
  ReadIntegerVector(in_stream, binary, &k_vec);

  ExpectToken(in_stream, binary, "<OBSCOUNT>");
  ReadBasicType(in_stream, binary, &count);
  ExpectToken(in_stream, binary, "<LLK>");
  ReadBasicType(in_stream, binary, &llk);

  if (add == false)
  {
    Resize(dim, num_comp, k_vec);
    count_ = count;
    llk_ = llk;
  }
  else
  {
    // match check
    if (num_comp_ != 0 || dim_ != 0)
    {
      if (num_comp != num_comp_ || dim != dim_)
        KALDI_ERR << "AccumMFA::Read, dimension or num_comp mismatch, "
                  << dim_ << ", " << num_comp_ << " vs. " << num_comp << ", "
                  << dim << " (mixing accs from different models?)";

      if (k_vec.size() != num_comp)
        KALDI_ERR << "AccumMFA::Read, num_comp and k_vec conflict, "
                  << num_comp << " vs. " << k_vec.size() << " (corrupt file?)";

      for (std::size_t i = 0; i < k_vec.size(); ++ i)
      {
        if (k_vec[i] != k_vec_[i])
          KALDI_ERR << "AccumMFA::Read, local dimension mismatch, "
                    << i << "th component, " << k_vec_[i]
                    << " vs. " << k_vec[i] << " (mixing accs from different models?)";
      }
    }
    else
    {
      Resize(dim, num_comp, k_vec);
    }

    // add count and llk
    count_ += count;
    llk_ += llk;
  }

  i = 0;
  ReadToken(in_stream, binary, &token);
  while (token != "</MFAACCS>") {
    if (token == "<FAACCS>") {
      fa_stat* pStat = fa_stat_vec_[i];
      ReadToken(in_stream, binary, &token);
      while(token != "</FAACCS>") {
        if (token == "<s0>") {
          BaseFloat s0;
          ReadBasicType(in_stream, binary, &s0);
          if (add)
            pStat->s0_ += s0;
          else
            pStat->s0_ = s0;
        } else if (token == "<W1ACCS>") {
          pStat->W1_.Read(in_stream, binary, add);
        } else if (token == "<W2ACCS>") {
          pStat->W2_.Read(in_stream, binary, add);
        } else if (token == "<sigmaACCS>") {
          pStat->sigma_.Read(in_stream, binary, add);
        } else {
          KALDI_ERR << "Unexpected token '" << token << "' in model file ";
        }
        ReadToken(in_stream, binary, &token);
      }
      ++ i;
      ReadToken(in_stream, binary, &token);
    }
  }

  if (i != num_comp_)
    KALDI_WARN << "AccumMFA::Read, find fewer accumulator than expected, " << i << ", vs " << num_comp_ << ".";
}

void AccumMFA::Write(std::ostream &out_stream, bool binary) const
{
  // write basic information
  WriteToken(out_stream, binary, "<MFAACCS>");
  WriteToken(out_stream, binary, "<VECSIZE>");
  WriteBasicType(out_stream, binary, dim_);
  WriteToken(out_stream, binary, "<NUMCOMPONENTS>");
  WriteBasicType(out_stream, binary, num_comp_);
  WriteToken(out_stream, binary, "<LOCALDIMENSIONS>");
  WriteIntegerVector(out_stream, binary, k_vec_);

  WriteToken(out_stream, binary, "<OBSCOUNT>");
  WriteBasicType(out_stream, binary, count_);
  WriteToken(out_stream, binary, "<LLK>");
  WriteBasicType(out_stream, binary, llk_);

  // write component accumulators
  for (std::size_t i = 0; i < num_comp_; ++ i)
  {
    fa_stat* pStat = fa_stat_vec_[i];
    WriteToken(out_stream, binary, "<FAACCS>");

    WriteToken(out_stream, binary, "<s0>");
    WriteBasicType(out_stream, binary, pStat->s0_);
    WriteToken(out_stream, binary, "<W1ACCS>");
    pStat->W1_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<W2ACCS>");
    pStat->W2_.Write(out_stream, binary);
    WriteToken(out_stream, binary, "<sigmaACCS>");
    pStat->sigma_.Write(out_stream, binary);

    WriteToken(out_stream, binary, "</FAACCS>");
  }

  WriteToken(out_stream, binary, "</MFAACCS>");
}

/// Check the compatibility
bool AccumMFA::Check(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec)
{
  if (num_comp != num_comp_ || dim != dim_)
     return false;

  if (k_vec.size() != num_comp)
  {
    KALDI_ERR << "in AccumMFA::Check : k_vec.size() != num_comp.";
    return false;
  }

  for (std::size_t i = 0; i < k_vec.size(); ++ i)
  {
    if (k_vec[i] != k_vec_[i])
      return false;
  }

  return true;
}

/// Accumulate one observation
BaseFloat AccumMFA::AccumulateForObservation(MFA& mfa, const VectorBase<BaseFloat> &data, BaseFloat weight/* = 1.0 */)
{
  // this class only supports normal MFA training
  KALDI_ASSERT(mfa.CovarianceType() == DIAG);

  // precompute
  if (mfa.IsPreCompute() == false)
    mfa.PreCompute();

  // calculate the component posterior, q and expected x.
  Vector<double> posteriorVec(num_comp_);
  double llk = 0.0;
  std::vector<Vector<BaseFloat>* > q_vec;
  std::vector<Vector<BaseFloat>* > expected_x_vec;
  // Doing Gaussian pre-selection
  std::vector<int32> gselect;
  if (mfa.IsPreSelGauss() == true)
  {
    mfa.GaussianSelection(data, &gselect);
  }
  for(std::size_t i = 0; i < num_comp_; ++ i)
  {
    if (mfa.IsPreSelGauss() == true && std::find(gselect.begin(), gselect.end(), i) == gselect.end())
    {
      q_vec.push_back(NULL);
      expected_x_vec.push_back(NULL);
      continue;
    }
    std::size_t k = k_vec_[i];
    Vector<BaseFloat>* p_q = new Vector<BaseFloat>(k);
    Vector<BaseFloat>* p_expected_x = new Vector<BaseFloat>(k);
    posteriorVec(i) = log((double)mfa.pi_vec_(i)) + mfa.LogLikelihood(data, i, p_q, p_expected_x);
    if (llk == 0.0)
      llk = posteriorVec(i);
    else
      llk = LogAdd(llk, posteriorVec(i));
    q_vec.push_back(p_q);
    expected_x_vec.push_back(p_expected_x);
  }

  // accumulate for each component
//#define TEST1 1
#ifdef TEST1
  double post_temp = 0.0;
#endif
  for(std::size_t i = 0; i < num_comp_; ++ i)
  {
    if (mfa.IsPreSelGauss() == true && std::find(gselect.begin(), gselect.end(), i) == gselect.end())
      continue;

    fa_stat* pStat = fa_stat_vec_[i];

    // calculate the real posterior
    double posterior = weight * exp(posteriorVec(i) - llk);

    // accumulate the posteriors
    pStat->s0_ += posterior;

    // calculate the sufficient statistics
    std::size_t k = k_vec_[i];
    Vector<BaseFloat>& expected_x = *(expected_x_vec[i]);
    SpMatrix<BaseFloat> expected_xx(k);  // <xx^T> = M_inv + <x><x>^T
    expected_xx = mfa.pre_data_vec_[i]->inv_M_;
    expected_xx.AddVec2(1.0, expected_x);

    // accumulate for W1
    SubMatrix<BaseFloat> W1_yx(pStat->W1_, 0, dim_, 0, k);
    W1_yx.AddVecVec(posterior, data, expected_x);
    for (std::size_t j = 0; j < dim_; ++ j)
      pStat->W1_(j, k) += posterior * data(j);

    // accumulate for W2
    Matrix<BaseFloat> W2_temp(pStat->W2_);
    SubMatrix<BaseFloat> W2_xx(W2_temp, 0, k, 0, k);
    W2_xx.AddSp(posterior, expected_xx);
    for (std::size_t j = 0; j < k; ++ j)
    {
      float temp = posterior * expected_x(j);
      W2_temp(k, j) += temp;
    }
    W2_temp(k, k) += posterior;
#ifdef TEST1
    post_temp += posterior;
#endif
    pStat->W2_.CopyFromMat(W2_temp, kTakeLower);

    // accumulate for sigma
    pStat->sigma_.AddVec2(posterior, data);

  }
#ifdef TEST1
  KALDI_LOG << "Sum of posterior: " << post_temp << "(weight: " << weight << ".)";
#endif
  // clear the memory
  for(std::size_t i = 0; i < num_comp_; ++ i)
  {
    if (q_vec[i] != NULL)
    {
      delete q_vec[i];
      q_vec[i] = NULL;
    }
    if (expected_x_vec[i] != NULL)
    {
      delete expected_x_vec[i];
      expected_x_vec[i] = NULL;
    }
  }
  q_vec.clear();
  expected_x_vec.clear();

  //  accumulate the observation count and llk
  count_ += weight;
  llk_ += llk;

  // return the current log likelihood
  return llk;
}

/// for computing the maximum-likelihood estimates of the parameters of
/// a MoFA model.
void AccumMFA::MleMFAUpdate(MFA *pMFA, bool deleteLowOccComp/* = false */, BaseFloat minOcc/* = 10*/)
{
  // this class only supports normal MFA training
  KALDI_ASSERT(pMFA->CovarianceType() == DIAG);

  // check dimension
  if (Check(pMFA->dim_, pMFA->num_comps_, pMFA->k_vec_) == false)
    KALDI_ERR << "MleMFAUpdate: the model and the accumulator is mismatch in dimension.";
#define TEST2
#ifdef TEST2
  double k_total = 0.0, pi_total = 0.0;
#endif
  // update each component
  std::vector<int32> lowOccCompVec;
  for(int32 i = 0; i < num_comp_; ++ i)
  {
    MFA::fa_info* pInfo = pMFA->fa_info_vec_[i];
    fa_stat* pStat = fa_stat_vec_[i];
    std::size_t k = k_vec_[i];

    // detect low occupation component and record it
    if (pStat->s0_ < minOcc)
    {
      lowOccCompVec.push_back(i);
      continue;
    }

    // compute \f$[W; mu] = W1 * W2^{-1}\f$
    SpMatrix<BaseFloat> invW2 = pStat->W2_;
    //KALDI_LOG << "W2 before invert : \n" << invW2;
    invW2.Invert();
    //KALDI_LOG << "W2 after invert : \n" << invW2;
    Matrix<BaseFloat> wmu(dim_, k + 1);
    wmu.AddMatSp(1.0, pStat->W1_, kNoTrans, invW2, 0.0);
    pInfo->W_ = wmu.Range(0, dim_, 0, k);
    for(std::size_t j = 0; j < dim_; ++ j)
      pInfo->mu_(j) = wmu(j, k);

    // compute pi_i = h1 / N, in fact h1 = W2(k, k)
    pMFA->pi_vec_(i) = pStat->W2_(k, k) / (float)count_;

#ifdef TEST2
    k_total += pStat->W2_(k, k);
    pi_total += pMFA->pi_vec_(i);
#endif

    /**
     * Recomputes the #sigma paramter
     *
     * \f{eqnarray*}
     * \widetilde{\Sigma} &=&
     * \frac{1}{N}\sum_j^k \textrm{diag}\left\{ \sum_i^N h_{ij} \left\langle yy^\top\right\rangle -
     * \sum_i^N h_{ij} \left\langle yx^\top\right\rangle \widetilde{W}^\top -
     * \sum_i^N h_{ij} \left\langle y\right\rangle \widetilde{\mu}^\top
     * \right\}\\
     * &=& \frac{1}{N} \sum_j^k \left\{ \Sigma_1 -
     * \textrm{diag}
     * \left[
     * W_1\left(\begin{array}{c}\widetilde{W}^\top \\ \widetilde{\mu}^\top\end{array}\right)\right]
     * \right\}\\
     * \f}
     */
    wmu.MulElements(pStat->W1_);
    pInfo->sigma_.diagCov_ = pStat->sigma_;
    pInfo->sigma_.diagCov_.AddColSumMat(-1.0, wmu);
    pInfo->sigma_.diagCov_.Scale(1.0 / (double)pStat->W2_(k, k));
  }

  if (lowOccCompVec.size() != 0)
  {
    KALDI_LOG << "Low occupation components detect, count = " << lowOccCompVec.size();
    if (deleteLowOccComp == true)
    {
      int32 num_comp1 = pMFA->NumComps();
      pMFA->DeleteComponents(lowOccCompVec);
      int32 num_comp2 = pMFA->NumComps();
      KALDI_LOG << "Remove low occupation components, component count change from "
          << num_comp1 << " to " << num_comp2;
    }
  }

#ifdef TEST2
  KALDI_LOG << "total count = " << k_total << " ( real count = " << count_ << ").";
  KALDI_LOG << "total pi = " << pi_total << " ( real total pi = 1.0).";
#endif
}

}
