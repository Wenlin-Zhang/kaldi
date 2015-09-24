// mfa/estimate-am-mfa-ebw.cc

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

#include "base/kaldi-common.h"
#include "mfa/estimate-am-mfa-ebw.h"
#include "thread/kaldi-thread.h"
#include <algorithm>  // for std::max

namespace kaldi {

void EbwAmMfaUpdater::Update(const MleAmMfaAccs &num_accs,
                             const MleAmMfaAccs &den_accs,
                             AmMfa *model,
                             AmMfaUpdateFlagsType flags,
                             BaseFloat *auxf_change_out,
                             BaseFloat *count_out) {

  KALDI_ASSERT((flags & (kAmMfaPhoneVectors | kAmMfaPhoneWeights | kAmMfaPhoneProjections |
                         kAmMfaCovarianceMatrix | kAmMfaFAMeans | kAmMfaSpeakerProjections)
                         ) != 0);

  num_accs.Check(*model, false);
  den_accs.Check(*model, false);
  model->mfa_.PreCompute();
  model->PreCompute();
  
  BaseFloat tot_impr = 0.0;

  /// update phone vectors
  if (flags & kAmMfaPhoneVectors)
     tot_impr += Update_y(num_accs, den_accs, model);

   ///  update phone weights
   if (flags & kAmMfaPhoneWeights)
      tot_impr += Update_w(num_accs, den_accs, model);

  // Calculate Y_i
  std::vector<Matrix<BaseFloat> > Y_i_vec_num, Y_i_vec_den;
  if (flags & (kAmMfaPhoneProjections | kAmMfaCovarianceMatrix))
  {
  	MleAmMfaUpdater::Compute_Y_i_s(num_accs, *model, &Y_i_vec_num);
  	MleAmMfaUpdater::Compute_Y_i_s(den_accs, *model, &Y_i_vec_den);
  }
  
  // Calculate Q_i
  std::vector<SpMatrix<BaseFloat> > Q_i_vec_num, Q_i_vec_den;
  if (flags & kAmMfaPhoneProjections) {
    MleAmMfaUpdater::Compute_Q_i_s(num_accs, *model, &Q_i_vec_num);
    MleAmMfaUpdater::Compute_Q_i_s(den_accs, *model, &Q_i_vec_den);
  }
  
  // Calculate s0, s1
  Vector<BaseFloat> s0_i_num, s0_i_den;
  Matrix<BaseFloat> s1_i_num, s1_i_den;
  if (flags & (kAmMfaPhoneProjections | kAmMfaCovarianceMatrix | kAmMfaFAMeans | kAmMfaSpeakerProjections))
  {
    MleAmMfaUpdater::Compute_s0_i_s1_i_s(num_accs, *model, &s0_i_num, &s1_i_num);
    MleAmMfaUpdater::Compute_s0_i_s1_i_s(den_accs, *model, &s0_i_den, &s1_i_den);
  }
  
  /// update phone projections
  if (flags & kAmMfaPhoneProjections)
    tot_impr += Update_M(num_accs, den_accs, s0_i_num, s0_i_den, Q_i_vec_num, Q_i_vec_den, Y_i_vec_num, Y_i_vec_den, model);

  /// update phone covariance
  if (flags & kAmMfaCovarianceMatrix) {
    // Calculate S_i, S_mean_i
    std::vector<SpMatrix<BaseFloat> > S_i_vec_num, S_i_vec_den;
    std::vector<SpMatrix<BaseFloat> > S_means_i_vec_num, S_means_i_vec_den;
    MleAmMfaUpdater::Compute_S_i_S_means_i_s(num_accs, *model, s0_i_num,
                                             s1_i_num, &S_i_vec_num,
                                             &S_means_i_vec_num);
    MleAmMfaUpdater::Compute_S_i_S_means_i_s(den_accs, *model, s0_i_den,
                                             s1_i_den, &S_i_vec_den,
                                             &S_means_i_vec_den);
    int32 dim = model->FeatureDim();
    Matrix<BaseFloat> Sigmma_i(dim, dim);
    for (int i = 0; i < S_i_vec_num.size(); ++i) {
      S_i_vec_num[i].AddSp(1.0, S_means_i_vec_num[i]);
      S_i_vec_num[i].AddSp(-1.0, S_i_vec_den[i]);
      S_i_vec_num[i].AddSp(-1.0, S_means_i_vec_den[i]);

      Matrix<BaseFloat>& Mi = model->mfa_.fa_info_vec_[i]->W_;
      Sigmma_i.AddMatMat(-1.0, Mi, kNoTrans, Y_i_vec_num[i], kTrans, 0.0);
      Sigmma_i.AddMatMat(-1.0, Y_i_vec_num[i], kNoTrans, Mi, kTrans, 1.0);
      Sigmma_i.AddMatMat(1.0, Mi, kNoTrans, Y_i_vec_den[i], kTrans, 1.0);
      Sigmma_i.AddMatMat(1.0, Y_i_vec_den[i], kNoTrans, Mi, kTrans, 1.0);
      Sigmma_i.AddSp(1.0, S_i_vec_num[i]);

      S_i_vec_num[i].CopyFromMat(Sigmma_i);
    }
    tot_impr += Update_Sigma(num_accs, den_accs, s0_i_num, s0_i_den,
                             S_i_vec_num, model);
  }

  /// update means
  if (flags & kAmMfaFAMeans)
  {
    Matrix<BaseFloat> s1_means_i_num, s1_means_i_den;
    MleAmMfaUpdater::Compute_s1_means_i_s(num_accs, *model, &s1_means_i_num);
    MleAmMfaUpdater::Compute_s1_means_i_s(den_accs, *model, &s1_means_i_den);
    s1_i_num.AddMat(-1.0, s1_i_den);
    s1_i_num.AddMat(-1.0, s1_means_i_num);
    s1_i_num.AddMat(1.0, s1_means_i_den);

    tot_impr += Update_Mu(s0_i_num, s0_i_den, s1_i_num, model);
  }

  /// update speaker projections
  if (flags & kAmMfaSpeakerProjections)
  {
    tot_impr += Update_N(num_accs, den_accs, s0_i_num, s0_i_den, model);
  }

  if (auxf_change_out) *auxf_change_out = tot_impr * num_accs.total_frames_;
  if (count_out) *count_out = num_accs.total_frames_;
  
  if (fabs(num_accs.total_frames_ - den_accs.total_frames_) >
      0.01*(num_accs.total_frames_ + den_accs.total_frames_))
    KALDI_WARN << "Num and den frame counts differ, "
               << num_accs.total_frames_ << " vs. " << den_accs.total_frames_;

  BaseFloat like_diff = num_accs.total_like_ - den_accs.total_like_;
  
  KALDI_LOG << "***Averaged differenced likelihood per frame is "
            << (like_diff/num_accs.total_frames_)
            << " over " << (num_accs.total_frames_) << " frames.";
  KALDI_LOG << "***Note: for this to be at all meaningful, if you use "
            << "\"canceled\" stats you will have to renormalize this over "
            << "the \"real\" frame count.";
}

BaseFloat EbwAmMfaUpdater::Update_y(const MleAmMfaAccs &num_accs,
                                 const MleAmMfaAccs &den_accs,
                                 AmMfa *model) const {
                                         
  KALDI_LOG << "Updating phone vectors.";
  
  const MFA& mfa = model->GetMFA();  
  BaseFloat count = 0.0, auxf_impr = 0.0;

  // Check the AmMfa model is precomputed
  KALDI_ASSERT(model->isPreCompute_ == true);

  for(int j = 0; j < model->NumStates(); ++ j)
  {
    const std::vector<int32>& faIndex = model->sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      
      Vector<BaseFloat>& y = model->sFaLocation_[j][k];
      
      BaseFloat gamma_ji_num = num_accs.s0_[j](k);
      BaseFloat gamma_ji_den = den_accs.s0_[j](k);

      if (gamma_ji_num + gamma_ji_den < options_.s0_thresh_) {
            KALDI_WARN << "Not updating phone vector for j = " << j << ", k = " << k
                       << ", because total count is "    << gamma_ji_num + gamma_ji_den << " < "   << options_.s0_thresh_;
            continue;
          }

      Vector<BaseFloat> s_ji_num(num_accs.s1_[j].Row(k));
      s_ji_num.AddVec(-gamma_ji_num, mfa.fa_info_vec_[i]->mu_);
      Vector<BaseFloat> s_ji_den(den_accs.s1_[j].Row(k));
      s_ji_den.AddVec(-gamma_ji_den, mfa.fa_info_vec_[i]->mu_);

      Vector<BaseFloat> g_num(mfa.k_vec_[i]), g_den(mfa.k_vec_[i]);
      g_num.AddMatVec(1.0, mfa.pre_data_vec_[i]->Sigma1W_, kTrans, s_ji_num, 0.0);
      g_den.AddMatVec(1.0, mfa.pre_data_vec_[i]->Sigma1W_, kTrans, s_ji_den, 0.0);

      SpMatrix<BaseFloat> H_num(mfa.pre_data_vec_[i]->M_), H_den(mfa.pre_data_vec_[i]->M_);
      H_num.Scale(gamma_ji_num);
      H_den.Scale(gamma_ji_den);
      
      // calculate modified g
      g_num.AddSpVec(-1.0, H_num, y, 1.0);
      g_den.AddSpVec(-1.0, H_den, y, 1.0);
      
      // diff the numerator and denominator
      g_num.AddVec(-1.0, g_den);
      H_num.AddSp(1.0, H_den);
      
      // calculate the learning factor
      BaseFloat gamma_ji = gamma_ji_num + gamma_ji_den;
      BaseFloat factor = (gamma_ji + options_.tau_y) / (gamma_ji * options_.lrate_y);
      H_num.Scale(factor);
      
      // calculate the dy
      Vector<BaseFloat> dy(y.Dim());
      auxf_impr += SolveQuadraticProblem(H_num, g_num, SolverOptions(), &dy);
      count += gamma_ji_num;
      
      // update the phone vector
      y.AddVec(1.0, dy);
    }
  }
  
  auxf_impr /= count;

  KALDI_LOG << "**Overall auxf improvement for phone vector (y) is " << auxf_impr
            << " over " << count << " frames";
  return auxf_impr;
}


BaseFloat EbwAmMfaUpdater::Update_M(const MleAmMfaAccs &num_accs,
                                 const MleAmMfaAccs &den_accs,
                                 const Vector<BaseFloat> &s0_i_num,
                                 const Vector<BaseFloat> &s0_i_den,
                                 const std::vector< SpMatrix<BaseFloat> > &Q_i_vec_num,
                                 const std::vector< SpMatrix<BaseFloat> > &Q_i_vec_den,
                                 const std::vector<Matrix<BaseFloat> > &Y_i_vec_num,
                                 const std::vector<Matrix<BaseFloat> > &Y_i_vec_den,
                                 AmMfa *model) const {
  int I = num_accs.NumFactors();
  int D = num_accs.FeatureDim();
  MFA* pMfa = &(model->mfa_);

  // Check the AmMfa model is precomputed
  KALDI_ASSERT(model->isPreCompute_ == true);

  Vector<BaseFloat> impr_vec(I);
  for (int32 i = 0; i < I; i++) {
    double gamma_i_num = s0_i_num(i), gamma_i_den = s0_i_den(i);
    if (gamma_i_num + gamma_i_den < options_.s0_thresh_) {
      KALDI_WARN << "Not updating phonetic basis for i = " << i
                 << " because total count is "    << gamma_i_num + gamma_i_den << " < "   << options_.s0_thresh_;
      continue;
    }
    
    int32 lD = pMfa->GetLocalDim(i);
    Matrix<BaseFloat> Mi(pMfa->fa_info_vec_[i]->W_);
    Matrix<BaseFloat> Y(D, lD);
    Y.AddMat(1.0, Y_i_vec_num[i]);
    Y.AddMatSp(-1.0, Mi, kNoTrans, Q_i_vec_num[i], 1.0);
    Y.AddMat(-1.0, Y_i_vec_den[i]);
    Y.AddMatSp(-1.0*-1.0, Mi, kNoTrans, Q_i_vec_den[i], 1.0);

    SpMatrix<BaseFloat> Q(lD); // This is a combination of the Q's for the numerator and denominator.
    Q.AddSp(1.0, Q_i_vec_num[i]);
    Q.AddSp(1.0, Q_i_vec_den[i]);

    BaseFloat state_count = 1.0e-10 + gamma_i_num + gamma_i_den; // the count
    // represented by the quadratic part of the stats.
    Q.Scale( (state_count + options_.tau_M) / state_count );
    Q.Scale( 1.0 / (options_.lrate_M + 1.0e-10) );
    
    Matrix<BaseFloat> deltaM(D, lD);
    SolverOptions solverOpt("M");
    solverOpt.eps = options_.epsilon;
    solverOpt.K = options_.max_cond;
    BaseFloat impr = SolveQuadraticMatrixProblem(Q, Y,
                                    model->invSigma_[i], solverOpt,
                                    &deltaM);

    impr_vec(i) = impr;
    Mi.AddMat(1.0, deltaM);
    pMfa->fa_info_vec_[i]->W_.CopyFromMat(Mi);
    if (i < 10 || impr / state_count > 3.0) {
      KALDI_LOG << "Objf impr for projection M for i = " << i << ", is "
                << (impr/(gamma_i_num + 1.0e-20)) << " over " << gamma_i_num
                << " frames";
    }
  }
  BaseFloat tot_count = s0_i_num.Sum(), tot_impr = impr_vec.Sum();
  
  tot_impr /= (tot_count + 1.0e-20);
  KALDI_LOG << "Overall auxiliary function improvement for model projections "
            << "M is " << tot_impr << " over " << tot_count << " frames";

  KALDI_VLOG(1) << "Updating M: num-count is " << s0_i_num;
  KALDI_VLOG(1) << "Updating M: den-count is " << s0_i_den;
  KALDI_VLOG(1) << "Updating M: objf-impr is " << impr_vec;
  
  return tot_impr;
}

BaseFloat EbwAmMfaUpdater::Update_N(const MleAmMfaAccs &num_accs,
                                    const MleAmMfaAccs &den_accs,
                                    const Vector<BaseFloat> &s0_i_num,
                                    const Vector<BaseFloat> &s0_i_den,
                                    AmMfa *model) const {
  int I = num_accs.NumFactors();
  int D = num_accs.FeatureDim();
  int K = num_accs.SpkrDim();

  // Check the AmMfa model is precomputed
  KALDI_ASSERT(model->isPreCompute_ == true);

  Vector<BaseFloat> impr_vec(I);
  for (int32 i = 0; i < I; i++) {
    double gamma_i_num = s0_i_num(i), gamma_i_den = s0_i_den(i);
    if (gamma_i_num + gamma_i_den < options_.s0_thresh_) {
      KALDI_WARN << "Not updating speaker basis for i = " << i
                 << "  because total count is "    << gamma_i_num + gamma_i_den << " < "   << options_.s0_thresh_;
      continue;
    }

    Matrix<BaseFloat> Y(D, K);
    Y.AddMat(1.0, num_accs.Z_vec_[i]);
    Y.AddMatSp(-1.0, model->N_[i], kNoTrans, num_accs.R_vec_[i], 1.0);
    Y.AddMat(-1.0, den_accs.Z_vec_[i]);
    Y.AddMatSp(-1.0 * -1.0, model->N_[i], kNoTrans, den_accs.R_vec_[i], 1.0);

    SpMatrix<BaseFloat> Q(K);  // This is a combination of the Q's for the numerator and denominator.
    Q.AddSp(1.0, num_accs.R_vec_[i]);
    Q.AddSp(1.0, den_accs.R_vec_[i]);

    BaseFloat state_count = 1.0e-10 + gamma_i_num + gamma_i_den;  // the count
    // represented by the quadratic part of the stats.
    Q.Scale((state_count + options_.tau_N) / state_count);
    Q.Scale(1.0 / (options_.lrate_N + 1.0e-10));

    Matrix<BaseFloat> deltaM(D, K);
    SolverOptions solverOpt("N");
    solverOpt.eps = options_.epsilon;
    solverOpt.K = options_.max_cond;
    BaseFloat impr = SolveQuadraticMatrixProblem(Q, Y, model->invSigma_[i],
                                                 solverOpt, &deltaM);

    impr_vec(i) = impr;
    model->N_[i].AddMat(1.0, deltaM);
    if (i < 10 || impr / state_count > 3.0) {
      KALDI_LOG << "Objf impr for projection M for i = " << i << ", is "
                << (impr / (gamma_i_num + 1.0e-20)) << " over " << gamma_i_num
                << " frames";
    }
  }
  BaseFloat tot_count = s0_i_num.Sum(), tot_impr = impr_vec.Sum();

  tot_impr /= (tot_count + 1.0e-20);
  KALDI_LOG << "Overall auxiliary function improvement for speaker projections "
            << "M is " << tot_impr << " over " << tot_count << " frames";

  KALDI_VLOG(1) << "Updating M: num-count is " << s0_i_num;
  KALDI_VLOG(1) << "Updating M: den-count is " << s0_i_den;
  KALDI_VLOG(1) << "Updating M: objf-impr is " << impr_vec;

  return tot_impr;

}


BaseFloat EbwAmMfaUpdater::Update_Mu(const Vector<BaseFloat> &gamma_num,
                                     const Vector<BaseFloat> &gamma_den,
                                     const Matrix<BaseFloat> & s1_mean,
                                     AmMfa *model)const
{
  int I = s1_mean.NumRows();
  MFA* pMfa = &(model->mfa_);
  // Check the AmMfa model is precomputed
  KALDI_ASSERT(model->isPreCompute_ == true);

  Vector<BaseFloat> impr_vec(I);
  for (int32 i = 0; i < I; i++) {
    BaseFloat num_count = gamma_num(i), den_count = gamma_den(i);
    if (num_count + den_count < options_.s0_thresh_) {
      KALDI_WARN << "Not updating subspace mean vector for i = " << i
                 << "  because total count is "  << num_count + den_count << " < "   << options_.s0_thresh_;
      continue;
    }

    KALDI_ASSERT(options_.lrate_mu <= 1.0);
    BaseFloat inv_lrate = 1.0 / options_.lrate_mu;
    BaseFloat E_den = 1.0 + inv_lrate, E_num = inv_lrate - 1.0;
    BaseFloat smoothing_count =
            (options_.tau_mu * inv_lrate) + // multiply tau_Sigma by inverse-lrate
            (E_den * den_count) +           // for compatibility with other updates.
            (E_num * num_count) +
            1.0e-10;

    Vector<BaseFloat> old_mu(model->mfa_.fa_info_vec_[i]->mu_);
    Vector<BaseFloat> mu_stat(s1_mean.Row(i));
    mu_stat.AddVec(smoothing_count, old_mu);
    BaseFloat count = num_count - den_count + smoothing_count;
    mu_stat.Scale(1.0 / count);

    model->mfa_.fa_info_vec_[i]->mu_.CopyFromVec(mu_stat);

    // calculate the impr
    const SpMatrix<BaseFloat>& invSigma = pMfa->pre_data_vec_[i]->inv_Sigma_;
    impr_vec(i) = 0.5 * count * VecSpVec(mu_stat, invSigma, mu_stat)
                  + 0.5 * count * VecSpVec(old_mu, invSigma, old_mu)
                  - count * VecSpVec(old_mu, invSigma, mu_stat);
  }

  KALDI_VLOG(1) << "Updating mu: numerator count is " << gamma_num;
  KALDI_VLOG(1) << "Updating mu: denominator count is " << gamma_den;
  KALDI_VLOG(1) << "Updating mu: objf-impr is " << impr_vec;
  
  double tot_count = gamma_num.Sum(), tot_impr = impr_vec.Sum();
  tot_impr /= (tot_count + 1.0e-20);
  KALDI_LOG << "**Overall auxf impr for N is " << tot_impr
            << " over " << tot_count << " frames";
  return tot_impr;
}

BaseFloat EbwAmMfaUpdater::Update_Sigma(const MleAmMfaAccs &num_accs,
                                     const MleAmMfaAccs &den_accs,
                                     const Vector<BaseFloat> &gamma_num,
                                     const Vector<BaseFloat> &gamma_den,
                                     const std::vector< SpMatrix<BaseFloat> > &S_means,
                                     AmMfa *model) const {
  int32 I = num_accs.NumFactors();
  KALDI_ASSERT(S_means.size() == I);
  Vector<BaseFloat> impr_vec(I);
  
  MFA* pMfa = &(model->mfa_);
  KALDI_ASSERT(pMfa->IsPreCompute());

  for (int32 i = 0; i < I; i++) {
    double num_count = gamma_num(i), den_count = gamma_den(i);

    // if s0 is very small, skip the updation
    if (num_count + den_count < options_.s0_thresh_)
    {
      KALDI_LOG << "The " << i << "th factor model (Sigma) is not updated because total count is "
    	                << num_count + den_count << " < "   << options_.s0_thresh_;
      continue;
    }

    // SigmaStats now contain the stats for estimating Sigma (as in the main SGMM paper),
    // differenced between num and den.
    SpMatrix<BaseFloat> SigmaInvOld(pMfa->pre_data_vec_[i]->inv_Sigma_);
    SpMatrix<BaseFloat> SigmaOld(SigmaInvOld);
    SigmaOld.Invert();

    BaseFloat count = num_count - den_count;
    KALDI_ASSERT(options_.lrate_Sigma <= 1.0);
    BaseFloat inv_lrate = 1.0 / options_.lrate_Sigma;
    // These formulas assure that the objective function behaves in
    // a roughly symmetric way w.r.t. num and den counts.
    double E_den = 1.0 + inv_lrate, E_num = inv_lrate - 1.0;
    BaseFloat smoothing_count =
        (options_.tau_Sigma * inv_lrate) + // multiply tau_Sigma by inverse-lrate
        (E_den * den_count) +              // for compatibility with other updates.
        (E_num * num_count) +
        1.0e-10;
    SpMatrix<BaseFloat> SigmaStats(S_means[i]);
    SigmaStats.AddSp(smoothing_count, SigmaOld);
    count += smoothing_count;
    SigmaStats.Scale(1.0 / count);
    SpMatrix<BaseFloat> SigmaInv(SigmaStats); // before floor and ceiling.  Currently sigma, not its inverse.
    bool verbose = false;
    //bool is_psd = false; // we cannot guarantee that Sigma Inv is positive semidefinite.
    int n_floor = SigmaInv.ApplyFloor(SigmaOld, options_.cov_min_value, verbose);
    SigmaInv.Invert(); // make it inverse variance.
    int n_ceiling = SigmaInv.ApplyFloor(SigmaInvOld, options_.cov_min_value, verbose);

    // this auxf_change.  
    double auxf_change = -0.5 * count *(TraceSpSp(SigmaInv, SigmaStats)
                                        - TraceSpSp(SigmaInvOld, SigmaStats)
                                        - SigmaInv.LogDet()
                                        + SigmaInvOld.LogDet());

    SigmaInv.Invert();
    // copy the diagonal elements to MFA model
    if (pMfa->covType_ == DIAG)
      pMfa->fa_info_vec_[i]->sigma_.diagCov_.CopyDiagFromSp(SigmaInv);
    else
      pMfa->fa_info_vec_[i]->sigma_.fullCov_.CopyFromSp(SigmaInv);

    impr_vec(i) = auxf_change;
    if (i < 10 || auxf_change / (num_count+den_count+1.0e-10) > 2.0
        || n_floor+n_ceiling > 0) {
      KALDI_LOG << "Updating variance: Auxf change per frame for Gaussian "
                << i << " is " << (auxf_change / num_count) << " over "
                << num_count << " frames " << "(den count was " << den_count
                << "), #floor,ceil was " << n_floor << ", " << n_ceiling;
    }
  }
  KALDI_VLOG(1) << "Updating Sigma: numerator count is " << gamma_num;
  KALDI_VLOG(1) << "Updating Sigma: denominator count is " << gamma_den;
  KALDI_VLOG(1) << "Updating Sigma: objf-impr is " << impr_vec;
  
  double tot_count = gamma_num.Sum(), tot_impr = impr_vec.Sum();
  tot_impr /= tot_count+1.0e-20;
  KALDI_LOG << "**Overall auxf impr for Sigma is " << tot_impr
            << " over " << tot_count << " frames";
  return tot_impr;
}

BaseFloat Update_w_for_GMM(Vector<BaseFloat>& weights_,
                           const Vector<BaseFloat>& num_occs_,
                           const Vector<BaseFloat>& den_occs_,
                           BaseFloat tau, BaseFloat lrate)
{
  KALDI_ASSERT(weights_.Dim() == num_occs_.Dim() && num_occs_.Dim() == den_occs_.Dim());
  if (weights_.Dim() == 1) return 0.0; // Nothing to do: only one mixture.

  Vector<BaseFloat> weights(weights_), num_occs(num_occs_), den_occs(den_occs_);
  num_occs.AddVec(tau, weights);

  BaseFloat impr = 0.0;
  double weight_auxf_at_start = 0.0, weight_auxf_at_end = 0.0;
    int32 num_comp = weights.Dim();
    for (int32 g = 0; g < num_comp; g++) {   // c.f. eq. 4.32 in Dan Povey's thesis.
      weight_auxf_at_start +=
          num_occs(g) * log (weights(g))
          - den_occs(g) * weights(g) / weights_(g);
    }
    for (int32 iter = 0; iter < 50; iter++) {
      Vector<BaseFloat> k_jm(num_comp); // c.f. eq. 3.35
      BaseFloat max_m = 0.0;
      for (int32 g = 0; g < num_comp; g++)
        max_m = std::max((BaseFloat)max_m, (BaseFloat)den_occs(g)/weights_(g));
      for (int32 g = 0; g < num_comp; g++)
        k_jm(g) = max_m - den_occs(g)/weights_(g);
      for (int32 g = 0; g < num_comp; g++) // c.f. eq. 3.34
        weights(g) = num_occs(g) + k_jm(g)*weights(g);
      weights.Scale(1.0 / weights.Sum()); // c.f. eq. 3.34 (denominator)
    }

    // adjust according to the learning rate
    weights.AddVec((1 - lrate) / lrate, weights_);
    weights.Scale(1.0 / weights.Sum());


    for (int32 g = 0; g < num_comp; g++) {   // c.f. eq. 3.32 in Dan Povey's thesis.
      if (weights(g) < 0)  // Check weight
      {
        printf("%d weight = %f < 0.\n", g, weights(g));
        KALDI_ASSERT(0);
      }
      weight_auxf_at_end +=
          num_occs(g) * log (weights(g))
          - den_occs(g) * weights(g) / weights_(g);
    }

    impr = weight_auxf_at_end - weight_auxf_at_start;
    weights_.CopyFromVec(weights);
    return impr;
}
BaseFloat EbwAmMfaUpdater::Update_w(const MleAmMfaAccs &num_accs,
                                    const MleAmMfaAccs &den_accs,
                                    AmMfa *model) const
{
  BaseFloat tot_impr = 0.0;
  for (int j = 0; j < model->NumStates(); ++j) {
    if (num_accs.s0_[j].Sum() + den_accs.s0_[j].Sum() < options_.s0_thresh_) {
      KALDI_LOG << "Not updating weights for this state because total count is "
                << num_accs.s0_[j].Sum() + den_accs.s0_[j].Sum() << " < "
                << options_.s0_thresh_;
      continue;
    }
    tot_impr += Update_w_for_GMM(model->sFaWeight_[j], num_accs.s0_[j],
                                 den_accs.s0_[j], options_.tau_w, options_.lrate_w);
    //printf("Total weight for state %d: %f\n", j,model->sFaWeight_[j].Sum() );
  }
  // MleAmMfaUpdater::ShrinkAmMfa(model);
  return tot_impr;
}

}  // namespace kaldi
