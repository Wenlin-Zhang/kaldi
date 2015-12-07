// mfa2/estimate-am-mfa2-ebw.cc

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

#include "base/kaldi-common.h"
#include "matrix/kaldi-graphical-lasso.h"
#include "mfa2/estimate-am-mfa2-ebw.h"
#include "thread/kaldi-thread.h"
#include <algorithm>  // for std::max

namespace kaldi {

void EbwAmMfa2Updater::Update(const MleAmMfa2Accs &num_accs,
                             const MleAmMfa2Accs &den_accs,
                             AmMfa2 *model,
                             AmMfaUpdateFlagsType flags,
                             BaseFloat *auxf_change_out,
                             BaseFloat *count_out) {

  KALDI_ASSERT((flags & (kAmMfaPhoneVectors | kAmMfaPhoneWeights  | kAmMfaCovarianceMatrix)) != 0);

  num_accs.Check(*model, false);
  den_accs.Check(*model, false);
  
  BaseFloat tot_impr = 0.0;

  /// update phone vectors
  if (flags & kAmMfaPhoneVectors)
     tot_impr += Update_y(num_accs, den_accs, model);

   ///  update phone weights
   if (flags & kAmMfaPhoneWeights)
      tot_impr += Update_w(num_accs, den_accs, model);
  
  /// update phone covariance
  if (flags & kAmMfaCovarianceMatrix)
    tot_impr += Update_Sigma(num_accs, den_accs, model);

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

BaseFloat EbwAmMfa2Updater::Update_y(const MleAmMfa2Accs &num_accs,
                                 const MleAmMfa2Accs &den_accs,
                                 AmMfa2 *model) const {
                                         
  KALDI_LOG << "Updating phone vectors.";
  
  const MFA& mfa = model->GetMFA();  
  BaseFloat count = 0.0, auxf_impr = 0.0;

  int32 dim = model->FeatureDim();
  for(int j = 0; j < model->NumStates(); ++ j)
  {
    const std::vector<int32>& faIndex = model->sFaIndex_[j];
    for (int k = 0; k < faIndex.size(); ++ k)
    {
      int i = faIndex[k];
      
      Vector<BaseFloat>& y = model->sFaLocation_[j][k];
      
      BaseFloat gamma_ji_num = num_accs.s0_[j](k);
      BaseFloat gamma_ji_den = den_accs.s0_[j](k);

      if (gamma_ji_num < options_.s0_thresh_) {
            KALDI_WARN << "Not updating phone vector for j = " << j << ", k = " << k
                       << ", because numerator count is "    << gamma_ji_num << " < "   << options_.s0_thresh_;
            continue;
          }

      Vector<BaseFloat> s_ji_num(num_accs.s1_[j].Row(k));
      s_ji_num.AddVec(-gamma_ji_num, mfa.fa_info_vec_[i]->mu_);
      Vector<BaseFloat> s_ji_den(den_accs.s1_[j].Row(k));
      s_ji_den.AddVec(-gamma_ji_den, mfa.fa_info_vec_[i]->mu_);

      Vector<BaseFloat> g_num(mfa.k_vec_[i]), g_den(mfa.k_vec_[i]);
      int32 localDim = mfa.GetLocalDim(i);
      const Matrix<BaseFloat>& W = mfa.GetLocalBases(i);
      Matrix<BaseFloat> InvSigmaW(dim, localDim);
      InvSigmaW.AddSpMat(1.0, model->sFaInvSigma_[j][k], W, kNoTrans, 0.0);
      g_num.AddMatVec(1.0, InvSigmaW, kTrans, s_ji_num, 0.0);
      g_den.AddMatVec(1.0, InvSigmaW, kTrans, s_ji_den, 0.0);

      SpMatrix<BaseFloat> M(localDim);
      M.SetUnit();
      M.AddMat2Sp(1.0, W, kTrans, model->sFaInvSigma_[j][k], 1.0);
      SpMatrix<BaseFloat> H_num(M), H_den(M);
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

BaseFloat EbwAmMfa2Updater::Update_Sigma(const MleAmMfa2Accs &num_accs,
                                     const MleAmMfa2Accs &den_accs,
                                     AmMfa2 *model) const {
	KALDI_ASSERT(num_accs.FeatureDim() == den_accs.FeatureDim());
	int dim = num_accs.FeatureDim();

	SpMatrix<BaseFloat> Sigma(dim);
	GraphicLassoConfig opt;
	opt.graphlasso_tau = options_.glasso_tau_;
	for (int i = 0; i < model->NumStates(); ++i) {
		for (int j = 0; j < model->NumComps(i); ++j) {
			if (num_accs.s0_[i](j) < options_.s0_thresh_) {
				KALDI_LOG<< "The " << i << "th factor model (Sigma) is not updated due to small occupation.";
				continue;
			}
			SpMatrix<BaseFloat> origSigma(dim);
			model->GetCov(i, j, &origSigma);
			BaseFloat eta = (1- options_.lrate_Sigma) * num_accs.s0_[i](j) + (1 + options_.lrate_Sigma) * den_accs.s0_[i](j);
			origSigma.Scale(eta);
			origSigma.AddSp(1.0, num_accs.s2_[i][j]);
			origSigma.AddSp(-1.0, den_accs.s2_[i][j]);
			origSigma.Scale(1.0 / (num_accs.s0_[i](j) - den_accs.s0_[i](j) + eta));
			GraphicalLasso(origSigma, &Sigma, &(model->sFaInvSigma_[i][j]), opt);
		}
	}

	return 0.0;
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
BaseFloat EbwAmMfa2Updater::Update_w(const MleAmMfa2Accs &num_accs,
                                    const MleAmMfa2Accs &den_accs,
                                    AmMfa2 *model) const
{
  BaseFloat tot_impr = 0.0;
  for (int j = 0; j < model->NumStates(); ++j) {
    if (num_accs.s0_[j].Sum() < options_.s0_thresh_) {
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
