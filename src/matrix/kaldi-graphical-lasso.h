// matrix/kaldi-graphical-lasso.h

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

#ifndef KALDI_MATRIX_KALDI_GRAPHICAL_LASSO_H_
#define KALDI_MATRIX_KALDI_GRAPHICAL_LASSO_H_

#include <string>
#include <vector>

#include "base/kaldi-common.h"
#include "matrix/matrix-lib.h"
#include "util/parse-options.h"
#include "matrix/kaldi-gpsr.h"

namespace kaldi {

/** \struct GpsrConfig
 *  Configuration variables needed in the Graphical Lasso algorithm.
 */
struct GraphicLassoConfig {
  int32 max_iters;        ///< Maximum number of iterations
  double graphlasso_tau;  ///< Regularization scale
  double stop_tol;        ///< Stop tolerance

  GraphicLassoConfig() {
    max_iters = 100;
    graphlasso_tau = 10;
    stop_tol = 1.0e-6;
  }

  void Register(ParseOptions *po)
  {
    std::string module = "GraphicLassoConfig: ";
    po->Register("max-iters", &max_iters, module+
                 "Maximum number of iterations of Graphical Lasso.");
    po->Register("graphlasso-tau", &graphlasso_tau, module+
                   "Regularization weight for Graphical Lasso.");
    po->Register("stop-tol", &stop_tol, module+
                       "Stop tolerance for Graphical Lasso.");
  }
};

template<typename Real>
Real Diff_L1(const SpMatrix<Real>& W1, const SpMatrix<Real>& W2)
{
  Real diff = 0.0;
  for(int32 i = 0; i < W1.NumRows(); ++ i)
  {
    for(int32 j = 0; j < i; ++ j)
      diff += fabs(W1(i, j) - W2(i, j));
  }
  return 2.0 * diff;
}

template<typename Real>
Real L1NormSp(const SpMatrix<Real>& M)
{
  Real ret = 0.0;
  for(int32 i = 0; i < M.NumRows(); ++ i)
  {
    for(int32 j = 0; j < i; ++ j)
      ret += 2 * fabs(M(i, j));
    ret += fabs(M(i, i));
  }
  return ret;
}

template<typename Real>
Real GraphicalLassoObj(const SpMatrix<Real>& S, const SpMatrix<Real>& Theta, Real tau)
{
  Real obj = 0.0;
  obj -= Theta.LogPosDefDet();
  obj += TraceSpSp(S, Theta);
  obj += tau * L1NormSp(Theta);
  return obj;
}

/// Function: Solve the graphical Lasso
/// minimize_{Theta > 0} tr(S*Theta) - logdet(Theta) + rho * ||Theta||_1
/// Ref: Friedman et al. (2007) Sparse inverse covariance estimation with the
/// graphical lasso. Biostatistics.
/// Note: This function needs to call an algorithm that solves the Lasso
/// problem. Here, we choose to use to the GPSR for this purpose. However,
/// any Lasso algorithm in the penelized form will work.
///
/// Input:
/// S -- sample covariance matrix
/// opt.graphlasso_tau --  regularization parameter
/// opt.max_iters -- maximum number of iterations
/// opt.stop_tol -- convergence tolerance level
///
/// Output:
/// pW -- regularized covariance matrix estimate, W = Theta^-1
///
template<typename Real>
void GraphicalLasso(const SpMatrix<Real>& S, SpMatrix<Real>* pW, SpMatrix<Real>* pInvW, const GraphicLassoConfig&  opt)
{
  SpMatrix<Real>& W = *pW;
  W.CopyFromSp(S);
  int32 dim = W.NumRows();

  // init W
  for(int32 i = 0; i < dim; ++ i)
  {
    W(i, i) += opt.graphlasso_tau;
  }

  bool bContinue = true;
  GpsrConfig gpsrConfig;
  gpsrConfig.gpsr_tau = opt.graphlasso_tau;
  gpsrConfig.max_sparsity = 1.0;

  Vector<Real> beta(dim - 1);
  Vector<Real> w12(dim - 1);
  Vector<Real> s12(dim - 1);
  SpMatrix<Real> W11(dim - 1);
  SpMatrix<Real> InvW11(dim - 1);
  Matrix<Real> Beta(dim, dim - 1);
//  Real l1_change = 0.0;

  SpMatrix<Real> W_old = W;
  int32 iter = 0;
  Real obj_old = GraphicalLassoObj(W_old, S, (Real)opt.graphlasso_tau);
  Real obj = 0.0, obj_change = -1.0;

  while(bContinue)
  {
    // for each row
    for(int32 i = 0; i < dim; ++ i)
    {
      // get W11 and s12
      for(int32 j = 0; j < dim; ++ j)
      {
        int32 j2 = j;
        if (j == i)
          continue;
        else if (j > i)
          --j2;
        for(int32 k = 0; k <= j; ++ k)
        {
          int32 k2 = k;
          if (k == i)
            continue;
          else if (k > i)
            -- k2;
          W11(j2, k2) = W(j, k);
        }

        // get s12
        s12(j2) = S(i, j);
        // get w12
        w12(j2) = W(i, j);
      }
      // Init beta
      InvW11 = W11;
      InvW11.Invert();
      beta.AddSpVec(1.0, InvW11, w12, 0.0);

      // solve the l1 problem
      if (W11.IsPosDef() == false)
        KALDI_WARN << "W11 is not positive definite!";
      Gpsr(gpsrConfig, W11, s12, &beta, "graphical-lasso");
      // Record beta for fast inverse matrix calculation
      Beta.CopyRowFromVec(beta, i);

      // Calculate w12 
      w12.AddSpVec(1.0, W11, beta, 0.0);
      // copy theta to W
      for(int32 j = 0; j < dim; ++ j)
      {
        int32 j2 = j;
        if (j == i)
          continue;
        else if (j > i)
          --j2;
        W(i, j) = w12(j2);
      }
    }

		++iter;

		obj = GraphicalLassoObj(W, S, (Real) opt.graphlasso_tau);
		if (obj > obj_old)
		{
			KALDI_WARN<< "obj_old = " << obj_old << ", obj = " << obj << ", obj > obj_old. Stop iteration. \n";
			break;
		}
		obj_change = fabs((obj_old - obj) / obj_old);
		KALDI_LOG<< "obj_old  = " << obj_old << ", obj  = " << obj << ", relative obj_change = "<< obj_change << "\n";
		if (obj_change < opt.stop_tol) {
			KALDI_VLOG(1) << "obj_change (" << obj_change
					<< ") bellow the predefined threshold (" << opt.stop_tol
					<< "), stop iteration (iter = " << iter << ").";
			bContinue = false;
		} else {
			// check iteration number
			if (opt.max_iters > 0 && iter > opt.max_iters) {
				KALDI_VLOG(1) << "Maximum iteration number (" << opt.max_iters
						<< ") reached, stop iteration.";
				break;
			}
			obj_old = obj;
			W_old.CopyFromSp(W);

//    // Check L1 change
//    l1_change = Diff_L1(W, W_old);
//    KALDI_LOG << "W_old = " <<  W_old << "\n";
//    KALDI_LOG << "W = " <<  W << "\n";
//    KALDI_LOG << "W_old 2 norm = " << W_old.FrobeniusNorm() << ", l1_change = " << l1_change << ", l1/2norm = "<< l1_change / W_old.FrobeniusNorm() <<  "\n";
//    if (l1_change / W.FrobeniusNorm() < opt.stop_tol)
//    {
//      KALDI_VLOG(1) << "L1 change (" << l1_change << ") bellow the predefined threshold ("
//          << opt.stop_tol << "), stop iteration (iter = " << iter << ").";
//      bContinue = false;
//    }
//    else
//    {
//      // check iteration number
//      if (opt.max_iters > 0 && iter > opt.max_iters)
//      {
//        KALDI_VLOG(1) << "Maximum iteration number (" << opt.max_iters << ") reached, stop iteration.";
//        break;
//      }
//
//      W_old.CopyFromSp(W);
    }

  }

  // Fast inverse covariance matrix calculation
  if (pInvW != NULL)
  {
     SpMatrix<Real>& invW = *pInvW;
     invW.CopyFromSp(W);
     invW.Invert();
//     for (int32 i = 0; i < dim; ++ i)
//    {
//      // get w12
//      for(int32 j = 0; j < dim; ++ j)
//      {
//        int32 j2 = j;
//        if (j == i)
//          continue;
//        else if (j > i)
//          --j2;
//        w12(j2) = W(i, j);
//      }
//      // calucluate the diagonal elements
//      invW(i, i) = 1.0 / (W(i, i) - VecVec(w12, Beta.Row(i)));
//      beta.SetZero();
//      beta.AddVec(-invW(i, i), Beta.Row(i));
//      for(int32 j = 0; j < dim; ++ j)
//      {
//        int32 j2 = j;
//        if (j == i)
//          continue;
//        else if (j > i)
//          --j2;
//        invW(i, j) = beta(j2);
//      }
//    }
  }
}

}

#endif
