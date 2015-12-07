// matrix/kaldi-graphical-lasso-test.cc

// Copyright 2013 Wen-Lin Zhang

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
#include "gmm/model-test-common.h"
#include "matrix/kaldi-gpsr.h"
#include "matrix/kaldi-graphical-lasso.h"
#include "util/kaldi-io.h"


using kaldi::int32;
using kaldi::BaseFloat;
namespace ut = kaldi::unittest;

namespace kaldi {

template<class Real> static void InitRand(VectorBase<Real> *v) {
  for (MatrixIndexT i = 0;i < v->Dim();i++)
    (*v)(i) = RandGauss();
}

template<class Real> static void InitRand(MatrixBase<Real> *M) {
 start:
  for (MatrixIndexT i = 0;i < M->NumRows();i++)
    for (MatrixIndexT j = 0;j < M->NumCols();j++)
      (*M)(i, j) = RandGauss();
    if (M->NumRows() != 0 && M->Cond() > 100) {
      KALDI_WARN << "Condition number of random matrix large" << M->Cond()
                 << ": trying again (this is normal)";
      goto start;
    }
}

template<class Real> static void InitRand(SpMatrix<Real> *M) {
 start_sp:
  for (MatrixIndexT i = 0;i < M->NumRows();i++)
    for (MatrixIndexT j = 0;j<=i;j++)
    {
      (*M)(i, j) = RandGauss();
      if (i == j && (*M)(i, j) < 0) // To ensure semi-definite
        (*M)(i, j) += 10;
    }

  if (M->NumRows() != 0 && M->Cond() > 100) {
    KALDI_WARN << "Condition number of random matrix large" << M->Cond()
               << ": trying again (this is normal)";
    goto start_sp;
  }
  if (M->IsPosDef() == false)
  {
    KALDI_WARN << "Not semi-definite, trying again (this is normal)";
    goto start_sp;
  }
}

template<class Real> static void UnitTestGpsr() {
  MatrixIndexT dim = (rand() % 10) + 10;
  SpMatrix<Real> S(dim), W(dim);
  SpMatrix<Real> invW(dim);
  InitRand(&S);
  GraphicLassoConfig opt;
  opt.graphlasso_tau = 1;
  GraphicalLasso(S, &W, &invW, opt);
  KALDI_LOG << "S = " << S;
  KALDI_LOG << "W = " << W;
  SpMatrix<Real> invS(S);
  invS.Invert();
  KALDI_LOG << "InvS = " << invS;
  KALDI_LOG << "InvW = " << invW;
  SpMatrix<Real> invW2(W);
  invW2.Invert();
  KALDI_LOG << "Direct calculate InvW = " << invW2; 

  Real old_obj = GraphicalLassoObj(S, invS, (Real)opt.graphlasso_tau);
  Real new_obj = GraphicalLassoObj(S, invW, (Real)opt.graphlasso_tau);
  Real new_obj2 = GraphicalLassoObj(S, invW2, (Real)opt.graphlasso_tau);

  KALDI_LOG << "objective " << old_obj << " v.s. " << new_obj << " v.s. " << new_obj2;
}

}

int main() {
  kaldi::g_kaldi_verbose_level = 1;
  kaldi::UnitTestGpsr<float>();
  std::cout << "Test OK.\n";
  return 0;
}
