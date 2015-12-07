// mfa2bin/am-mfa2-info.cc

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

#include <iomanip>

#include "base/kaldi-common.h"
#include "util/common-utils.h"

#include "mfa2/am-mfa2.h"
#include "hmm/transition-model.h"


int main(int argc, char *argv[]) {
  try {
    using namespace kaldi;
    typedef kaldi::int32 int32;
    const char *usage =
        "Print various information about an AmMfa2.\n"
        "Usage: am-mfa2-info [options] <model-in> [model-in2 ... ]\n";

    bool am_mfa2_detailed = false;
    bool trans_detailed = false;
    bool mfa_detailed = false;

    ParseOptions po(usage);
    po.Register("am-mfa2-detailed", &am_mfa2_detailed,
                "Print detailed information about substates.");
    po.Register("mfa-detailed", &mfa_detailed,
                    "Print detailed information about MFA background model.");
    po.Register("trans-detailed", &trans_detailed,
                "Print detailed information about transition model.");

    po.Read(argc, argv);
    if (po.NumArgs() < 1) {
      po.PrintUsage();
      exit(1);
    }

    for (int i = 1, max = po.NumArgs(); i <= max; ++i) {
      std::string model_in_filename = po.GetArg(i);
      AmMfa2 am_mfa2;
      TransitionModel trans_model;
      {
        bool binary;
        Input ki(model_in_filename, &binary);
        trans_model.Read(ki.Stream(), binary);
        am_mfa2.Read(ki.Stream(), binary);
      }

      {
        using namespace std;
        am_mfa2.Check(false);
        cout << "Check AmMfa2 model OK!\n";
        cout.setf(ios::left);
        cout << "\nModel file: " << model_in_filename << endl;

        cout << " Mfa information:\n"
             << setw(40) << "  # of components" << am_mfa2.GetMFA().NumComps() << endl;
        if (mfa_detailed)
        {
          const MFA& mfa = am_mfa2.GetMFA();
          for(int32 j = 0; j < mfa.NumComps(); ++ j)
          {
            cout << "  local dimensions for state " << setw(13) << j << mfa.GetLocalDim(j) << endl;
          }
          cout << endl;
        }

        cout << " AmMfa information:\n"
          << setw(40) << "  # of HMM states" << am_mfa2.NumStates() << endl
          << setw(40) << "  Dimension of feature vectors"
          << am_mfa2.FeatureDim() << endl;

        int32 total_comp = 0;
        int32 total_parm_cnt = 0;
        int32 total_invcov_nonzero = 0;
        int32 total_invcov_parm_cnt = 0;
        int32 dim = am_mfa2.FeatureDim();
        for (int32 j = 0; j < am_mfa2.NumStates(); j++) {
          if (am_mfa2_detailed)
          {
            cout << "  # of components for state " << setw(13) << j
                 << am_mfa2.NumComps(j) << endl;
            cout << "weights: ";
            const Vector<BaseFloat>& weights = am_mfa2.GetWeights(j);
            for(int32 i = 0; i < am_mfa2.NumComps(j); ++ i )
            {
            	cout << weights(i) << "\t";

            	// accumulate and calculate the nonzero percentage of the precision matrix
            	int32 invcov_nonzero = 0;
            	const SpMatrix<BaseFloat>& inv_cov = am_mfa2.GetInvCov(j, i);
            	for(int32 k = 0; k < dim; ++k)
            	{
            		for(int32 l = 0; l <=k; ++ l)
            		{
            			if (fabs(inv_cov(k, l)) > 1.0e-5)
            			{
            				if (l == k)  invcov_nonzero += 1;
            				else invcov_nonzero += 2;
            			}
            		}
            	}
            	cout << "inverse covaraince nonzero percentage = " << ((BaseFloat)invcov_nonzero) / (dim * dim) << endl;
            	total_invcov_nonzero += invcov_nonzero;
            	total_invcov_parm_cnt += dim * dim;
            }
            cout << endl;
          }
          total_comp += am_mfa2.NumComps(j);
          total_parm_cnt += am_mfa2.GetStateParmCnt(j);
        }
        cout << setw(40) << "  Total # of components " << total_comp << endl;
        cout << setw(40) << "  Average # of components per state "
             << ((BaseFloat)total_comp) / am_mfa2.NumStates() << endl;
        cout << setw(40) << "  Toal # of state parameters: " << total_parm_cnt << endl;
        cout << setw(40) << "  Average nonzeo percentage of the invcov: " <<  ((BaseFloat)total_invcov_nonzero) / total_invcov_parm_cnt<< endl;

        cout << "\nTransition model information:\n"
             << setw(40) << " # of HMM states" << trans_model.NumPdfs() << endl
             << setw(40) << " # of transition states"
             << trans_model.NumTransitionStates() << endl;
          int32 total_indices = 0;
          for (int32 s = 0; s < trans_model.NumTransitionStates(); s++) {
            total_indices += trans_model.NumTransitionIndices(s);
            if (trans_detailed) {
              cout << "  # of transition ids for state " << setw(8) << s
                   << trans_model.NumTransitionIndices(s) << endl;
            }
          }
          cout << setw(40) << "  Total # of transition ids " << total_indices
               << endl;
      }
    }

    return 0;
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
}
