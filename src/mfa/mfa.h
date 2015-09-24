// mfa/mfa.h

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

#ifndef KALDI_MFA_H_
#define KALDI_MFA_H_ 1

#include<vector>

#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "matrix/matrix-lib.h"
#include "itf/options-itf.h"
#include "gmm/diag-gmm.h"

namespace kaldi {

class FullGmm;

/*
 * Init the MFA model options
 */
struct InitMFAOptions
{
  BaseFloat lambda_percentage_;
  int32 phn_space_dim_;


  InitMFAOptions() : lambda_percentage_(0.9), phn_space_dim_(-1)
  {
  }

  void Register(OptionsItf *po) {
    std::string module = "InitMFAOptions: ";
    po->Register("mfa-init-lambda", &lambda_percentage_,
                 module+"The reserved latent dimension weight percentage.");
    po->Register("mfa-init-dim", &phn_space_dim_, "Phone space dimension (-1 for automatic determination).");
  }
};

/**
 * Check the consistence of the MFA information
 */
inline bool CheckMFAInfo(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec)
{
  if (dim <= 0 || num_comp <= 0)
    return false;
  if (num_comp != k_vec.size())
    return false;
  for(std::size_t i = 0; i < num_comp; ++ i)
    if (k_vec[i] < 0 || k_vec[i] > dim)
      return false;
  return true;
}

typedef struct Covariance_
{
  Vector<BaseFloat> diagCov_;
  SpMatrix<BaseFloat> fullCov_;
}Covariance;

typedef enum CovType_
{
  DIAG,
  FULL,
  UNKN
}CovType;

CovType CharToCovType(char strCovType);

char CovTypeToChar(CovType type);

struct MfaGselectConfig {
  /// Number of highest-scoring diagonal-covariance Gaussians per frame.
  int32 diag_gmm_nbest;

  MfaGselectConfig() {
    diag_gmm_nbest = 25;
  }

  MfaGselectConfig(const MfaGselectConfig& opt) {
    diag_gmm_nbest = opt.diag_gmm_nbest;
  }

  void Register(OptionsItf *po) {
    po->Register("diag-gmm-nbest", &diag_gmm_nbest, "Number of highest-scoring"
        " diagonal-covariance Gaussians selected per frame.");
  }
};


/** \class MFA Definition for Mixture of Factor Analyzers
 */
class MFA {

  /// this makes it a little easier to modify the internals
  friend class AccumMFA;
  friend class MleAmMfaUpdater;
  friend class EbwAmMfaUpdater;
  friend class AmMfa;
  friend class MleAmMfaUpdater2;
  friend class EbwAmMfaUpdater2;
  friend class AmMfa2;

public:

  /// constructor
  MFA(): covType_(DIAG), num_comps_(0), dim_(0), isPreComputed_(false), isPreSelectGauss_(false)
  {
  }

  /// constructor
  MFA(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec, CovType covType):
    isPreComputed_(false), isPreSelectGauss_(false)
  {
    Resize(dim, num_comp, k_vec, covType);
  }

  MFA(const MFA* mfa)
  {
    CopyFromMFA(mfa);
  }

  CovType CovarianceType()const { return covType_; }
  std::size_t Dim()const { return dim_; }
  std::size_t NumComps()const { return num_comps_; }
  const std::vector<std::size_t>& LocalDims()const { return k_vec_; }
  const Vector<BaseFloat>& Weights()const { return pi_vec_; }
  BaseFloat GetWeight(std::size_t i)const { KALDI_ASSERT(i >= 0 && i < num_comps_); return pi_vec_(i); }
  std::size_t GetLocalDim(std::size_t i)const { KALDI_ASSERT(i >= 0 && i < num_comps_); return k_vec_[i]; }
  const Matrix<BaseFloat>& GetLocalBases(std::size_t i)const
  {
	  KALDI_ASSERT(i >= 0 && i < num_comps_);
	  return fa_info_vec_[i]->W_;
  }
  const Vector<BaseFloat>& GetLocalCenter(std::size_t i)const
  {
    KALDI_ASSERT(i >= 0 && i < num_comps_);
  	return fa_info_vec_[i]->mu_;
  }
  const Covariance& GetLocalVar(std::size_t i)const
  {
    KALDI_ASSERT(i >= 0 && i < num_comps_);
  	return fa_info_vec_[i]->sigma_;
  }
  void GetCovarianceMatrix(int32 i, SpMatrix<BaseFloat>& cov)const
  {
    if (cov.NumRows() != dim_)
      cov.Resize(dim_);
    fa_info* pFaInfo = fa_info_vec_[i];
    switch (covType_)
    {
      case DIAG:
        cov.AddDiagVec(1.0, pFaInfo->sigma_.diagCov_);
        break;
      case FULL:
        cov.CopyFromSp(pFaInfo->sigma_.fullCov_);
        break;
      default:
        KALDI_ERR << "Invalid covariance matrix type.";
        break;
    }
  }
  void GetInvCovarianceMatrix(int32 i, SpMatrix<BaseFloat>& cov)const
  {
    if (cov.NumRows() != dim_)
      cov.Resize(dim_);
    if (this->IsPreCompute() == true)
      cov.CopyFromSp(this->pre_data_vec_[i]->inv_Sigma_);
    else
    {
      GetCovarianceMatrix(i, cov);
      cov.Invert();
    }
  }

  void Resize(std::size_t dim, std::size_t num_comp, const std::vector<std::size_t>& k_vec, CovType covType)
  {
    if (CheckMFAInfo(dim, num_comp, k_vec) == false)
      KALDI_ERR << "CheckMFAInfo Error! Invalid MFA dimension size!";

    if (num_comps_ > 0)
      Clear();

    dim_ = dim;
    num_comps_ = num_comp;
    covType_ = covType;
    pi_vec_.Resize(num_comps_);
    k_vec_ = k_vec;
    for(std::size_t i = 0; i < num_comps_; ++ i)
    {
      fa_info* pInfo = new fa_info(dim_, k_vec_[i], covType_);
      fa_info_vec_.push_back(pInfo);
    }
  }

  void CopyFromMFA(const MFA& mfa);

  void Clear()
  {
    UnPreCompute();

    for(std::size_t i = 0; i < num_comps_; ++ i)
    {
      delete (fa_info_vec_[i]);
      fa_info_vec_[i] = NULL;
    }
    fa_info_vec_.clear();

    dim_ = num_comps_ = 0;
    k_vec_.clear();
    pi_vec_.Resize(0);
  }

  /// Set one of the component parameters
  void SetFA(std::size_t i, std::size_t k, const MatrixBase<double>& W, const VectorBase<double>& mu, const VectorBase<double>& sigma)
  {
    // Do any parameter check
    KALDI_ASSERT(i >= 0 && i < num_comps_);
    KALDI_ASSERT(k > 0 && k < dim_);
    KALDI_ASSERT(W.NumRows() == dim_ && W.NumCols() == k);
    KALDI_ASSERT(mu.Dim() == dim_);
    KALDI_ASSERT(sigma.Dim() == dim_);

    // this function is only valid for diagonal covariance type, which is the normal MFA
    KALDI_ASSERT(covType_ == DIAG);

    k_vec_[i] = k;
    fa_info* pInfo = new fa_info(dim_, k, covType_);
    pInfo->W_.CopyFromMat(W);
    pInfo->mu_.CopyFromVec(mu);
    pInfo->sigma_.diagCov_.CopyFromVec(sigma);
    delete fa_info_vec_[i];
    fa_info_vec_[i] = pInfo;
  }

  ///
  virtual ~MFA()
  {
    Clear();
  }

  /// Initialize the model
  void Init(const FullGmm& fgmm, const InitMFAOptions& opts);

  /// Convert the covariance matrix to diagonal type
  void ConvertToDiagCov();

  /// Convert the covariance matrix to full type
  void ConvertToFullCov();

  /// Convert the model to a full covariance GMM, cpw determines whether copy the weight
  void ConvertToFullGmm(FullGmm* pFGmm, bool cpw = true)const;

  /// Delete components
  void DeleteComponents(const std::vector<int32> indexVec);

  /// whether using gselect
  bool IsPreSelGauss() const {
    return isPreSelectGauss_;
  }

  /// Set the gselect flag.
  /// Note that the MfaGselectConfig is stored in the MFA object,
  /// So during decoding, there is no need to pass it to the GaussianSelection call.
  /// We can simply set the bPreSelGauss to be true to turn on the gselect option.
  /// If we leave this flag to false, and don't give the gselect rspecifier,
  /// the gselect option will be turned off.
  void SetPreSelGauss(bool bPreSelGauss = true, const MfaGselectConfig* pOpt =
                          NULL);

  /// Computes the top-scoring Gaussian indices (used for pruning of later
  /// stages of computation). Returns frame log-likelihood given selected
  /// Gaussians from full UBM.
  void GaussianSelection(const VectorBase<BaseFloat> &data,
                                std::vector<int32> *gselect) const;

  /// Returns the log-likelihood of a data point (vector) given the MFA
  BaseFloat LogLikelihood(const VectorBase<BaseFloat> &data, std::vector<int32> *gselect = NULL);

  /**
     * Computes the log likelihood of a data vector under one of the
     * mixture components.
     *
     * To compute the likelihood of the data vector, we make use of
     * the following formula:
     * \f[
     *     p(y) = N(\mu, W W^\top + \Sigma)
     * \f]
     * To avoid computing and inverting this \f$d \times d\f$
     * covariance matrix, we use one the matrix inversion lemma.
     * This yields
     * \f[
     *   (W W^\top + \Sigma)^{-1} =
     *   \Sigma^{-1} - \Sigma^{-1} W (I + W^\top \Sigma^{-1} W)^{-1}
     *      W^\top \Sigma^{-1}
     * \f]
     * Plugging this into the Gaussian model, we get that the
     * likelihood is
     * \f[
     *   p(y) = \frac{1}{Z} \exp \left\{
     *    -\frac{1}{2} [(y - \mu)^\top \Sigma^{-1}(y - \mu) - q M^{-1}q^\top]
     *   \right\}
     * \f]
     *
     * where \f$q\f$ is defined as in the documentation for #q,
     * \f$M\f$ is defined as in the documentation for #M, and
     * \f$Z\f$ is the normalization constant which is \f$(2 \pi)^{-d/2}
     * |W W^\top + \Sigma|^{-1/2}\f$.  To compute this determinant, we
     * use the fact that \f$|W W^\top + \Sigma| = |\Sigma| \cdot |I +
     * W^\top \Sigma^{-1} W| = |\Sigma| \cdot |M|\f$.
     *
     * @param  i
     *         the index of the mixture component for which the
     *         log likelihood is computed
     * @param  data
     *         An \f$d \times 1\f$ vector representing an observation
     *         of (a subset of) \f$Y\f$.
     * @return The log likelihood of the observed values under the
     *         mixture component (using the natural logarithm).
     */
  BaseFloat LogLikelihood(const VectorBase<BaseFloat> &data, std::size_t i, VectorBase<BaseFloat>* q1 = NULL, VectorBase<BaseFloat>* expected_x1 = NULL);

  /// is precompute or not
  bool IsPreCompute()const
  {
    return isPreComputed_;
  }

  /// precompute the data
  void PreCompute();

  /// release the precomputed data
  void UnPreCompute();

  void Write(std::ostream &os, bool binary) const;
  void Read(std::istream &is, bool binary);

  void ComputeFeatureNormalizer(Matrix<BaseFloat>* pMat)const;

protected:
    /**
     * Information about one factor analyzer model in the mixture.
     */
    struct fa_info {
      /**
       * Constructor.
       *
       * @param dim the dimensionality of the latent continuous
       *          variable \f$X\f$
       * @param k the dimensionality of the observed continuous
       *          variable \f$Y\f$
       */
      fa_info(std::size_t dim, std::size_t k, CovType covType)
      {
        W_.Resize(dim, k);
        mu_.Resize(dim);
        switch(covType)
        {
          case DIAG:
            sigma_.diagCov_.Resize(dim);
            break;
          case FULL:
            sigma_.fullCov_.Resize(dim);
            break;
          default:
            KALDI_ERR << "Invalid covariance matrix type.";
            break;
        }
      }

      //! The \f$d \times k\f$ factor loading matrix \f$W\f$.
      Matrix<BaseFloat> W_;

      //! The \f$d \times 1\f$ prior mean vector \f$\mu\f$.
      Vector<BaseFloat> mu_;

      //! The \f$d \times 1\f$ variance vector \f$\sigma\f$.
      Covariance sigma_;
    };

    /**
      * Pre-compute data for fast likelihood evaluation
      */
    struct pre_compute_data {
      //! M = I + W^T \Sigma^{-1} W
      SpMatrix<BaseFloat> M_;

      // inverse covariance
      SpMatrix<BaseFloat> inv_Sigma_;

      // inverse matrix of M
      SpMatrix<BaseFloat> inv_M_;

      //! Sigma1W = \Sigma^{-1} W
      Matrix<BaseFloat> Sigma1W_;
    };

    /**
     * The factor covariance type
     */
    CovType_ covType_;

    /**
     * The number of mixture components, i.e., the arity of the latent
     * discrete variable \f$Z\f$.
     */
    std::size_t num_comps_;

    //! The dimensionality of the observed continuous variable \f$Y\f$.
    std::size_t dim_;

    /**
     * A vector of \f$k\f$ values representing the mixing proportions
     * of the model.  \f$\pi_i\f$ is the prior probability a sample is
     * generated from the \f$i^{\textrm{th}}\f$ factor analyzer.
     */
    Vector<BaseFloat> pi_vec_;

    /**
     * A vector of objects representing the latent dimension of each local subspace.
     */
    std::vector<std::size_t> k_vec_;

    /**
     * A vector of objects representing the components of the mixture
     * model.
     */
    std::vector<fa_info*> fa_info_vec_;

    /**
     * Precompute data vector
     */
    std::vector<pre_compute_data*> pre_data_vec_;

    /**
     * Whether the precompute data is computed.
     */
    bool isPreComputed_;


    /// Following is for gmm pre-selection
    bool isPreSelectGauss_;
    /// Pre-selection options
    MfaGselectConfig gsOpt_;
    /// The background DiagGmm for pre-selection
    DiagGmm diagBgGmm_;
};

}

#endif
