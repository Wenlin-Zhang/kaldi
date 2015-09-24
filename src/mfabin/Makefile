
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = mfa-init mfa-acc-stats mfa-est mfa-sum-accs mfa-info \
           am-mfa-init am-mfa-acc-stats am-mfa-sum-accs am-mfa-est \
           am-mfa-align-compiled am-mfa-decode-faster am-mfa-latgen-faster \
           am-mfa-convert-cov-type get-am-mfa-state-occs am-mfa-to-diag-gmm \
           am-mfa-acc-stats2 am-mfa-est-ebw am-mfa-rescore-lattice \
           am-mfa-est-spkvecs am-mfa-est-spkvecs-gpost am-mfa-post-to-gpost \
           am-mfa-info am-mfa-acc-stats-gpost am-mfa-init-spk-proj am-mfa-shrink \
           am-mfa-gselect am-mfa-post-to-mfa-post-sum am-mfa-shrink-by-mfa-post-sum


OBJFILES =


TESTFILES = 

# Note: we intentionally list kaldi-gmm.a both before and after kaldi-transform.a, because
# each requires symbols from the other.
ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a  ../feat/kaldi-feat.a \
          ../mfa/kaldi-mfa.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
          ../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk
