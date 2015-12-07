
all:
EXTRA_CXXFLAGS = -Wno-sign-compare
include ../kaldi.mk

BINFILES = am-mfa2-acc-stats am-mfa2-sum-accs am-mfa2-est am-mfa-to-am-mfa2\
           am-mfa2-align-compiled am-mfa2-decode-faster am-mfa2-latgen-faster \
           get-am-mfa2-state-occs am-mfa2-acc-stats2 am-mfa2-est-ebw am-mfa2-rescore-lattice \
           am-mfa2-post-to-gpost am-mfa2-info am-mfa2-acc-stats-gpost


OBJFILES =


TESTFILES = 

# Note: we intentionally list kaldi-gmm.a both before and after kaldi-transform.a, because
# each requires symbols from the other.
ADDLIBS = ../decoder/kaldi-decoder.a ../lat/kaldi-lat.a  ../feat/kaldi-feat.a \
          ../mfa2/kaldi-mfa2.a ../mfa/kaldi-mfa.a ../transform/kaldi-transform.a ../gmm/kaldi-gmm.a \
          ../hmm/kaldi-hmm.a ../tree/kaldi-tree.a ../matrix/kaldi-matrix.a  \
          ../thread/kaldi-thread.a ../util/kaldi-util.a ../base/kaldi-base.a


include ../makefiles/default_rules.mk
