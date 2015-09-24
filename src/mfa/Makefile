all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

OBJFILES = mfa.o mle-mfa.o am-mfa.o decodable-am-mfa.o estimate-am-mfa.o estimate-am-mfa-ebw.o

LIBNAME = kaldi-mfa

ADDLIBS = ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../gmm/kaldi-gmm.a

include ../makefiles/default_rules.mk

