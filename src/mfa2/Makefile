all:

OPENFST_CXXFLAGS = 
OPENFST_LDLIBS = 
include ../kaldi.mk

OBJFILES = am-mfa2.o decodable-am-mfa2.o estimate-am-mfa2.o estimate-am-mfa2-ebw.o

LIBNAME = kaldi-mfa2

ADDLIBS = ../tree/kaldi-tree.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a ../gmm/kaldi-gmm.a ../mfa/kaldi-mfa.a

include ../makefiles/default_rules.mk

