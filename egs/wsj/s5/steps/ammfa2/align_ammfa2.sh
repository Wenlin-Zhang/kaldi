#!/bin/bash
# Copyright 2015  Wen-Lin Zhang
# Apache 2.0

# Computes training alignments given an AmMfa system.  
# If the system is built on top of SAT, you should supply
# transforms with the --transform-dir option.

# If you supply the --use-graphs option, it will use the training
# graphs from the source directory.

# Begin configuration section.  
stage=0
nj=4
cmd=run.pl
use_graphs=false # use graphs from srcdir
# Begin configuration.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
beam=10
retry_beam=40
transform_dir=
# End configuration options.

echo "$0 $@"  # Print the command line for logging

[ -f path.sh ] && . ./path.sh # source the path.
. parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "usage: steps/align_ammfa2.sh <data-dir> <lang-dir> <src-dir> <align-dir>"
   echo "e.g.:  steps/align_ammfa2.sh --transform-dir exp/tri3b data/train data/lang \\"
   echo "           exp/ammfa4a exp/ammfa4a_ali"
   echo "main options (for others, see top of script file)"
   echo "  --config <config-file>                           # config containing options"
   echo "  --nj <nj>                                        # number of parallel jobs"
   echo "  --use-graphs true                                # use graphs in src-dir"
   echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
   exit 1;
fi

data=$1
lang=$2
srcdir=$3
dir=$4

oov=`cat $lang/oov.int` || exit 1;
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
splice_opts=`cat $srcdir/splice_opts 2>/dev/null` # frame-splicing options.
sdata=$data/split$nj

## copy splice options and split data
mkdir -p $dir/log
cp $srcdir/splice_opts $dir 2>/dev/null # frame-splicing options.
echo $nj > $dir/num_jobs
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## copy model
cp $srcdir/{tree,final.mdl} $dir || exit 1;

## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    cp $srcdir/final.mat $dir    
   ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
  echo "$0: **WARNING**: you seem to be using an AM-MFA system trained with transforms,"
  echo "  but you are not providing the --transform-dir option during alignment."
fi
#if [ -f $srcdir/trans.1 ]; then
#  echo "$0: copy transforms from $srcdir"
#  cp $srcdir/trans.* $dir
#  echo "$0: use transforms from $dir"
#  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
#fi
##

## Set up model and alignment model.
mdl=$srcdir/final.mdl
if [ -f $srcdir/final.alimdl ]; then
  alimdl=$srcdir/final.alimdl
else
  alimdl=$srcdir/final.mdl
fi
[ ! -f $mdl ] && echo "$0: no such model $mdl" && exit 1;

## Work out where we're getting the graphs from.
if $use_graphs; then
  [ "$nj" != "`cat $srcdir/num_jobs`" ] && \
    echo "$0: you specified --use-graphs true, but #jobs mismatch." && exit 1;
  [ ! -f $srcdir/fsts.1.gz ] && echo "No graphs in $srcdir" && exit 1;
  graphdir=$srcdir
  ln.pl $srcdir/fsts.*.gz $dir
else
  graphdir=$dir
  if [ $stage -le 0 ]; then
    echo "$0: compiling training graphs"
    tra="ark:utils/sym2int.pl --map-oov $oov -f 2- $lang/words.txt $sdata/JOB/text|";   
    $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log  \
      compile-train-graphs $dir/tree $dir/final.mdl  $lang/L.fst "$tra" \
        "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
  fi
fi

if [ $alimdl == $mdl ]; then 
  # Speaker-independent decoding-- just one pass.  Not normal.
  T=`am-mfa-info $mdl | grep 'speaker vector space' | awk '{print $NF}'` || exit 1;
  [ "$T" -ne 0 ] && echo "No alignment model, yet speaker vector space nonempty" && exit 1;

  if [ $stage -le 2 ]; then
    echo "$0: aligning data in $data using model $dir/final.mdl (no speaker-vectors)"
    $cmd JOB=1:$nj $dir/log/align.JOB.log  \
       am-mfa2-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam \
       $dir/final.mdl "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
       "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
  fi
  echo "$0: done aligning data."
else

   echo "$0: Sorry, we did not support speaker adapted decoding for ammfa2 currently."

fi

utils/summarize_warnings.pl $dir/log

exit 0;
