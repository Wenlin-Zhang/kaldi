#!/bin/bash
# Copyright 2015  Wen-Lin Zhang
#
# This trains a MoFA (i.e. a mixture of factor analyzers), starting from a full-covariance GMM.
#

# Begin configuration section.
nj=4
cmd=run.pl
silence_weight=  # You can set it to e.g. 0.0, to weight down silence in training.
stage=-1
num_iters=4
use_gselect=1
num_gselect=25
phn_dim=-1      # set the phone dim to a fixed phn_dim
init_lambda=0.9 # adaptively set the phone dim, perserve 90% dimension
# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: steps/train_mfa.sh <data> <lang> <ali-dir> <ubm-dir> <exp>"
  echo " e.g.: steps/train_mfa.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/ubm4a exp/mfa4a"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --silence-weight <sil-weight>                    # weight for silence (e.g. 0.5 or 0.0)"
  echo "  --num-iters <#iters>                             # Number of iterations of E-M"
  exit 1;
fi

data=$1
lang=$2
alidir=$3
ubmdir=$4
dir=$5

for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $ubmdir/final.ubm; do
  [ ! -f $f ] && echo "No such file $f" && exit 1;
done

# Set various variables.
silphonelist=`cat $lang/phones/silence.csl` || exit 1;
nj=`cat $alidir/num_jobs` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
splice_opts=`cat $alidir/splice_opts 2>/dev/null` # frame-splicing options.

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: using transforms from $alidir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$alidir/trans.JOB ark:- ark:- |"
fi
##

if [ ! -z "$silence_weight" ]; then
  weights_opt="--weights='ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- | weight-silence-post $silence_weight $silphonelist $alidir/final.mdl ark:- ark:- | post-to-weights ark:- ark:- |'"
else
  weights_opt=
fi

# build initial MoFA model
if [ $stage -le -1 ]; then
  echo "$0: build initial MFA"
  $cmd $dir/log/cluster.log \
    mfa-init --verbose=2 --mfa-init-dim=$phn_dim --mfa-init-lambda=$init_lambda $ubmdir/final.ubm $dir/0.mfa   || exit 1;
fi

if [ $use_gselect -eq 1 ]; then
  gselect_opt="--use-gselect --diag-gmm-nbest=$num_gselect"
else
  gselect_opt=
fi

x=0
while [ $x -lt $num_iters ]; do
  echo "Pass $x"
  if [ $stage -le $x ]; then
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
      mfa-acc-stats $gselect_opt $weights_opt $dir/$x.mfa "$feats" $dir/$x.JOB.acc || exit 1;
    lowcount_opt="--remove-low-count-gaussians=false"

    $cmd $dir/log/update.$x.log \
      mfa-est --verbose=2 $dir/$x.mfa "mfa-sum-accs - $dir/$x.*.acc |" $dir/$[$x+1].mfa || exit 1;
  fi
  rm $dir/*.acc $dir/$x.mfa
  x=$[$x+1]
done

rm $dir/final.mfa 2>/dev/null
mv $dir/$x.mfa $dir/final.mfa || exit 1;

