#!/bin/bash
# Copyright 2013  Wen-Lin Zhang.  Apache 2.0.

# MMI training (or optionally boosted MMI, if you give the --boost option),
# for AmMfa.  4 iterations (by default) of Extended Baum-Welch update.
#
# Begin configuration section.
cmd=run.pl
num_iters=4
boost=0.0
cancel=true # if true, cancel num and den counts on each frame.
acwt=0.1
stage=0

transform_dir=
# End configuration section

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 5 ]; then
  echo "Usage: steps/train_mmi_ammfa.sh <data> <lang> <ali> <denlats> <exp>"
  echo " e.g.: steps/train_mmi_ammfa.sh data/train_si84 data/lang exp/tri2b_ali_si84 exp/tri2b_denlats_si84 exp/tri2b_mmi"
  echo "Main options (for others, see top of script file)"
  echo "  --boost <boost-weight>                           # (e.g. 0.1), for boosted MMI.  (default 0)"
  echo "  --cancel (true|false)                            # cancel stats (true by default)"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --config <config-file>                           # config containing options"
  echo "  --stage <stage>                                  # stage to do partial re-run from."  
  echo "  --transform-dir <transform-dir>                  # directory to find fMLLR transforms."
  exit 1;
fi

data=$1
lang=$2
alidir=$3
denlatdir=$4
dir=$5
mkdir -p $dir/log

for f in $data/feats.scp $alidir/{tree,final.mdl,ali.1.gz} $denlatdir/lat.1.gz; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done
nj=`cat $alidir/num_jobs` || exit 1;
[ "$nj" -ne "`cat $denlatdir/num_jobs`" ] && \
  echo "$alidir and $denlatdir have different num-jobs" && exit 1;

sdata=$data/split$nj
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
cp $alidir/splice_opts $dir 2>/dev/null
echo $nj > $dir/num_jobs

cp $alidir/{final.mdl,tree} $dir

silphonelist=`cat $lang/phones/silence.csl` || exit 1;

# Set up featuresl

if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $alidir/final.mat ark:- ark:- |"
    cp $alidir/final.mat $dir    
    ;;
  *) echo "Invalid feature type $feat_type" && exit 1;
esac

if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" \
    && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
else
  echo "$0: no fMLLR transforms."
fi

if [ -f $alidir/vecs.1 ]; then
  echo "$0: using speaker vectors from $alidir"
  spkvecs_opt="--spk-vecs=ark:$alidir/vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk"
else
  echo "$0: no speaker vectors."
  spkvecs_opt=
fi

lats="ark:gunzip -c $denlatdir/lat.JOB.gz|"
if [[ "$boost" != "0.0" && "$boost" != 0 ]]; then
  lats="$lats lattice-boost-ali --b=$boost --silence-phones=$silphonelist $alidir/final.mdl ark:- 'ark,s,cs:gunzip -c $alidir/ali.JOB.gz|' ark:- |"
fi


cur_mdl=$alidir/final.mdl
x=0
while [ $x -lt $num_iters ]; do
  echo "Iteration $x of MMI training"
  # Note: the num and den states are accumulated at the same time, so we
  # can cancel them per frame.
  if [ $stage -le $x ]; then

    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.y.log \
      am-mfa-rescore-lattice $spkvecs_opt $cur_mdl "$lats" "$feats" ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      sum-post --merge=$cancel --scale1=-1 \
      ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
      am-mfa-acc-stats2 --update-flags="y" $spkvecs_opt $cur_mdl "$feats" ark,s,cs:- \
        $dir/num_acc.$x.JOB.y.acc $dir/den_acc.$x.JOB.y.acc || exit 1;
    n=`echo $dir/{num,den}_acc.$x.*.y.acc | wc -w`;
    [ "$n" -ne $[$nj*2] ] && \
      echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
    $cmd $dir/log/den_acc_sum.$x.y.log \
      am-mfa-sum-accs $dir/den_acc.$x.y.acc $dir/den_acc.$x.*.y.acc || exit 1;
    rm $dir/den_acc.$x.*.y.acc
    $cmd $dir/log/num_acc_sum.$x.y.log \
      am-mfa-sum-accs $dir/num_acc.$x.y.acc $dir/num_acc.$x.*.y.acc || exit 1;
    rm $dir/num_acc.$x.*.y.acc
    echo "$x.1 Update y ..."
    $cmd $dir/log/update.$x.1.y.log \
      am-mfa-est-ebw --update-flags="y" $cur_mdl $dir/num_acc.$x.y.acc $dir/den_acc.$x.y.acc $dir/$[$x+1].1.y.mdl || exit 1;
    #cur_mdl=$dir/$[$x+1].1.y.mdl

    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.M.log \
      am-mfa-rescore-lattice $spkvecs_opt $cur_mdl "$lats" "$feats" ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      sum-post --merge=$cancel --scale1=-1 \
      ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
      am-mfa-acc-stats2 --update-flags="M" $spkvecs_opt $cur_mdl "$feats" ark,s,cs:- \
        $dir/num_acc.$x.JOB.M.acc $dir/den_acc.$x.JOB.M.acc || exit 1;
    n=`echo $dir/{num,den}_acc.$x.*.M.acc | wc -w`;
    [ "$n" -ne $[$nj*2] ] && \
      echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
    $cmd $dir/log/den_acc_sum.$x.M.log \
      am-mfa-sum-accs $dir/den_acc.$x.M.acc $dir/den_acc.$x.*.M.acc || exit 1;
    rm $dir/den_acc.$x.*.M.acc
    $cmd $dir/log/num_acc_sum.$x.M.log \
      am-mfa-sum-accs $dir/num_acc.$x.M.acc $dir/num_acc.$x.*.M.acc || exit 1;
    rm $dir/num_acc.$x.*.M.acc
    echo "$x.2 Update M ..."
    $cmd $dir/log/update.$x.2.M.log \
      am-mfa-est-ebw --update-flags="M" $dir/$[$x+1].1.y.mdl $dir/num_acc.$x.M.acc $dir/den_acc.$x.M.acc $dir/$[$x+1].2.M.mdl || exit 1;
    #cur_mdl=$dir/$[$x+1].2.M.mdl

    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.Smlog \
      am-mfa-rescore-lattice $spkvecs_opt $cur_mdl "$lats" "$feats" ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      sum-post --merge=$cancel --scale1=-1 \
      ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
      am-mfa-acc-stats2 --update-flags="Sm" $spkvecs_opt $cur_mdl "$feats" ark,s,cs:- \
        $dir/num_acc.$x.JOB.Sm.acc $dir/den_acc.$x.JOB.Sm.acc || exit 1;
    n=`echo $dir/{num,den}_acc.$x.*.Sm.acc | wc -w`;
    [ "$n" -ne $[$nj*2] ] && \
      echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
    $cmd $dir/log/den_acc_sum.$x.Sm.log \
      am-mfa-sum-accs $dir/den_acc.$x.Sm.acc $dir/den_acc.$x.*.Sm.acc || exit 1;
    rm $dir/den_acc.$x.*.Sm.acc
    $cmd $dir/log/num_acc_sum.$x.Sm.log \
      am-mfa-sum-accs $dir/num_acc.$x.Sm.acc $dir/num_acc.$x.*.Sm.acc || exit 1;
    rm $dir/num_acc.$x.*.Sm.acc
    echo "$x.3 Update Sm ..."
    $cmd $dir/log/update.$x.3.Sm.log \
      am-mfa-est-ebw --update-flags="Sm" $dir/$[$x+1].2.M.mdl $dir/num_acc.$x.Sm.acc $dir/den_acc.$x.Sm.acc $dir/$[$x+1].3.Sm.mdl || exit 1;
    #cur_mdl=$dir/$[$x+1].3.Sm.mdl
    
    $cmd JOB=1:$nj $dir/log/acc.$x.JOB.w.log \
      am-mfa-rescore-lattice $spkvecs_opt $cur_mdl "$lats" "$feats" ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      sum-post --merge=$cancel --scale1=-1 \
      ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
      am-mfa-acc-stats2 --update-flags="w" $spkvecs_opt $cur_mdl "$feats" ark,s,cs:- \
        $dir/num_acc.$x.JOB.w.acc $dir/den_acc.$x.JOB.w.acc || exit 1;
    n=`echo $dir/{num,den}_acc.$x.*.w.acc | wc -w`;
    [ "$n" -ne $[$nj*2] ] && \
      echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
    $cmd $dir/log/den_acc_sum.$x.w.log \
      am-mfa-sum-accs $dir/den_acc.$x.w.acc $dir/den_acc.$x.*.w.acc || exit 1;
    rm $dir/den_acc.$x.*.w.acc
    $cmd $dir/log/num_acc_sum.$x.w.log \
      am-mfa-sum-accs $dir/num_acc.$x.w.acc $dir/num_acc.$x.*.w.acc || exit 1;
    rm $dir/num_acc.$x.*.w.acc
    echo "$x.4 Update w ..."
    $cmd $dir/log/update.$x.4.w.log \
      am-mfa-est-ebw --update-flags="w" $dir/$[$x+1].3.Sm.mdl $dir/num_acc.$x.w.acc $dir/den_acc.$x.w.acc $dir/$[$x+1].4.w.mdl || exit 1;
    #cur_mdl=$dir/$[$x+1].4.w.mdl

    if [ -f $alidir/vecs.1 ]; then
      $cmd JOB=1:$nj $dir/log/acc.$x.JOB.N.log \
        am-mfa-rescore-lattice $spkvecs_opt $cur_mdl "$lats" "$feats" ark:- \| \
        lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
        sum-post --merge=$cancel --scale1=-1 \
        ark:- "ark,s,cs:gunzip -c $alidir/ali.JOB.gz | ali-to-post ark:- ark:- |" ark:- \| \
        am-mfa-acc-stats2 --update-flags="N" $spkvecs_opt $cur_mdl "$feats" ark,s,cs:- \
          $dir/num_acc.$x.JOB.N.acc $dir/den_acc.$x.JOB.N.acc || exit 1;
      n=`echo $dir/{num,den}_acc.$x.*.N.acc | wc -w`;
      [ "$n" -ne $[$nj*2] ] && \
        echo "Wrong number of MMI accumulators $n versus 2*$nj" && exit 1;
      $cmd $dir/log/den_acc_sum.$x.N.log \
        am-mfa-sum-accs $dir/den_acc.$x.N.acc $dir/den_acc.$x.*.N.acc || exit 1;
      rm $dir/den_acc.$x.*.N.acc
      $cmd $dir/log/num_acc_sum.$x.N.log \
        am-mfa-sum-accs $dir/num_acc.$x.N.acc $dir/num_acc.$x.*.N.acc || exit 1;
      rm $dir/num_acc.$x.*.N.acc
      echo "$x.0 Update N ..."
      $cmd $dir/log/update.$x.5.N.log \
        am-mfa-est-ebw --update-flags="N" $dir/$[$x+1].4.w.mdl $dir/num_acc.$x.N.acc $dir/den_acc.$x.N.acc $dir/$[$x+1].5.N.mdl || exit 1;
      echo "$x. Copy final model ..."    
      cp $dir/$[$x+1].5.N.mdl $dir/$[$x+1].mdl
    else
      echo "$x. Copy final model ..."    
      cp $dir/$[$x+1].4.w.mdl $dir/$[$x+1].mdl
    fi

    rm $dir/$[$x+1].*.mdl
  fi
  cur_mdl=$dir/$[$x+1].mdl


  # Some diagnostics: the objective function progress and auxiliary-function
  # improvement.  Note: this code is same as in train_mmi.sh
  x=$[$x+1]
done

echo "MMI training finished"

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

exit 0;
