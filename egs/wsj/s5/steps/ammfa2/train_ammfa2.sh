#!/bin/bash

# Copyright 2013  Wen-Lin Zhang.  Apache 2.0.

# Am-MFA2 training

# Begin configuration section.
nj=4
cmd=run.pl
stage=-5
context_opts= # e.g. set it to "--context-width=5 --central-position=2"  for a quinphone system.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=10   # Total number of iterations
realign_iters="3 6" # Iters to realign on.
inner_iters=1  # Number of inner iterations for update the weight vector
min_comp_change=0.05  # Minimal change of count of weights
beam=8
retry_beam=40
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves

min_comp=10       # minimal mixture count per state
max_comp=-1     # maximal mixture count per state
weight_method=7   # shrink weight option (0-Direct|1-ShrinkHard|2-ShrinkAver|3-ShrinkSoft|4-Floor|5-Factor)
weight_parm=1.0e-5   # weight estimation parameters
gselect_direct=1  # whether use direct gselect method

glasso_tau=0.01   # L1 weight for the graphica lasso method for the sparse inverse covariance matrix estimation

# Example options
# --weight-method 2 --weight-parm 0.1  *****Shrink by 0.1/I
# --weight-method 3 --weight-parm 0.9  *****Shrink by 90% accumulate weights
# --weight-method 4 --weight-parm 0.1  *****Shrink by floor 0.1/I
# --weight-method 5 --weight-parm 0.9  *****Shrink by a factor of 0.9


# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 5 ]; then
  echo "Usage: steps/train_am_mfa2.sh <num-leaves> <data> <lang> <ali-dir> <exp-dir>"
  echo " e.g.: steps/train_am_mfa2.sh 3500 data/train_si84 data/lang \\"
  echo "                      exp/ammfa_ali exp/ammfa2"
  echo "main options (for others, see top of script file)"
  echo "  --config <config-file>                           # config containing options"
  echo "  --cmd (utils/run.pl|utils/queue.pl <queue opts>) # how to run jobs."
  echo "  --fcov  <0|1>                                    # whether use full covariance matrix"
  echo "  --silence-weight <sil-weight>                    # weight for silence (e.g. 0.5 or 0.0)"
  echo "  --num-iters <#iters>                             # Number of iterations of E-M (default: 30)"
  echo "  --weight_method <0-Direct|1-ShrinkHard|2-ShrinkAver|3-ShrinkSoft|4-Floor|5-Factor>"   
  echo "                                                   # method to update weight. (default=2)"
  echo "  --weight_parm <0.1>                              # weight parameter (default=0.1)"
  echo "  --min-comp <10>                                  # minimal mixture count per state (default=10)" 
  echo "  --max-comp <20>                                  # maximal mixture count per state (default=-1) "                          
  exit 1;
fi


num_leaves=$1
data=$2
lang=$3
alidir=$4
dir=$5

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
oov=`cat $lang/oov.int`
silphonelist=`cat $lang/phones/silence.csl`
feat_dim=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/feature dimension/{print $NF}'` || exit 1;
[ $feat_dim -eq $feat_dim ] || exit 1; # make sure it's numeric.
nj=`cat $alidir/num_jobs` || exit 1;
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;

## Set up features.
if [ -f $alidir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) cp $alidir/final.mat $dir   
    feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $dir/final.mat ark:- ark:- |";;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ -f $alidir/trans.1 ]; then
  echo "$0: copy transforms from $alidir"
  cp $alidir/trans.* $dir
  echo "$0: use transforms from $dir"
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$dir/trans.JOB ark:- ark:- |"
else
  echo "No CMLLR transform found!"
fi

if [ $stage -le -4 ]; then
  echo "$0: Copy tree ..."
  cp $alidir/tree $dir/tree
fi

if [ $stage -le -3 ]; then
  echo "$0: Convert the AmMfa model (final.mdl) to AmMfa2 model (0.mdl)"  
  # Convert AmMfa model to AmMfa2 model
  am-mfa-to-am-mfa2 $alidir/final.mdl $dir/0.mdl
fi

if [ $stage -le -2 ]; then
  echo "$0: compiling training graphs"
  text="ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text|"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
    "$text" "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: Ciot alignments" 
  cp $alidir/ali.*.gz $dir
fi

# set up weight estimation option
weight_opt="--min-comp=$min_comp --max-comp=$max_comp --weight_method=$weight_method --weight_parm=$weight_parm"

x=0
shrink_weight=true
previous_comp_count=-1
no_comp_changes=0
while [ $x -lt $num_iters ]; do
   echo "$0: training pass $x *********************************** "
   if [ $stage -le $x ]; then
     if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
       echo "$0: re-aligning data"
       $cmd JOB=1:$nj $dir/log/align.$x.JOB.log  \
         am-mfa2-align-compiled $scale_opts --beam=$beam --retry-beam=$retry_beam \
         $dir/$x.mdl "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
         "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
     fi

     if [ $x -ge 1 ]; then
       # update tw by inner iterations
       cp $dir/$x.mdl $dir/$[$x+1].tw.0.mdl
       inner_cnt=0
       while [ $inner_cnt -lt $inner_iters ]; do
         echo "$0: stage $x.1.$inner_cnt.1 accumulate statistics for update tw"
         $cmd JOB=1:$nj $dir/log/acc.$x.JOB.tw.$inner_cnt.log \
           am-mfa2-acc-stats --ammfa2-gselect-direct=$gselect_direct \
              --update-flags="tw" $dir/$[$x+1].tw.$inner_cnt.mdl "$feats" \
              "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
              $dir/$x.JOB.tw.$inner_cnt.acc || exit 1;
         am-mfa2-sum-accs $dir/$x.tw.$inner_cnt.acc $dir/$x.*.tw.$inner_cnt.acc
         rm $dir/$x.*.tw.$inner_cnt.acc 2>/dev/null
         echo "$0: stage $x.1.$inner_cnt.2 update tw"
         $cmd $dir/log/update.$x.tw.$inner_cnt.log \
           am-mfa2-est --update-flags="tw" $weight_opt  $dir/$[$x+1].tw.$inner_cnt.mdl $dir/$x.tw.$inner_cnt.acc \
           $dir/$[$x+1].tw.$[$inner_cnt+1].mdl || exit 1;
         inner_cnt=$[$inner_cnt+1];
       done
       cp $dir/$[$x+1].tw.$inner_iters.mdl $dir/$[$x+1].0.mdl
       rm $dir/$[$x+1].tw.*.mdl
       comp_count=`am-mfa2-info $dir/$[$x+1].0.mdl 2>/dev/null | awk '/Average # of components per state/{print $NF}'` || exit 1;
       echo "Average # of components per state : $comp_count (previous=$previous_comp_count)"
       previous_comp_count=$comp_count

       echo "$0: stage $x.2 accumulate statistics for update y"
       $cmd JOB=1:$nj $dir/log/acc.$x.JOB.y.log \
         am-mfa2-acc-stats --ammfa2-gselect-direct=$gselect_direct --update-flags="y" $dir/$[$x+1].0.mdl "$feats" \
           "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
           $dir/$x.JOB.y.acc || exit 1;
       am-mfa2-sum-accs $dir/$x.y.acc $dir/$x.*.y.acc
       rm $dir/$x.*.y.acc 2>/dev/null
       echo "$0: stage $x.3 update y"
       $cmd $dir/log/update.$x.y.log \
         am-mfa2-est --update-flags="y" $weight_opt  $dir/$[$x+1].0.mdl $dir/$x.y.acc \
         $dir/$[$x+1].y.mdl || exit 1;

     else
       echo "$0: stage $x.1-3 skip update tw and y"
       cp $dir/$x.mdl $dir/$[$x+1].y.mdl
     fi
     
     echo "$0: stage $x.4 accumulate statistics for update S"
     $cmd JOB=1:$nj $dir/log/acc.$x.JOB.S.log \
       am-mfa2-acc-stats --ammfa2-gselect-direct=$gselect_direct --update-flags="S" $dir/$[$x+1].y.mdl "$feats" \
         "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
         $dir/$x.JOB.S.acc || exit 1;
     am-mfa2-sum-accs $dir/$x.S.acc $dir/$x.*.S.acc
     rm $dir/$x.*.S.acc 2>/dev/null
     echo "$0: stage $x.5 update S"
     $cmd $dir/log/update.$x.S.log \
       am-mfa2-est --update-flags="S" --glasso-tau=$glasso_tau $dir/$[$x+1].y.mdl $dir/$x.S.acc \
       $dir/$[$x+1].S.mdl || exit 1;
     
     mv $dir/$[$x+1].S.mdl $dir/$[$x+1].mdl
     rm $dir/$[$x+1].*.mdl

   fi
   
   rm $dir/*.acc $dir/$x.mdl
   x=$[$x+1];
done

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

utils/summarize_warnings.pl $dir/log

echo Done
