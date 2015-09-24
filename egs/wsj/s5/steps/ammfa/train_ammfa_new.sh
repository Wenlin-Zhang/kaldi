#!/bin/bash

# Copyright 2013  Wen-Lin Zhang.  Apache 2.0.

# Am-MFA training

# Begin configuration section.
nj=4
cmd=run.pl
stage=-5
context_opts= # e.g. set it to "--context-width=5 --central-position=2"  for a quinphone system.
scale_opts="--transition-scale=1.0 --acoustic-scale=0.1 --self-loop-scale=0.1"
num_iters=25   # Total number of iterations
num_iters_alimdl=3 # Number of iterations for estimating alignment model.
realign_iters="5 10 15" # Iters to realign on.
#phone_space_iters="3 8 13 18 23 28" # Iters to estimate the phone space
inner_iters=1  # Number of inner iterations for update the weight vector
min_comp_change=0.05  # Minimal change of count of weights
spk_dim=0    # You can use this to set the speaker subspace dim. [default: 0]
min_iter_for_spk_update=3 # Iters to update speaker subspace
update_M_iter=4 # min iters to update phone subspace
update_y_iter=0 # min iters to update phone corrdinate
update_w_iter=1 # min iters to update phone weight
power=0.25 # Exponent for number of gaussians according to occurrence counts
beam=8
retry_beam=40
cluster_thresh=-1  # for build-tree control final bottom-up clustering of leaves

min_comp=10       # minimal mixture count per state
max_comp=-1.0     # maximal mixture count per state
weight_method=2   # shrink weight option (0-Direct|1-ShrinkHard|2-ShrinkAver|3-ShrinkSoft|4-Floor|5-Factor)
weight_parm=0.1   # weight estimation parameters
gselect_direct=0  # whether use direct gselect method

# Example options
# --weight-method 2 --weight-parm 0.1  *****Shrink by 0.1/I
# --weight-method 3 --weight-parm 0.9  *****Shrink by 90% accumulate weights
# --weight-method 4 --weight-parm 0.1  *****Shrink by floor 0.1/I
# --weight-method 5 --weight-parm 0.9  *****Shrink by a factor of 0.9

use_l1=0   # whether use L1 regularization
fcov=1     # whether use full covariance

# End configuration section.

echo "$0 $@"  # Print the command line for logging

if [ -f path.sh ]; then . ./path.sh; fi
. parse_options.sh || exit 1;


if [ $# != 6 ]; then
  echo "Usage: steps/train_am_mfa.sh <num-leaves> <data> <lang> <ali-dir> <mfa> <exp-dir>"
  echo " e.g.: steps/train_am_mfa.sh 3500 32 data/train_si84 data/lang \\"
  echo "                      exp/tri3b_ali_si84 exp/mfa4a/final.mfa exp/ammfa4a"
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
mfa=$5
dir=$6

# Check some files.
for f in $data/feats.scp $lang/L.fst $alidir/ali.1.gz $alidir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done


# Set some variables.
oov=`cat $lang/oov.int`
silphonelist=`cat $lang/phones/silence.csl`
feat_dim=`gmm-info $alidir/final.mdl 2>/dev/null | awk '/feature dimension/{print $NF}'` || exit 1;
[ $feat_dim -eq $feat_dim ] || exit 1; # make sure it's numeric.
[ -z $spk_dim ] && spk_dim=$feat_dim
nj=`cat $alidir/num_jobs` || exit 1;
ciphonelist=`cat $lang/phones/context_indep.csl` || exit 1;

mkdir -p $dir/log
echo $nj > $dir/num_jobs
sdata=$data/split$nj;
splice_opts=`cat $alidir/splice_opts 2>/dev/null`
cp $alidir/splice_opts $dir 2>/dev/null
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
spkvecs_opt=  # Empty option for now, until we estimate the speaker vectors.

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
fi

if [ $stage -le -5 ]; then
  echo "$0: accumulating tree stats"
  $cmd JOB=1:$nj $dir/log/acc_tree.JOB.log \
    acc-tree-stats  --ci-phones=$ciphonelist $alidir/final.mdl "$feats" \
    "ark:gunzip -c $alidir/ali.JOB.gz|" $dir/JOB.treeacc || exit 1;
  [ "`ls $dir/*.treeacc | wc -w`" -ne "$nj" ] && echo "$0: Wrong #tree-stats" && exit 1;
  sum-tree-stats $dir/treeacc $dir/*.treeacc 2>$dir/log/sum_tree_acc.log || exit 1;
  rm $dir/*.treeacc
fi

if [ $stage -le -4 ]; then
  echo "$0: Getting questions for tree clustering."
  # preparing questions, roots file...
  cluster-phones $dir/treeacc $lang/phones/sets.int $dir/questions.int 2> $dir/log/questions.log || exit 1;
  cat $lang/phones/extra_questions.int >> $dir/questions.int
  compile-questions $lang/topo $dir/questions.int $dir/questions.qst 2>$dir/log/compile_questions.log || exit 1;

  echo "$0: Building the tree"
  $cmd $dir/log/build_tree.log \
    build-tree --verbose=1 --max-leaves=$num_leaves \
    --cluster-thresh=$cluster_thresh $dir/treeacc $lang/phones/roots.int \
    $dir/questions.qst $lang/topo $dir/tree || exit 1;
fi

if [ $stage -le -3 ]; then
  echo "$0: Initializing the model"  
  $cmd $dir/log/init_ammfa.log \
    am-mfa-init --spk-space-dim=$spk_dim $lang/topo $dir/tree $mfa $dir/init.mdl || exit 1;
  if [ $fcov -eq 1 ]; then
    echo "$0: Convering to full covariance model"
    $cmd $dir/log/convert_full_cov.log \
      am-mfa-convert-cov-type --type="full" $dir/init.mdl $dir/0.mdl
  else
    ln -s $dir/init.mdl $dir/0.mdl
  fi
fi

if [ $stage -le -2 ]; then
  echo "$0: compiling training graphs"
  text="ark:sym2int.pl --map-oov $oov -f 2- $lang/words.txt < $sdata/JOB/text|"
  $cmd JOB=1:$nj $dir/log/compile_graphs.JOB.log \
    compile-train-graphs $dir/tree $dir/0.mdl  $lang/L.fst  \
    "$text" "ark:|gzip -c >$dir/fsts.JOB.gz" || exit 1;
fi

if [ $stage -le -1 ]; then
  echo "$0: Converting alignments" 
  $cmd JOB=1:$nj $dir/log/convert_ali.JOB.log \
    convert-ali $alidir/final.mdl $dir/0.mdl $dir/tree "ark:gunzip -c $alidir/ali.JOB.gz|" \
    "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
fi

# set up weight estimation option
weight_opt="--min-comp=$min_comp --max-comp=$max_comp --weight_method=$weight_method --weight_parm=$weight_parm"

x=0
init_spk_proj=0
shrink_weight=true
previous_comp_count=-1
no_comp_changes=0
while [ $x -lt $num_iters ]; do
   echo "$0: training pass $x *********************************** "
   if echo $realign_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
     echo "$0: re-aligning data"
     $cmd JOB=1:$nj $dir/log/align.$x.JOB.log  \
       am-mfa-align-compiled $spkvecs_opt --utt2spk=ark:$sdata/JOB/utt2spk $scale_opts --beam=$beam --retry-beam=$retry_beam \
       $dir/$x.mdl "ark:gunzip -c $dir/fsts.JOB.gz|" "$feats" \
       "ark:|gzip -c >$dir/ali.JOB.gz" || exit 1;
   fi
   
   if [ $spk_dim -gt 0 -a $x -ge $min_iter_for_spk_update ]; then
     if [ $stage -le $x ]; then
       if [ $init_spk_proj -le 0 ]; then
         echo "$0: init the speaker space ... "
         am-mfa-init-spk-proj $dir/$x.mdl $spk_dim $dir/$x.N.mdl
         rm $dir/$x.mdl
         mv $dir/$x.N.mdl $dir/$x.mdl
         init_spk_proj=1
       fi
       echo "$0: re-estimate the speaker vectors ..."
       $cmd JOB=1:$nj $dir/log/spkvecs.$x.JOB.log \
         ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
         weight-silence-post 0.01 $silphonelist $dir/$x.mdl ark:- ark:- \| \
         am-mfa-est-spkvecs --spk2utt=ark:$sdata/JOB/spk2utt $spkvecs_opt $dir/$x.mdl "$feats" ark,s,cs:- \
           ark:$dir/tmp_vecs.JOB '&&' mv $dir/tmp_vecs.JOB $dir/vecs.JOB || exit 1;
     fi
     spkvecs_opt="--spk-vecs=ark:$dir/vecs.JOB"
   fi

   if [ $stage -le $x ]; then

     # update w
     if [ $x -ge $update_w_iter ]; then
       cp $dir/$x.mdl $dir/$[$x+1].w.0.mdl
       inner_cnt=0
       while [ $inner_cnt -lt $inner_iters ]; do
         echo "$0: stage $x.1.$inner_cnt.1 accumulate statistics for update w"
         $cmd JOB=1:$nj $dir/log/acc.$x.JOB.w.$inner_cnt.log \
           am-mfa-acc-stats --ammfa-gselect-direct=$gselect_direct --utt2spk=ark:$sdata/JOB/utt2spk $spkvecs_opt \
              --update-flags="w" $dir/$[$x+1].w.$inner_cnt.mdl "$feats" \
              "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
              $dir/$x.JOB.w.$inner_cnt.acc || exit 1;
         am-mfa-sum-accs $dir/$x.w.$inner_cnt.acc $dir/$x.*.w.$inner_cnt.acc
         echo "$0: stage $x.1.$inner_cnt.2 update w"
         $cmd $dir/log/update.$x.w.$inner_cnt.log \
           am-mfa-est --update-flags="w" --use-l1=$use_l1 $weight_opt  $dir/$[$x+1].w.$inner_cnt.mdl $dir/$x.w.$inner_cnt.acc \
           $dir/$[$x+1].w.$[$inner_cnt+1].mdl || exit 1;
         inner_cnt=$[$inner_cnt+1];
         rm $dir/$x.*.acc 2>/dev/null
       done
       cp $dir/$[$x+1].w.$inner_iters.mdl $dir/$[$x+1].0.mdl
       rm $dir/$[$x+1].w.*.mdl
       comp_count=`am-mfa-info $dir/$[$x+1].0.mdl 2>/dev/null | awk '/Average # of components per state/{print $NF}'` || exit 1;
       echo "Average # of components per state : $comp_count (previous=$previous_comp_count)"
       #if [ $(echo "$previous_comp_count > 0" | bc) -eq 1 ]; then
       #   if [ $(echo "$previous_comp_count-$comp_count < $min_comp_change" | bc) -eq 1 ]; then
       #      no_comp_changes=$[$no_comp_changes+1]
       #      echo "Detect $no_comp_changes times that the component change is less than $min_comp_change."
       #   else
       #      echo "The component change is larger than $min_comp_change, reset the component change counter to 0."
       #      no_comp_changes=0
       #   fi
       #   if [ $(echo "$no_comp_changes >= 3" | bc) -eq 1 ]; then
       #      echo "Detect 3 times that the component change is less than $min_comp_change, no weight shrinkage from now on."
       #      weight_opt="--min-comp=$min_comp --max-comp=$max_comp --weight_method=0"
       #      if [ $(echo "$num_iters > $[$x+3]" | bc) -eq 1 ]; then
       #        num_iters=$[$x+3];
       #        echo "Set the total iteration number to $num_iters";
       #      fi
       #   fi
       #fi
       previous_comp_count=$comp_count
     else
       echo "$0: stage $x.1 skip update w"
       cp $dir/$x.mdl $dir/$[$x+1].0.mdl
     fi

     # set up other update flags
     if [ $x -ge $update_y_iter ]; then
       update_flags="y"
     fi
     if [ $x -ge $update_M_iter ]; then
       update_flags="${update_flags}M"
     else
       update_flags="${update_flags}" # no M on early iters, if --update-m-iter option given.
     fi
     update_flags="${update_flags}mSt"

     #if echo $phone_space_iters | grep -w $x >/dev/null && [ $stage -le $x ]; then
     #  echo "$0: stage $x estimate the phone space parameters"
     #  update_flags="MmS"
     #else
     #  echo "$0: stage $x estimate the state-dependent parameters"
     #  update_flags="ywt"
     #fi

     echo "$0: stage $x.2 accumulate statistics for update non-speaker-dependent parameters (update-flags = $update_flags)"
     $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
       am-mfa-acc-stats --ammfa-gselect-direct=$gselect_direct --utt2spk=ark:$sdata/JOB/utt2spk $spkvecs_opt --update-flags=$update_flags $dir/$[$x+1].0.mdl "$feats" \
         "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
         $dir/$x.JOB.acc || exit 1;
     am-mfa-sum-accs $dir/$x.acc $dir/$x.*.acc
     rm $dir/$x.*.acc 2>/dev/null
     echo "split 1 -- " && grep 'Overall like per frame' $dir/log/acc.$x.1.log
     echo "$0: stage $x.3 update non-speaker-dependent parameters"
     $cmd $dir/log/update.$x.log \
       am-mfa-est --update-flags=$update_flags --use-l1=$use_l1 $weight_opt  $dir/$[$x+1].0.mdl $dir/$x.acc \
       $dir/$[$x+1].1.mdl || exit 1;
     
     if [ $spk_dim -gt 0 -a $x -ge $min_iter_for_spk_update ]; then # update the speaker subspace
       echo "$0: stage $x.3 accumulate for the speaker subspace ..."
       if [ $stage -le $x ]; then
         echo "$0: accumulate statistics for speaker subspace ..."
         $cmd JOB=1:$nj $dir/log/acc.$x.JOB.N.log \
           am-mfa-acc-stats --ammfa-gselect-direct=$gselect_direct --utt2spk=ark:$sdata/JOB/utt2spk $spkvecs_opt --update-flags="N" $dir/$[$x+1].1.mdl "$feats" \
            "ark,s,cs:gunzip -c $dir/ali.JOB.gz | ali-to-post ark:- ark:-|" \
            $dir/$x.JOB.N.acc || exit 1;
         am-mfa-sum-accs $dir/$x.N.acc $dir/$x.*.N.acc
         rm $dir/$x.*.N.acc 2>/dev/null
         echo "$0: stage $x.4 update the speaker subspace ..."
         $cmd $dir/log/update.$x.N.log \
           am-mfa-est --update-flags="N" --use-l1=$use_l1 $weight_opt  $dir/$[$x+1].1.mdl $dir/$x.N.acc \
           $dir/$[$x+1].N.mdl || exit 1;
         mv $dir/$[$x+1].N.mdl $dir/$[$x+1].mdl
         rm $dir/$[$x+1].*.mdl
       fi
     else
       mv $dir/$[$x+1].1.mdl $dir/$[$x+1].mdl
       rm $dir/$[$x+1].*.mdl
     fi
   fi
   
   rm $dir/*.acc
   #rm $dir/$x.mdl
   x=$[$x+1];
done

rm $dir/final.mdl 2>/dev/null
ln -s $x.mdl $dir/final.mdl

if [ $spk_dim -gt 0 ]; then
  # We need to create an "alignment model" that's been trained
  # without the speaker vectors, to do the first-pass decoding with.
  # in test time.

  # We do this for a few iters, in this recipe.
  final_mdl=$dir/$x.mdl
  cur_alimdl=$dir/$x.mdl
 
  echo "$0: get gpost from last alignments ..."
  $cmd JOB=1:$nj $dir/log/ali.gpost.JOB.log \
     ali-to-post "ark:gunzip -c $dir/ali.JOB.gz|" ark:- \| \
     am-mfa-post-to-gpost --utt2spk=ark:$sdata/JOB/utt2spk $spkvecs_opt $cur_alimdl \
        "$feats" ark,s,cs:- "ark:|gzip -c >$dir/ali.gpost.JOB.gz"

  while [ $x -lt $[$num_iters+$num_iters_alimdl] ]; do
    echo "$0: building alignment model (pass $x)"
    if [ $stage -le $x ]; then
      if [ $x -eq $num_iters ]; then
        update_flags="MmS"
        echo "(pass $x) accumulate and update the phone projections, mean and covariance ..."
      else
        update_flags="yMmS"
        echo "(pass $x) accumulate and update the phone vectors, phone projections, mean and covariance..."
      fi
        
      $cmd JOB=1:$nj $dir/log/acc.$x.JOB.log \
        am-mfa-acc-stats-gpost --update-flags=$update_flags $cur_alimdl "$feats" \
          "ark,s,cs:gunzip -c $dir/ali.gpost.JOB.gz |" \
          $dir/$x.JOB.aliacc || exit 1;
      am-mfa-sum-accs $dir/$x.aliacc $dir/$x.*.aliacc
      rm $dir/$x.*.aliacc 2>/dev/null
      $cmd $dir/log/update_ali.$x.log \
        am-mfa-est --update-flags=$update_flags --remove-speaker-space=true --use-l1=$use_l1 $weight_opt $cur_alimdl \
          $dir/$x.aliacc $dir/$[$x+1].alimdl || exit 1;

      [ $x -gt $num_iters ]  && rm $dir/$x.alimdl
    fi

    cur_alimdl=$dir/$[$x+1].alimdl
    x=$[$x+1]
  done
  rm $dir/final.alimdl 2>/dev/null 
  ln -s $x.alimdl $dir/final.alimdl
fi

utils/summarize_warnings.pl $dir/log

echo Done
