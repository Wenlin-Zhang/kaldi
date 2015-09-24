#!/bin/bash

# Copyright 2013  Wen-Lin Zhang.  Apache 2.0.

# This script does decoding with an MFA-GMM system, without speaker vectors. 
# If the system was built on top of fMLLR transforms from a conventional system, you should
# provide the --transform-dir option.

# Begin configuration section.
stage=1
alignment_model=
transform_dir=    # dir to find fMLLR transforms.
nj=4 # number of decoding jobs.
acwt=0.1  # Just a default value, used for adaptation and beam-pruning..
scoring_opts=
cmd=run.pl
beam=15.0
max_active=7000
lat_beam=8.0 # Beam we use in lattice generation.
# End configuration section.

echo "$0 $@"  # Print the command line for logging

[ -f ./path.sh ] && . ./path.sh; # source the path.
. parse_options.sh || exit 1;

if [ $# -ne 3 ]; then
  echo "Usage: steps/decode_ammfa.sh [options] <graph-dir> <data-dir> <decode-dir>"
  echo " e.g.: steps/decode_ammfa.sh --transform-dir exp/tri3b/decode_dev93_tgpr \\"
  echo "      exp/ammfa4a/graph_tgpr data/test_dev93 exp/ammfa4a/decode_dev93_tgpr"
  echo "main options (for others, see top of script file)"
  echo "  --transform-dir <decoding-dir>           # directory of previous decoding"
  echo "                                           # where we can find transforms for SAT systems."
  echo "  --alignment-model <ali-mdl>              # Model for the first-pass decoding."
  echo "  --config <config-file>                   # config containing options"
  echo "  --nj <nj>                                # number of parallel jobs"
  echo "  --cmd <cmd>                              # Command to run in parallel with"
  echo "  --beam <beam>                            # Decoding beam; default 13.0"
  exit 1;
fi

graphdir=$1
data=$2
dir=$3
srcdir=`dirname $dir`; # Assume model directory one level up from decoding directory.

for f in $graphdir/HCLG.fst $data/feats.scp $srcdir/final.mdl; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

sdata=$data/split$nj;
silphonelist=`cat $graphdir/phones/silence.csl` || exit 1
splice_opts=`cat $srcdir/splice_opts 2>/dev/null`

mkdir -p $dir/log
[[ -d $sdata && $data/feats.scp -ot $sdata ]] || split_data.sh $data $nj || exit 1;
echo $nj > $dir/num_jobs


## Set up features.
if [ -f $srcdir/final.mat ]; then feat_type=lda; else feat_type=delta; fi
echo "$0: feature type is $feat_type"

case $feat_type in
  delta) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | add-deltas ark:- ark:- |";;
  lda) feats="ark,s,cs:apply-cmvn --norm-vars=false --utt2spk=ark:$sdata/JOB/utt2spk scp:$sdata/JOB/cmvn.scp scp:$sdata/JOB/feats.scp ark:- | splice-feats $splice_opts ark:- ark:- | transform-feats $srcdir/final.mat ark:- ark:- |"
    ;;
  *) echo "$0: invalid feature type $feat_type" && exit 1;
esac
if [ ! -z "$transform_dir" ]; then
  echo "$0: using transforms from $transform_dir"
  [ ! -f $transform_dir/trans.1 ] && echo "$0: no such file $transform_dir/trans.1" && exit 1;
  [ "$nj" -ne "`cat $transform_dir/num_jobs`" ] \
    && echo "$0: #jobs mismatch with transform-dir." && exit 1;
  feats="$feats transform-feats --utt2spk=ark:$sdata/JOB/utt2spk ark,s,cs:$transform_dir/trans.JOB ark:- ark:- |"
elif grep 'transform-feats --utt2spk' $srcdir/log/acc.0.1.log 2>/dev/null; then
  echo "$0: **WARNING**: you seem to be using an SGMM system trained with transforms,"
  echo "  but you are not providing the --transform-dir option in test time."
fi
##

## Work out name of alignment model. ##
if [ -z "$alignment_model" ]; then
  if [ -f "$srcdir/final.alimdl" ]; then alignment_model=$srcdir/final.alimdl;
  else alignment_model=$srcdir/final.mdl; fi
fi
[ ! -f "$alignment_model" ] && echo "$0: no alignment model $alignment_model " && exit 1;

# Generate state-level lattice which we can rescore.  This is done with the 
# alignment model and no speaker-vectors.
if [ $stage -le 2 ]; then
  echo "$0: step1. generate the pre_lat in first pass decoding ..."
  $cmd JOB=1:$nj $dir/log/decode_pass1.JOB.log \
    am-mfa-latgen-faster --max-active=$max_active --beam=$beam --lattice-beam=$lat_beam \
    --acoustic-scale=$acwt --determinize-lattice=false --allow-partial=true \
    --word-symbol-table=$graphdir/words.txt $alignment_model \
    $graphdir/HCLG.fst "$feats" "ark:|gzip -c > $dir/pre_lat.JOB.gz" || exit 1;
fi

## Check if the model has speaker vectors
spkdim=`am-mfa-info $srcdir/final.mdl | grep 'speaker vector' | awk '{print $NF}'`

if [ $spkdim -gt 0 ]; then  ### For models with speaker vectors:

# Estimate speaker vectors (1st pass).  Prune before determinizing
# because determinization can take a while on un-pruned lattices.
# Note: the am-mfa-post-to-gpost stage is necessary because we have
# a separate alignment-model and final model, otherwise we'd skip it 
# and use am-mfa-est-spkvecs.
  if [ $stage -le 3 ]; then
    echo "$0: step2. generate the speaker vectors in first pass decoding ..."
    $cmd JOB=1:$nj $dir/log/vecs_pass1.JOB.log \
      gunzip -c $dir/pre_lat.JOB.gz \| \
      lattice-prune --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      weight-silence-post 0.0 $silphonelist $alignment_model ark:- ark:- \| \
      am-mfa-post-to-gpost $alignment_model "$feats" ark:- ark:- \| \
      am-mfa-est-spkvecs-gpost --spk2utt=ark:$sdata/JOB/spk2utt \
        $srcdir/final.mdl "$feats" ark,s,cs:- "ark:$dir/pre_vecs.JOB" || exit 1;
  fi

# Estimate speaker vectors (2nd pass).  Since we already have spk vectors,
# at this point we need to rescore the lattice to get the correct posteriors.
  if [ $stage -le 4 ]; then
     echo "$0: step3. generate the speaker vectors in second pass decoding ..."
    $cmd JOB=1:$nj $dir/log/vecs_pass2.JOB.log \
      gunzip -c $dir/pre_lat.JOB.gz \| \
      am-mfa-rescore-lattice --spk-vecs=ark:$dir/pre_vecs.JOB --utt2spk=ark:$sdata/JOB/utt2spk \
        $srcdir/final.mdl ark:- "$feats" ark:- \| \
      lattice-prune --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$vecs_beam ark:- ark:- \| \
      lattice-to-post --acoustic-scale=$acwt ark:- ark:- \| \
      weight-silence-post 0.0 $silphonelist $srcdir/final.mdl ark:- ark:- \| \
      am-mfa-est-spkvecs --spk2utt=ark:$sdata/JOB/spk2utt --spk-vecs=ark:$dir/pre_vecs.JOB \
        $srcdir/final.mdl "$feats" ark,s,cs:- "ark:$dir/vecs.JOB" || exit 1;
  fi
  rm $dir/pre_vecs.*

# Now rescore the state-level lattices with the adapted features and the
# corresponding model.  Prune and determinize the lattices to limit
# their size.
  if [ $stage -le 6 ]; then
     echo "$0: step4. rescore the lattices ..."
    $cmd JOB=1:$nj $dir/log/rescore.JOB.log \
      am-mfa-rescore-lattice --utt2spk=ark:$sdata/JOB/utt2spk --spk-vecs=ark:$dir/vecs.JOB \
      $srcdir/final.mdl "ark:gunzip -c $dir/pre_lat.JOB.gz|" "$feats" ark:- \| \
      lattice-determinize-pruned --acoustic-scale=$acwt --beam=$lat_beam ark:- \
      "ark:|gzip -c > $dir/lat.JOB.gz" || exit 1;
  fi
  rm $dir/pre_lat.*.gz
else

  for n in `seq 1 $nj`; do
    mv $dir/pre_lat.${n}.gz $dir/lat.${n}.gz
  done

fi



# The output of this script is the files "lat.*.gz"-- we'll rescore this at 
# different acoustic scales to get the final output.
if [ $stage -le 7 ]; then
  [ ! -x local/score.sh ] && \
    echo "Not scoring because local/score.sh does not exist or not executable." && exit 1;
  echo "score best paths"
  local/score.sh $scoring_opts --cmd "$cmd" $data $graphdir $dir
  echo "score confidence and timing with sclite"
  #local/score_sclite_conf.sh --cmd "$cmd" --language turkish $data $graphdir $dir
fi
echo "Decoding done."
exit 0;
