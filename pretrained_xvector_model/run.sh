#!/bin/bash
# gender classification  based on : https://github.com/lr2582858/kaldi-timit-sre-ivector/blob/master/v1/run.sh
                                    https://github.com/zeroQiaoba/ivector-xvector/blob/master/xvector/enroll.sh
# Amir Harati, May 2019

. ./cmd.sh
. ./path.sh

#guss_num=512
#ivector_dim=200
lda_dim=50
nj=2
#exp=exp/ivector_gauss${guss_num}_dim${ivector_dim}
exp=xvectors

set -e # exit on error

####### Bookmark: scp prep #######

datadir=`pwd`/data/
logdir=`pwd`/data/log
featdir=mfcc
nnet_dir=`pwd`/exp/xvector_nnet_1a

bash local/amt_gender_data_prep.sh ~/data/amt_gender/wav

echo ==========================================
echo "FeatureForSpeaker start on" `date`
echo ========================================== 
# Extract speaker features MFCC.
for x in train test; do
    steps/make_mfcc.sh --mfcc-config conf/mfcc.conf --nj 8 --cmd "$train_cmd" \
    $datadir/$x $logdir/make_enrollmfcc $featdir
    echo ==========================================
    echo "generate vad file in data/train, VAD start on" `date`
    echo ==========================================
    # Compute VAD decisions. These will be shared across both sets of features.
    sid/compute_vad_decision.sh --nj $nj --cmd "$train_cmd" \
    $datadir/$x $logdir/make_enrollvad $featdir
    utils/fix_data_dir.sh $datadir/$x
    echo ==== FeatureForSpeaker test successfully `date` ===========
done


###### Bookmark: split the test to enroll and eval ######
mkdir -p data/test/enroll data/test/eval
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/enroll
cp data/test/{spk2utt,feats.scp,vad.scp} data/test/eval
python local/split_data_enroll_eval.py data/test/utt2spk  data/test/enroll/utt2spk  data/test/eval/utt2spk  0
trials=data/test/test.trials
python local/produce_trials.py data/test/eval/utt2spk $trials
#utils/fix_data_dir.sh data/test/enroll
utils/fix_data_dir.sh data/test/eval

# create enril for train
mkdir -p data/train/enroll  data/train/eval
cp data/train/{spk2utt,feats.scp,vad.scp} data/train/enroll
python local/split_data_enroll_eval.py data/train/utt2spk  data/train/enroll/utt2spk  data/train/eval/utt2spk 2000
utils/fix_data_dir.sh data/train/enroll


echo ==========================================
echo "EXTRACT start on" `date`
echo ==========================================
# Extract the xVectors
sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd" --nj 2 \
$nnet_dir $datadir/train $exp/xvector_train
sid/nnet3/xvector/extract_xvectors.sh --cmd "$train_cmd" --nj 2 \
$nnet_dir $datadir/test $exp/xvector_test


echo ========= EXTRACT just for testing `date`=============

###### Bookmark: scoring ######
trials=data/test/test.trials
# basic cosine scoring on i-vectors
bash local/cosine_scoring.sh data/train/enroll data/test/eval \
    $exp/xvector_train $exp/xvector_test $trials $exp/scores

# cosine scoring after reducing the i-vector dim with LDA
bash local/lda_scoring.sh data/train data/train/enroll data/test/eval \
  $exp/xvector_train $exp/xvector_train $exp/xvector_test $trials $exp/scores $lda_dim

# cosine scoring after reducing the i-vector dim with PLDA
bash local/plda_scoring.sh data/train data/train/enroll data/test/eval \
  $exp/xvector_train $exp/xvector_train $exp/xvector_test $trials $exp/scores

# print eer
for i in cosine lda plda; do
  eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
  printf "%15s %5.2f \n" "$i eer:" $eer
done > $exp/scores/results.txt

cat $exp/scores/results.txt

#  copy ivector to text file
mkdir -p kaldi_outputs

$KALDI_ROOT/src/bin/copy-vector scp:$exp/xvector_train/xvector.scp ark,t:- > kaldi_outputs/train_ivec.txt
$KALDI_ROOT/src/bin/copy-vector scp:$exp/xvector_test/xvector.scp ark,t:- > kaldi_outputs/test_ivec.txt


$KALDI_ROOT/src/bin/copy-vector scp:$exp/xvector_train/spk_xvector.scp ark,t:- > kaldi_outputs/norm_spk_train_ivec.txt
$KALDI_ROOT/src/bin/copy-vector scp:$exp/xvector_test/spk_xvector.scp ark,t:- > kaldi_outputs/norm_spk_test_ivec.txt

$KALDI_ROOT/src/ivectorbin/ivector-normalize-length scp:$exp/xvector_train/xvector.scp ark,t:- > kaldi_outputs/norm_train_ivec.txt
$KALDI_ROOT/src/ivectorbin/ivector-normalize-length scp:$exp/xvector_test/xvector.scp ark,t:- > kaldi_outputs/norm_test_ivec.txt


# LDA  vectors
ivector-transform $exp/xvector_train/transform.mat scp:$exp/xvector_train/spk_xvector.scp ark:- | ivector-normalize-length ark:- ark,t:- > kaldi_outputs/lda_spk_train_ivec.txt
ivector-transform $exp/xvector_train/transform.mat scp:$exp/xvector_test/spk_xvector.scp ark:- | ivector-normalize-length ark:- ark,t:- > kaldi_outputs/lda_spk_test_ivec.txt

ivector-transform $exp/xvector_train/transform.mat scp:$exp/xvector_train/xvector.scp ark:- | ivector-normalize-length ark:- ark,t:- > kaldi_outputs/lda_train_ivec.txt
ivector-transform $exp/xvector_train/transform.mat scp:$exp/xvector_test/xvector.scp ark:- | ivector-normalize-length ark:- ark,t:- > kaldi_outputs/lda_test_ivec.txt

#PLDA (using custom code)

run.pl logs_plda_trans.log \
  ivector-plda-transform --normalize-length=true \
    --simple-length-normalization=true\
    --num-utts=ark:$exp/xvector_train/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/xvector_train/plda - |" \
    "ark:ivector-subtract-global-mean $exp/xvector_train/mean.vec scp:$exp/xvector_train/spk_xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$exp/xvector_train/xvector.scp ark:- | ivector-subtract-global-mean $exp/xvector_train/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ./kaldi_outputs/plda_train_ivec.txt|| exit 1;

run.pl logs_plda_trans.log \
  ivector-plda-transform --normalize-length=true \
    --simple-length-normalization=true\
    --num-utts=ark:$exp/xvector_train/num_utts.ark \
    "ivector-copy-plda --smoothing=0.0 $exp/xvector_train/plda - |" \
    "ark:ivector-subtract-global-mean $exp/xvector_train/mean.vec scp:$exp/xvector_train/spk_xvector.scp ark:- | ivector-normalize-length ark:- ark:- |" \
    "ark:ivector-normalize-length scp:$exp/xvector_test/xvector.scp ark:- | ivector-subtract-global-mean $exp/xvector_train/mean.vec ark:- ark:- | ivector-normalize-length ark:- ark:- |" \
    ./kaldi_outputs/plda_test_ivec.txt|| exit 1;

exit 0