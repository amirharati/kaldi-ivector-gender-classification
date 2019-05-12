#!/bin/bash
# gender classification  based on : https://github.com/lr2582858/kaldi-timit-sre-ivector/blob/master/v1/run.sh
# Amir Harati, May 2019

. ./cmd.sh
. ./path.sh

guss_num=512
ivector_dim=200
lda_dim=50
nj=2
exp=exp/ivector_gauss${guss_num}_dim${ivector_dim}

set -e # exit on error

####### Bookmark: scp prep #######

bash local/amt_gender_data_prep.sh ~/data/amt_gender/wav

###### Bookmark: MFCC extraction ######
mfccdir=mfcc
for x in train test; do
  steps/make_mfcc.sh --cmd "$train_cmd" --nj $nj data/$x exp/make_mfcc/$x $mfccdir
  sid/compute_vad_decision.sh --nj 2 --cmd "$train_cmd" data/$x exp/make_mfcc/$x $mfccdir
  utils/fix_data_dir.sh data/$x
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

###### Bookmark: i-vector train ######
# train diag ubm
sid/train_diag_ubm.sh --nj $nj --cmd "$train_cmd" --num-threads 16 \
  data/train $guss_num $exp/diag_ubm
#train full ubm
sid/train_full_ubm.sh --nj $nj --cmd "$train_cmd" data/train \
  $exp/diag_ubm $exp/full_ubm

#train ivector
sid/train_ivector_extractor.sh --cmd "$train_cmd"  --nj 2 \
  --ivector-dim $ivector_dim --num-iters 5 $exp/full_ubm/final.ubm data/train \
  $exp/extractor

###### Bookmark: i-vector extraction ######
extract train ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
  $exp/extractor data/train $exp/ivector_train
#extract enroll ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
  $exp/extractor data/train/enroll  $exp/ivector_enroll
#extract eval ivector
sid/extract_ivectors.sh --cmd "$train_cmd" --nj $nj \
  $exp/extractor data/test/eval  $exp/ivector_eval

###### Bookmark: scoring ######

# basic cosine scoring on i-vectors
bash local/cosine_scoring.sh data/train/enroll data/test/eval \
  $exp/ivector_enroll $exp/ivector_eval $trials $exp/scores

# cosine scoring after reducing the i-vector dim with LDA
bash local/lda_scoring.sh data/train data/train/enroll data/test/eval \
  $exp/ivector_train $exp/ivector_enroll $exp/ivector_eval $trials $exp/scores $lda_dim

# cosine scoring after reducing the i-vector dim with PLDA
bash local/plda_scoring.sh data/train data/train/enroll data/test/eval \
  $exp/ivector_train $exp/ivector_enroll $exp/ivector_eval $trials $exp/scores

# print eer
for i in cosine lda plda; do
  eer=`compute-eer <(python local/prepare_for_eer.py $trials $exp/scores/${i}_scores) 2> /dev/null`
  printf "%15s %5.2f \n" "$i eer:" $eer
done > $exp/scores/results.txt

cat $exp/scores/results.txt


#  copy ivector to text file
mkdir outputs
$KALDI_ROOT/src/bin/copy-vector scp:$exp/ivector_enroll/ivector.scp ark,t:- > outputs/enroll_ivec.txt
$KALDI_ROOT/src/bin/copy-vector scp:$exp/ivector_train/ivector.scp ark,t:- > outputs/train_ivec.txt
$KALDI_ROOT/src/bin/copy-vector scp:$exp/ivector_eval/ivector.scp ark,t:- > outputs/eval_ivec.txt

$KALDI_ROOT/src/bin/copy-vector scp:$exp/ivector_enroll/spk_ivector.scp ark,t:- > outputs/norm_spk_enroll_ivec.txt
$KALDI_ROOT/src/bin/copy-vector scp:$exp/ivector_train/spk_ivector.scp ark,t:- > outputs/norm_spk_train_ivec.txt
$KALDI_ROOT/src/bin/copy-vector scp:$exp/ivector_eval/spk_ivector.scp ark,t:- > outputs/norm_spk_eval_ivec.txt

$KALDI_ROOT/src/ivectorbin/ivector-normalize-length scp:$exp/ivector_enroll/ivector.scp ark,t:- > outputs/norm_enroll_ivec.txt
$KALDI_ROOT/src/ivectorbin/ivector-normalize-length scp:$exp/ivector_train/ivector.scp ark,t:- > outputs/norm_train_ivec.txt
$KALDI_ROOT/src/ivectorbin/ivector-normalize-length scp:$exp/ivector_enroll/ivector.scp ark,t:- > outputs/norm_train_ivec.txt


# LDA  vectors
ivector-transform $exp/ivector_train/transform.mat scp:$exp/ivector_enroll/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark,t:- > outputs/lda_spk_enroll_ivec.txt
ivector-transform $exp/ivector_train/transform.mat scp:$exp/ivector_train/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark,t:- > outputs/lda_spk_train_ivec.txt
ivector-transform $exp/ivector_train/transform.mat scp:$exp/ivector_eval/spk_ivector.scp ark:- | ivector-normalize-length ark:- ark,t:- > outputs/lda_spk_eval_ivec.txt

ivector-transform $exp/ivector_train/transform.mat scp:$exp/ivector_enroll/ivector.scp ark:- | ivector-normalize-length ark:- ark,t:- > outputs/lda_enroll_ivec.txt
ivector-transform $exp/ivector_train/transform.mat scp:$exp/ivector_train/ivector.scp ark:- | ivector-normalize-length ark:- ark,t:- > outputs/lda_train_ivec.txt
ivector-transform $exp/ivector_train/transform.mat scp:$exp/ivector_eval/ivector.scp ark:- | ivector-normalize-length ark:- ark,t:- > outputs/lda_eval_ivec.txt

#TODO qadd plda 

exit 0
