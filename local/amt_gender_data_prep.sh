#!/bin/bash

. ./path.sh || exit 1;

sph2pipe=$KALDI_ROOT/tools/sph2pipe_v2.5/sph2pipe

if [ $# != 1 ]; then
  echo "Usage: $0 <audio-path>"
  echo " $0 ~/data/amt_gender/wav"
  exit 1;
fi

aishell_audio_dir=$1

train_dir=data/local/train
test_dir=data/local/test

mkdir -p $train_dir
mkdir -p $test_dir


# find wav audio file for train, dev and test resp.
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/train" > $train_dir/wav.flist || exit 1;
find $aishell_audio_dir -iname "*.wav" | grep -i "wav/test" > $test_dir/wav.flist || exit 1;


# Transcriptions preparation
for dir in $train_dir $test_dir; do
  echo Preparing $dir spk2utt  utt2spk  wav.scp
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{print $NF}' > $dir/utt.list
  sed -e 's/\.wav//' $dir/wav.flist | awk -F '/' '{i=NF-1;printf("%s %s\n",$NF,$i)}' > $dir/utt2spk_all
  paste -d' ' $dir/utt.list $dir/wav.flist > $dir/wav.scp_all
  #awk '{printf("%s '$sph2pipe' -f wav %s |\n", $2, $1);}' < $dir/wav.flist > $dir/wav2.flist
  #paste -d' ' $dir/utt.list $dir/wav2.flist > $dir/wav.scp_all
  
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/utt2spk_all | sort -u > $dir/utt2spk
  utils/filter_scp.pl -f 1 $dir/utt.list $dir/wav.scp_all | sort -u > $dir/wav.scp
  utils/utt2spk_to_spk2utt.pl $dir/utt2spk > $dir/spk2utt
done

mkdir -p data/train data/test
for f in spk2utt utt2spk wav.scp; do
  cp $train_dir/$f data/train/$f || exit 1;
  cp $test_dir/$f data/test/$f || exit 1;
done

echo "$0: AMT_GENDER data preparation succeeded"
exit 0;
