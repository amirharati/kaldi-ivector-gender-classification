amt gender classification using ivector
amt data is orginized as:
~/amt_gender/wav
	    ./female
	    ./male
with wave files under each directory.
usage: bash run.sh

Under kaldi_outputs we generate ivectors which can be used for further ML with other tools.
Both length-normalized and raw ivectors are for utterance and model/speaker are generated.
Also LDA vector are generated. PLDA will be added.

copy: ivector-plda-transform.cc to Kaldi/src/ivectorbin  (and also add it to makefile) and compile the  kaldi.
