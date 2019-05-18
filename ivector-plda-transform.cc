// ivectorbin/ivector-plda-transform.cc
//  Amir Harati May 2019 based on  ivector-plda-scoring.cc


// Copyright 2013  Daniel Povey

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include "base/kaldi-common.h"
#include "util/common-utils.h"
#include "ivector/plda.h"

#include <ostream>

int main(int argc, char *argv[]) {
  using namespace kaldi;
  typedef kaldi::int32 int32;
  typedef std::string string;
  try {
    const char *usage =
        "Usage: ivector-plda-transform <plda> <train-ivector-rspecifier> <test-ivector-rspecifier>\n"
        " <output_file>\n"
        "\n"
        "e.g.: ivector-plda-trasnform --num-utts=ark:exp/train/num_utts.ark plda "
        "ark:exp/train/spk_ivectors.ark ark:exp/test/ivectors.ark output_file\n"
        "See also: ivector-compute-dot-products, ivector-compute-plda\n";

    ParseOptions po(usage);

    std::string num_utts_rspecifier;

    PldaConfig plda_config;
    plda_config.Register(&po);
    po.Register("num-utts", &num_utts_rspecifier, "Table to read the number of "
                "utterances per speaker, e.g. ark:num_utts.ark\n");

    po.Read(argc, argv);

    if (po.NumArgs() != 4) {
      po.PrintUsage();
      exit(1);
    }

    std::string plda_rxfilename = po.GetArg(1),
        train_ivector_rspecifier = po.GetArg(2),
        test_ivector_rspecifier = po.GetArg(3),
        output_transforms = po.GetArg(4); 

    std::ofstream myfile;
    //std::cout<<output_transforms<<"****\n";
    myfile.open(output_transforms);

    //  diagnostics:
    double tot_test_renorm_scale = 0.0, tot_train_renorm_scale = 0.0;
    int64 num_train_ivectors = 0, num_train_errs = 0, num_test_ivectors = 0;

    //int64 num_trials_done = 0, num_trials_err = 0;

    Plda plda;
    ReadKaldiObject(plda_rxfilename, &plda);

    int32 dim = plda.Dim();

    SequentialBaseFloatVectorReader train_ivector_reader(train_ivector_rspecifier);
    SequentialBaseFloatVectorReader test_ivector_reader(test_ivector_rspecifier);
    RandomAccessInt32Reader num_utts_reader(num_utts_rspecifier);

    typedef unordered_map<string, Vector<BaseFloat>*, StringHasher> HashType;

    // These hashes will contain the iVectors in the PLDA subspace
    // (that makes the within-class variance unit and diagonalizes the
    // between-class covariance).  They will also possibly be length-normalized,
    // depending on the config.
    HashType train_ivectors, test_ivectors;

    KALDI_LOG << "Reading train iVectors";
    for (; !train_ivector_reader.Done(); train_ivector_reader.Next()) {
      std::string spk = train_ivector_reader.Key();
      if (train_ivectors.count(spk) != 0) {
        KALDI_ERR << "Duplicate training iVector found for speaker " << spk;
      }
      const Vector<BaseFloat> &ivector = train_ivector_reader.Value();
      int32 num_examples;
      if (!num_utts_rspecifier.empty()) {
        if (!num_utts_reader.HasKey(spk)) {
          KALDI_WARN << "Number of utterances not given for speaker " << spk;
          num_train_errs++;
          continue;
        }
        num_examples = num_utts_reader.Value(spk);
      } else {
        num_examples = 1;
      }
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      tot_train_renorm_scale += plda.TransformIvector(plda_config, ivector,
                                                      num_examples,
                                                      transformed_ivector);
      train_ivectors[spk] = transformed_ivector;
      num_train_ivectors++;
    }
    KALDI_LOG << "Read " << num_train_ivectors << " training iVectors, "
              << "errors on " << num_train_errs;
    if (num_train_ivectors == 0)
      KALDI_ERR << "No training iVectors present.";
    KALDI_LOG << "Average renormalization scale on training iVectors was "
              << (tot_train_renorm_scale / num_train_ivectors);

    KALDI_LOG << "Reading test iVectors";
    for (; !test_ivector_reader.Done(); test_ivector_reader.Next()) {
      std::string utt = test_ivector_reader.Key();
      if (test_ivectors.count(utt) != 0) {
        KALDI_ERR << "Duplicate test iVector found for utterance " << utt;
      }
      const Vector<BaseFloat> &ivector = test_ivector_reader.Value();
      int32 num_examples = 1; // this value is always used for test (affects the
                              // length normalization in the TransformIvector
                              // function).
      Vector<BaseFloat> *transformed_ivector = new Vector<BaseFloat>(dim);

      tot_test_renorm_scale += plda.TransformIvector(plda_config, ivector,
                                                     num_examples,
                                                     transformed_ivector);
      test_ivectors[utt] = transformed_ivector;
      num_test_ivectors++;
  
      myfile << utt  << *transformed_ivector;
    }
    KALDI_LOG << "Read " << num_test_ivectors << " test iVectors.";
    if (num_test_ivectors == 0)
      KALDI_ERR << "No test iVectors present.";
    KALDI_LOG << "Average renormalization scale on test iVectors was "
              << (tot_test_renorm_scale / num_test_ivectors);

    
  } catch(const std::exception &e) {
    std::cerr << e.what();
    return -1;
  }
  
}
