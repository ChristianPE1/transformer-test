#ifndef DATASET_CUH
#define DATASET_CUH

#include "vocab.cuh"
#include <vector>
#include <string>
#include <utility>

class Dataset
{
private:
    std::vector<std::pair<std::string, std::string>> sentence_pairs;
    std::vector<std::pair<std::string, std::string>> train_pairs;
    std::vector<std::pair<std::string, std::string>> test_pairs;

    Vocab eng_vocab;
    Vocab spa_vocab;

public:
    void loadTSV(const std::string &tsv_file);
    void buildVocabularies();
    void createTrainTestSplit(float train_ratio = 0.8f);

    std::vector<std::pair<std::vector<int>, std::vector<int>>> getBatch(
        size_t batch_size, bool use_train = true);

    const Vocab &getEngVocab() const { return eng_vocab; }
    const Vocab &getSpaVocab() const { return spa_vocab; }

    size_t getTrainSize() const { return train_pairs.size(); }
    size_t getTestSize() const { return test_pairs.size(); }
};

#endif // DATASET_CUH