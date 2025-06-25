#include "dataset.cuh"
#include "tsv_parser.cuh"
#include <algorithm>
#include <random>
#include <iostream>

void Dataset::loadTSV(const std::string &tsv_file)
{
    TSVParser::parseFile(tsv_file, sentence_pairs);
    std::cout << "Dataset loaded with " << sentence_pairs.size() << " pairs" << std::endl;
}

void Dataset::buildVocabularies()
{
    std::vector<std::string> eng_sentences, spa_sentences;

    for (const auto &[eng, spa] : sentence_pairs)
    {
        eng_sentences.push_back(eng);
        spa_sentences.push_back(spa);
    }

    eng_vocab.buildFromSentences(eng_sentences);
    spa_vocab.buildFromSentences(spa_sentences);

    std::cout << "English vocabulary size: " << eng_vocab.size() << std::endl;
    std::cout << "Spanish vocabulary size: " << spa_vocab.size() << std::endl;
}

void Dataset::createTrainTestSplit(float train_ratio)
{
    // Shuffle the data
    std::random_device rd;
    std::mt19937 g(rd());
    std::shuffle(sentence_pairs.begin(), sentence_pairs.end(), g);

    // Split the data
    size_t train_size = sentence_pairs.size() * train_ratio;
    train_pairs.assign(sentence_pairs.begin(), sentence_pairs.begin() + train_size);
    test_pairs.assign(sentence_pairs.begin() + train_size, sentence_pairs.end());

    std::cout << "Train set: " << train_pairs.size() << " pairs" << std::endl;
    std::cout << "Test set: " << test_pairs.size() << " pairs" << std::endl;
}

std::vector<std::pair<std::vector<int>, std::vector<int>>> Dataset::getBatch(
    size_t batch_size, bool use_train)
{

    std::vector<std::pair<std::vector<int>, std::vector<int>>> batch;
    const auto &pairs = use_train ? train_pairs : test_pairs;

    for (size_t i = 0; i < batch_size && i < pairs.size(); ++i)
    {
        auto eng_ids = eng_vocab.sentenceToIds(pairs[i].first);
        auto spa_ids = spa_vocab.sentenceToIds(pairs[i].second);
        batch.push_back({eng_ids, spa_ids});
    }

    return batch;
}