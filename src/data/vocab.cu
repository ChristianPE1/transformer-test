#include "vocab.cuh"
#include <algorithm>
#include <iostream>
#include <sstream>

Vocab::Vocab()
{
    // Add special tokens
    word_to_id["<pad>"] = 0;
    word_to_id["<unk>"] = 1;
    word_to_id["<sos>"] = 2;
    word_to_id["<eos>"] = 3;

    id_to_word[0] = "<pad>";
    id_to_word[1] = "<unk>";
    id_to_word[2] = "<sos>";
    id_to_word[3] = "<eos>";
}

void Vocab::buildFromSentences(const std::vector<std::string> &sentences)
{
    std::unordered_map<std::string, int> word_count;

    // Count words
    for (const auto &sentence : sentences)
    {
        std::istringstream iss(sentence);
        std::string word;
        while (iss >> word)
        {
            if (word_to_id.find(word) == word_to_id.end())
            {
                word_count[word]++;
            }
        }
    }

    // Sort by frequency and add to vocabulary
    std::vector<std::pair<std::string, int>> sorted_words(word_count.begin(), word_count.end());
    std::sort(sorted_words.begin(), sorted_words.end(),
              [](const auto &a, const auto &b)
              { return a.second > b.second; });

    int next_id = 4; // Start after special tokens
    for (const auto &[word, count] : sorted_words)
    {
        if (count >= 10 && next_id < 1000)
        { // Only include top 1000 words by frequency
            word_to_id[word] = next_id;
            id_to_word[next_id] = word;
            next_id++;
        }
    }

    std::cout << "Vocabulary built with " << word_to_id.size() << " words" << std::endl;
}

void Vocab::addWord(const std::string &word)
{
    if (word_to_id.find(word) == word_to_id.end())
    {
        int id = word_to_id.size();
        word_to_id[word] = id;
        id_to_word[id] = word;
    }
}

int Vocab::getWordId(const std::string &word) const
{
    auto it = word_to_id.find(word);
    return (it != word_to_id.end()) ? it->second : word_to_id.at("<unk>");
}

std::string Vocab::getWord(int id) const
{
    auto it = id_to_word.find(id);
    return (it != id_to_word.end()) ? it->second : "<unk>";
}

std::vector<int> Vocab::sentenceToIds(const std::string &sentence) const
{
    std::vector<int> ids;
    std::istringstream iss(sentence);
    std::string word;

    while (iss >> word)
    {
        ids.push_back(getWordId(word));
    }

    return ids;
}

std::string Vocab::idsToSentence(const std::vector<int> &ids) const
{
    std::string sentence;
    for (int id : ids)
    {
        std::string word = getWord(id);
        if (word != "<pad>")
        {
            if (!sentence.empty())
                sentence += " ";
            sentence += word;
        }
    }
    return sentence;
}