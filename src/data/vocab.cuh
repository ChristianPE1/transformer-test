#ifndef VOCAB_CUH
#define VOCAB_CUH

#include <unordered_map>
#include <vector>
#include <string>

class Vocab
{
private:
    std::unordered_map<std::string, int> word_to_id;
    std::unordered_map<int, std::string> id_to_word;

public:
    Vocab();

    void buildFromSentences(const std::vector<std::string> &sentences);
    void addWord(const std::string &word);

    int getWordId(const std::string &word) const;
    std::string getWord(int id) const;

    std::vector<int> sentenceToIds(const std::string &sentence) const;
    std::string idsToSentence(const std::vector<int> &ids) const;

    size_t size() const { return word_to_id.size(); }
};

#endif // VOCAB_CUH