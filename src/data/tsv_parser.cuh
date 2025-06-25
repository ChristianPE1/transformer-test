#ifndef TSV_PARSER_CUH
#define TSV_PARSER_CUH

#include <vector>
#include <string>
#include <utility>

class TSVParser
{
public:
   static void parseFile(const std::string &filename,
                         std::vector<std::pair<std::string, std::string>> &pairs);
   static std::string preprocessSentence(const std::string &sentence);
};

#endif // TSV_PARSER_CUH