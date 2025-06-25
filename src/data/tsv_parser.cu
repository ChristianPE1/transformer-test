#include "tsv_parser.cuh"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <cctype>
#include <locale>
#include <codecvt>

void TSVParser::parseFile(const std::string &filename,
                          std::vector<std::pair<std::string, std::string>> &pairs)
{
   std::ifstream file(filename);
   std::string line;

   if (!file.is_open())
   {
      std::cerr << "Error: Could not open file " << filename << std::endl;
      return;
   }
   // para resolver codificaion utf-8
   file.imbue(std::locale(std::locale(), new std::codecvt_utf8<wchar_t>));

   while (std::getline(file, line))
   {
      std::istringstream iss(line);
      std::string id1, english, id2, spanish;

      // Parse: id1 \t english \t id2 \t spanish
      if (std::getline(iss, id1, '\t') &&
          std::getline(iss, english, '\t') &&
          std::getline(iss, id2, '\t') &&
          std::getline(iss, spanish))
      {

         // Preprocess both sentences
         std::string proc_eng = preprocessSentence(english);
         std::string proc_spa = preprocessSentence(spanish);

         pairs.push_back({proc_eng, proc_spa});
      }
   }

   std::cout << "Loaded " << pairs.size() << " sentence pairs from " << filename << std::endl;
}

std::string TSVParser::preprocessSentence(const std::string &sentence)
{
   std::string processed = sentence;

   // Convert to lowercase
   std::transform(processed.begin(), processed.end(), processed.begin(), ::tolower);

   // Remove special characters (keep only letters and spaces)
   std::string clean;
   for (char c : processed)
   {
      if (std::isalpha(c) || std::isspace(c))
      {
         clean += c;
      }
      else
      {
         clean += ' ';
      }
   }

   // Remove multiple spaces
   std::string result;
   bool prev_space = false;
   for (char c : clean)
   {
      if (c == ' ')
      {
         if (!prev_space)
         {
            result += c;
         }
         prev_space = true;
      }
      else
      {
         result += c;
         prev_space = false;
      }
   }

   // Trim and add special tokens
   // Remove leading and trailing spaces
   size_t start = result.find_first_not_of(' ');
   size_t end = result.find_last_not_of(' ');
   if (start != std::string::npos)
   {
      result = result.substr(start, end - start + 1);
   }
   else
   {
      result = "";
   }

   return "<sos> " + result + " <eos>";
}