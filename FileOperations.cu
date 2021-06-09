
#include "FileOperations.h"
#include <fstream>

std::string getFilenameFromCommandLine(int argc, char* argv[])
{
    if(argc == 2)
        return std::string(argv[1]);
    else
        throw FileReadException("Please specify filename.");
}

void readFileToGame(GameOfLife& game, std::string filename)
{
    std::ifstream ifs(filename.c_str(), std::ifstream::in);
    if(!ifs.good()) throw FileReadException("File not found, or unable to read file.");

    // Detect file format. If more formats get introduced, 
    // or detection logic gets more complicated, this
    // should be turned into a chain of responsibility.
    if(filename.find(".rle") == filename.length() - 4)
        loadGame<GameRleFileLoader>(game, ifs);
    else
        loadGame<GameRawFileLoader>(game, ifs);
}
