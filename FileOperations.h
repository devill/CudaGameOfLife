
#ifndef FILE_OPERATIONS_H
#define FILE_OPERATIONS_H

#include "GameOfLife.h"
#include "GameFileLoader.h"
#include "FileReadException.hpp"

#include <string>

std::string getFilenameFromCommandLine(int argc, char* argv[]);

void readFileToGame(GameOfLife& game, std::string filename);

#endif
