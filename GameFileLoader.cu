
#include "GameOfLife.h"
#include "GameFileLoader.h"
#include "FileReadException.hpp"

void GameFileLoader::load()
{
    game.clear();
    readDimensions();
    calculateDimensionShifts();
    loadData();
}

void GameFileLoader::calculateDimensionShifts()
{
    xShift = int((game.getWidth() - dataWidth)/2);
    yShift = int((game.getHeight() - dataHeight)/2);
}

void GameRawFileLoader::loadData()
{
    for(int j = 0; j < dataHeight; ++j)
    {
        std::string line;
        stream >> line;
        for(int i = 0; i < dataWidth; ++i)
        {
            if(line[i] == '*') 
                game.createCell(xShift + i , yShift + j);
            else
                game.killCell(xShift + i, yShift + j);
        }
    }
}

void GameRawFileLoader::readDimensions()
{
    stream >> dataWidth >> dataHeight;
}

void GameRleFileLoader::loadData()
{
    char nextChar;
    int x = xShift;
    int y = yShift;

    int repeat = 1;

    while(true)
    {
        nextChar = stream.peek();
        if(!stream.good()) throw FileReadException("Unexpected end of file");
        
        if(nextChar >= '0' && nextChar <= '9')
        {
            repeat = readIntFromStream();
        }
        else 
        {
            stream.get(nextChar);
            switch(nextChar)
            {
                case '!': // End of file marker
                    return;
                case 'b': // Dead cell marker
                    x+=repeat;
                    repeat = 1;
                    break;
                case 'o': // Living cell marker
                    for(; repeat > 0; --repeat)
                    {
                        game.createCell(x,y);
                        ++x;
                    }
                    repeat = 1;
                    break;
                case '$': // New line marker
                    x = xShift; 
                    y += repeat;
                    repeat = 1;
                    break;
            }
        }
    }
}

int GameRleFileLoader::readIntFromStream()
{
    int repeat = 0; 
    char characterRead;
    characterRead = stream.peek();
    while(characterRead >= '0' && characterRead <= '9')
    {
        stream.get(characterRead);
        repeat *= 10;
        repeat += characterRead - '0';
        characterRead = stream.peek();
    }
    return repeat;
}

void GameRleFileLoader::readDimensions()
{
    char headerLine[256];
    stream.getline(headerLine, 256);

    dataWidth = readIntForKey("x", headerLine);
    dataHeight = readIntForKey("y", headerLine);
}

int GameFileLoader::readIntForKey(std::string key, std::string source)
{
    int retval = 0;
    int i;
    for(i = source.find(key) + key.length(); source[i] == ' ' || source[i] == '\t' || source[i] == '='; ++i);
    while(source[i] >= '0' && source[i] <= '9')
    {
        retval *= 10;
        retval += source[i] - '0';
        ++i;
    }
    return retval;
}
