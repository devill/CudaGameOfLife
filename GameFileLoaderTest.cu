
#include "gtest/gtest.h"
#include "GameOfLife.h"
#include "GameFileLoader.h"

#include <sstream>

class GameLoaderTest : public testing::Test {
    protected:
        bool checkCells(const GameOfLife& game, int* table)
        {
            for(int i = 0; i < game.getWidth(); ++i)
                for(int j = 0; j < game.getHeight(); ++j)
                {
                    EXPECT_EQ((table[game.findIndex(i,j)] == 1),game.isAlive(i,j)) << "Coordinates are: " << i << " " << j << " Index is: " << game.findIndex(i,j);
                    if((table[game.findIndex(i,j)] == 1) != game.isAlive(i,j)) return false;
                }
            return true;
        }
    protected:
        std::stringstream input;
};

TEST_F(GameLoaderTest,LoadFromStream)
{
    input << "3 3" << std::endl;
    input << "..*" << std::endl;
    input << "***" << std::endl;
    input << ".*." << std::endl;

    GameOfLife game(3,3);
    loadGame<GameRawFileLoader>(game, input);

    int expected[9] = { 0, 0, 1,
                        1, 1, 1,
                        0, 1, 0 };
    checkCells(game, expected);
}

TEST_F(GameLoaderTest,LoadFromStreamToLarger)
{
    input << "3 3" << std::endl;
    input << "..*" << std::endl;
    input << "***" << std::endl;
    input << ".*." << std::endl;

    GameOfLife game(5,5);
    loadGame<GameRawFileLoader>(game, input);

    int expected[25] = { 0, 0, 0, 0, 0,
                        0, 0, 0, 1, 0,
                        0, 1, 1, 1, 0,
                        0, 0, 1, 0, 0,
                        0, 0, 0, 0, 0 };
    checkCells(game, expected);
}

TEST_F(GameLoaderTest,LoadZeroFromRLEStream)
{
    input << "x = 0, y = 0" << std::endl;
    input << "!" << std::endl;

    GameOfLife game(3,3);
    loadGame<GameRleFileLoader>(game, input);

    int expected[9] = { 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0 };
    checkCells(game, expected);
}

TEST_F(GameLoaderTest,LoadBlinkerFromRLEStream)
{

    input << "x = 3, y = 3" << std::endl;
    input << "bob$bob$bob!" << std::endl;

    GameOfLife game(3,3);
    loadGame<GameRleFileLoader>(game, input);

    int expected[9] = { 0, 1, 0,
                        0, 1, 0,
                        0, 1, 0 };
    checkCells(game, expected);
}

TEST_F(GameLoaderTest,LoadGliderFromRLEStream)
{
    input << "x = 3, y = 3" << std::endl;
    input << "3o$o$bo!" << std::endl;
    
    GameOfLife game(3,3);
    loadGame<GameRleFileLoader>(game, input);

    int expected[9] = { 1, 1, 1,
                        1, 0, 0,
                        0, 1, 0 };
    checkCells(game, expected);
}

TEST_F(GameLoaderTest,LoadDimensions)
{
    input << "x = 4, y = 6" << std::endl;
    input << "3ob$o3b$bo2$4o$obbo!" << std::endl;
    
    GameOfLife game(4,6);
    loadGame<GameRleFileLoader>(game, input);

    int expected[24] = { 1, 1, 1, 0,
                         1, 0, 0, 0,
                         0, 1, 0, 0,
                         0, 0, 0, 0, 
                         1, 1, 1, 1, 
                         1, 0, 0, 1 };
    checkCells(game, expected);
}
