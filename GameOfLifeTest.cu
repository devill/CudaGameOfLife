
#include "gtest/gtest.h"
#include "GameOfLife.h"

class GameOfLifeTest : public testing::Test {
    protected:
        void setCells(GameOfLife& game, int* table)
        {
            for(int i = 0; i < game.getWidth(); ++i)
                for(int j = 0; j < game.getHeight(); ++j)
                {
                    if(table[game.findIndex(i,j)] == 1)
                        game.createCell(i,j);
                    else
                        game.killCell(i,j);
                }
        }

        bool checkCells(const GameOfLife& game, int* table)
        {
            for(int i = 0; i < game.getWidth(); ++i)
                for(int j = 0; j < game.getHeight(); ++j)
                {
                    EXPECT_EQ((table[game.findIndex(i,j)] == 1),game.isAlive(i,j)) << "Coordinates are: " << i << " " << j;
                    if((table[game.findIndex(i,j)] == 1) != game.isAlive(i,j)) return false;
                }
            return true;
        }

        bool testSquareEvolution(int size, int* expected, int* input, int generations = 1)
        {
            GameOfLife game(size,size);
            setCells(game,input);
            for(int i = 0; i < generations; ++i)
                game.evolve();
            return checkCells(game,expected);
        }
};

TEST_F(GameOfLifeTest,Construction)
{
    GameOfLife game(10,30);
    EXPECT_EQ(10,game.getWidth());
    EXPECT_EQ(30,game.getHeight());
}

TEST_F(GameOfLifeTest,CreateCell)
{
    GameOfLife game(100,100);
    ASSERT_FALSE(game.isAlive(50,30));
    ASSERT_FALSE(game.isAlive(30,50));
    game.createCell(50,30);
    ASSERT_TRUE(game.isAlive(50,30));
    ASSERT_FALSE(game.isAlive(10,50));
    game.killCell(50,30);
    ASSERT_FALSE(game.isAlive(50,30));
}

TEST_F(GameOfLifeTest,TestLoader)
{
    GameOfLife game(3,3);
    int table[9] = { 0, 1, 0,
                     1, 1, 1,
                     0, 1, 0 };

    setCells(game,table);
    EXPECT_TRUE(checkCells(game,table));
}

TEST_F(GameOfLifeTest,EmptyEvolves)
{
    int table[9] = { 0, 0, 0,
                     0, 0, 0,
                     0, 0, 0 };
    EXPECT_TRUE(testSquareEvolution(3, table, table));
}

TEST_F(GameOfLifeTest,SingleCell)
{
    int input[9] =    { 0, 0, 0,
                        0, 1, 0,
                        0, 0, 0 };
    int expected[9] = { 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0 };
    EXPECT_TRUE(testSquareEvolution(3, expected, input));
}

TEST_F(GameOfLifeTest,SurviveWithTwoNeighbours)
{
    int input[9] =    { 0, 0, 0,
                        1, 1, 1,
                        0, 0, 0 };
    int expected[9] = { 0, 1, 0,
                        0, 1, 0,
                        0, 1, 0 };
    EXPECT_TRUE(testSquareEvolution(3, expected, input));
}

TEST_F(GameOfLifeTest,CreateWithThreeNeighbours)
{
    int input[9] =    { 0, 1, 0,
                        1, 0, 1,
                        0, 0, 0 };
    int expected[9] = { 0, 1, 0,
                        0, 1, 0,
                        0, 0, 0 };
    EXPECT_TRUE(testSquareEvolution(3, expected, input));
}

TEST_F(GameOfLifeTest,SurviveWithThreeNeighbours)
{
    int input[9] =    { 0, 1, 0,
                        1, 1, 1,
                        0, 0, 0 };
    int expected[9] = { 1, 1, 1,
                        1, 1, 1,
                        0, 1, 0 };
    EXPECT_TRUE(testSquareEvolution(3, expected, input));
}

TEST_F(GameOfLifeTest,StarveWithMoreNeighbours)
{
    int input[9] =    { 0, 1, 0,
                        1, 1, 1,
                        0, 1, 0 };
    int expected[9] = { 1, 1, 1,
                        1, 0, 1,
                        1, 1, 1 };
    EXPECT_TRUE(testSquareEvolution(3, expected, input));
}

TEST_F(GameOfLifeTest,EvolveMoreGenerations)
{
    int input[25] =    { 0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0,
                         0, 1, 1, 1, 0,
                         0, 0, 0, 0, 0,
                         0, 0, 0, 0, 0 };
    EXPECT_TRUE(testSquareEvolution(5, input, input, 100));
}

TEST_F(GameOfLifeTest,TestClear)
{
    GameOfLife game(3,3);
    int input[9] =    { 0, 1, 0,
                        1, 1, 1,
                        0, 1, 0 };
    int expected[9] = { 0, 0, 0,
                        0, 0, 0,
                        0, 0, 0 };
    setCells(game,input);
    game.clear();
    EXPECT_TRUE(checkCells(game,expected));
}
