
#ifndef GAME_FILE_LOADER_H
#define GAME_FILE_LOADER_H

#include "GameOfLife.h"
#include <boost/noncopyable.hpp>

class GameFileLoader : private boost::noncopyable
{
    public:
        GameFileLoader(GameOfLife& argGame, std::istream& argStream) : game(argGame), stream(argStream) {};
        virtual ~GameFileLoader() {};

        void load();

    protected:
        virtual void readDimensions() = 0;
        virtual void loadData() = 0;
        int readIntForKey(std::string key, std::string source);

    private:
        void calculateDimensionShifts();

    protected:
        GameOfLife& game;
        std::istream& stream;

        int dataWidth;
        int dataHeight;
        int xShift;
        int yShift;
};

class GameRawFileLoader : public GameFileLoader
{
    public:
        GameRawFileLoader(GameOfLife& argGame, std::istream& argStream) : GameFileLoader(argGame, argStream) {};
        virtual ~GameRawFileLoader() {}

    protected:
        virtual void readDimensions(); 
        virtual void loadData();
};

class GameRleFileLoader : public GameFileLoader
{
    public:
        GameRleFileLoader(GameOfLife& argGame, std::istream& argStream) : GameFileLoader(argGame, argStream) {};
        virtual ~GameRleFileLoader() {}

    protected:
        virtual void readDimensions();
        virtual void loadData();

    private:
        int readIntFromStream();
};

template<class Loader>
void loadGame(GameOfLife& game, std::istream& stream)
{
    Loader loader(game,stream);
    loader.load();
}

#endif
