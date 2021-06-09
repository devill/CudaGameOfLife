
#ifndef GAME_OF_LIFE_H
#define GAME_OF_LIFE_H

#include "SyncedSwapBuffer.hpp"

class GameOfLife
{
    friend __global__ void evolveOnDevice(GameOfLife game);

    public:
        GameOfLife(int argWidth, int argHeight);
        GameOfLife(GameOfLife& game);

        void evolve();
        void clear() { buffer.clear(); }

        void createCell(int x, int y);
        void killCell(int x, int y);
        bool isAlive(int x, int y) const;
       
        __device__ bool isAliveOnDevice(int x, int y) const { return 0 != buffer.deviceNew(x,y); }
        __device__ bool wasAliveOnDevice(int x, int y) const { return 0 != buffer.deviceOld(x,y); }
        __device__ bool cellDied(int x, int y) const { return !isAliveOnDevice(x,y) && wasAliveOnDevice(x,y); }

        __host__ __device__ int getWidth() const { return buffer.getWidth(); }
        __host__ __device__ int getHeight() const { return buffer.getHeight(); }

        __host__ __device__ int findIndex(int x, int y) const { return buffer.findIndex(x,y); }
        __host__ __device__ bool withinBounds(int x, int y) const { return buffer.withinBounds(x,y); }

    private:
        GameOfLife& operator=(GameOfLife& game);
        
        __device__ void evolveCellOnDevice(int x, int y);

        __device__ void createNewCellOnDevice(int x, int y);

        __device__ int countNeighboursThatWereAliveOnDevice(int x, int y) const;

    private:
        SyncedSwapBuffer buffer;
};

__global__ void evolveOnDevice(GameOfLife game);

#endif
