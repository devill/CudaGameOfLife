
#include "GameOfLife.h"

#include <cuda.h>
#include <cuda_runtime_api.h>

#include "CudaError.hpp"

GameOfLife::GameOfLife(int argWidth, int argHeight) 
    : buffer(argWidth,argHeight)
{}

GameOfLife::GameOfLife(GameOfLife& game) 
    : buffer(game.buffer)
{}

void GameOfLife::evolve()
{
    buffer.copyHostToDevice(); 

    buffer.swapDeviceBuffers();
    buffer.clearDeviceNewData();

    dim3 threadGrid(16,16);
    dim3 blockGrid(buffer.getWidth()/16+1,buffer.getHeight()/16+1);
    evolveOnDevice<<<blockGrid,threadGrid>>>(*this);

}

__global__ void evolveOnDevice(GameOfLife game)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(!game.withinBounds(x,y)) return;

    game.evolveCellOnDevice(x,y);
}

__device__ void GameOfLife::evolveCellOnDevice(int x, int y)
{
    int count = countNeighboursThatWereAliveOnDevice(x,y);
   
    if( (count == 3) || (wasAliveOnDevice(x,y) && count == 2) )
        createNewCellOnDevice(x,y);
}

void GameOfLife::createCell(int x, int y)
{
    buffer.copyDeviceToHost();
    buffer.host(x,y) = 1;
}

void GameOfLife::killCell(int x, int y)
{
    buffer.copyDeviceToHost();
    buffer.host(x,y) = 0;
}

bool GameOfLife::isAlive(int x, int y) const
{
    buffer.copyDeviceToHost();
    return 0 != buffer.host(x,y);
}

__device__ void GameOfLife::createNewCellOnDevice(int x, int y)
{
    buffer.deviceNew(x,y) = 1;
}

__device__ int GameOfLife::countNeighboursThatWereAliveOnDevice(int x, int y) const
{
    int count = 0;
    for(int i = -1; i < 2; ++i)
        for(int j = -1; j < 2; ++j)
            if(withinBounds(x+i,y+j) && (i!=0 || j!=0) && wasAliveOnDevice(x+i,y+j))
                count += 1;
    return count;
}

