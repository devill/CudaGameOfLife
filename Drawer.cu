
#include "Drawer.h"
#include "CudaError.hpp"

__global__ void redrawOnDevice(GameOfLife game, uchar4* pixels)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    if(game.withinBounds(i,j)){
        int index = game.findIndex(i,j);

        // RED
        if(game.cellDied(i,j)) 
            pixels[index].x = 127;
        else if(pixels[index].x > 0)
            --pixels[index].x;

        // GREEN
        pixels[index].y = (game.isAliveOnDevice(i,j)) ? 255 : 0;

        // BLUE
        if(game.isAliveOnDevice(i,j) && pixels[index].z < 255)
            ++pixels[index].z;

        // ALPHA
        pixels[index].w = 255;
    }
}

Drawer::Drawer(GameOfLife& argGame) : game(argGame), speed(1) {}
Drawer::~Drawer() { };

void Drawer::redraw(uchar4* pixels, int ticks)
{
    dim3 threadGrid(16,16);
    dim3 blockGrid(game.getWidth()/16+1,game.getHeight()/16+1);
    
    for(int i = 0; i < speed; ++i)
    {
        game.evolve();
        redrawOnDevice<<<blockGrid,threadGrid>>>(game, pixels);
    }
}
