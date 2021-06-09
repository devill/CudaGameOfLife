
#include "GameOfLife.h"
#include <boost/noncopyable.hpp>

class Drawer : private boost::noncopyable
{
    public:
        Drawer(GameOfLife& argGame);
        ~Drawer();

        void redraw(uchar4* pixels, int ticks);

        void increaseSpeed() { speed*=2; }
        void decreaseSpeed() { if(speed > 1) speed /= 2; }
    private:
        GameOfLife& game;
        int speed;
};

