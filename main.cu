
#include "GameOfLife.h"
#include "FileOperations.h"
#include "GlutDisplay.h"
#include "Drawer.h"

#include <boost/bind.hpp>

#include <iostream>
#include <string>

Drawer* globalDrawer;

void keyHandler(unsigned char key, int x, int y);
void terminate();

int main(int argc, char* argv[])
{
    using namespace boost;

    GlutDisplay& display = GlutDisplay::createInstance(argc, argv);
    GameOfLife game(display.getWidth(), display.getHeight());

    try
    {
        std::string filename = getFilenameFromCommandLine(argc, argv);
        readFileToGame(game, filename);
    }
    catch(FileReadException e)
    {
        std::cerr << e.what() << std::endl;
        terminate();
    }

    Drawer drawer(game);
    // Wish glut supported function objects. Than I wouldn't need a global here...
    globalDrawer = &drawer;
    glutKeyboardFunc( keyHandler );

    display.startAnimation(boost::bind(&Drawer::redraw, boost::ref(drawer), _1, _2));
}

void keyHandler(unsigned char key, int x, int y)
{
    switch (key) 
    {
        case '+':
            globalDrawer->increaseSpeed();
            break;
        case '-':
            globalDrawer->decreaseSpeed();
            break;
        default:
            terminate();
            break;
    }
}

void terminate()
{
    GlutDisplay::resetInstance();
    exit(0);
}
