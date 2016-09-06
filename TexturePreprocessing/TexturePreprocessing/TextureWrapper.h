//  TextureWrapper.h
//  TexturePreprocessing

//  Copyright (c) 2015 Jan Gruber. All rights reserved.
//

#ifndef __TexturePreprocessing__TextureWrapper__
#define __TexturePreprocessing__TextureWrapper__

#include <stdio.h>
#include <string>

#include "tiffio.h"
#include "OpenCLMgr.h"




class TextureWrapper
{
public:
    TextureWrapper(const std::string& filename);
    ~TextureWrapper();
    
    int preprocessTexture(uint16 flag);
    void readRGBATiff(std::string filename);
    void writeRGBATiff(std::string filename);
    
    //OpenCLMgr * mgr;
    
    enum {
        PROCESS_AT_ONCE = 0x01,
        PROCESS_X = 0x02,
        PROCESS_Y = 0x04,
        TEST = 0x08,
    };

private:

    uint32* inRaster;
    uint32* outRaster;
    uint32 width;
    uint32 imagelength;
    
    int blend();
    int blendSeparate(uint16 flag);
    int formTestMosaic();
    
public:
    static OpenCLMgr * mgr;
    
};


#endif /* defined(__TexturePreprocessing__TextureWrapper__) */
