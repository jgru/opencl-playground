//
//  ArrayUtils.h
//  PrefixSum
//
//  Created by gru on 1/6/16.
//  Copyright (c) 2016 Jan Gruber. All rights reserved.
//

#ifndef __PrefixSum__ArrayUtils__
#define __PrefixSum__ArrayUtils__


#include "OpenCLMgr.h"

class ArrayUtils
{
    public:
        static int calcPrefixSumOnGPU(cl_int* input, cl_int* output, size_t len);
        static void calcPrefixSumOnCPU(cl_int* input, cl_int* output, size_t len);
    
    private:
        static int prefixSumRecursive(OpenCLMgr& mgr, cl_mem aBuffer, cl_mem bBuffer, size_t len, size_t lws);
    static size_t calcNextMultiple(size_t len, size_t n);
};

#endif /* defined(__PrefixSum__ArrayUtils__) */
