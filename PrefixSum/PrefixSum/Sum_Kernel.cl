/**********************************************************************
Copyright ©2013 Advanced Micro Devices, Inc. All rights reserved.

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

•	Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
•	Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or
 other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
 DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS
 OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
 NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
********************************************************************/

#define GX get_global_id(0)
#define GLOBAL_SIZE get_global_size(0)

#define LX get_local_id(0)
#define GROUP_ID get_group_id(0)
#define LOCAL_SIZE get_local_size(0)

// Kernel code
__kernel void PrefixSumKernel(__global const int* aBuffer, __global int* bBuffer, __global int* cBuffer, __local int* localArr)
{
    localArr[LX] = aBuffer[GX];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    int k = 8;
    //up-sweep
    for(int d=0; d<k; d++){
        if ((((LX+1)<<(d+1))-1) < 256)
            localArr[((LX+1)<<(d+1))-1] = localArr[((LX+1)<<(d+1))-1]+ localArr[((LX+1)<<(d+1))-1-(1<<d)];

        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    if(LX==256-1)
        localArr[LX]=0;
    
    for (int d=k-1 ; d>=0 ; d--){
        if ((((LX+1)<<(d+1))-1) < 256){
            int tmp = localArr[((LX+1)<<(d+1))-1];
            localArr[((LX+1)<<(d+1))-1] += localArr[((LX+1)<<(d+1))-1-(1<<d)];
            localArr[((LX+1)<<(d+1))-1-(1<<d)] = tmp;
        
        }
        
         barrier(CLK_LOCAL_MEM_FENCE);

    }
    
    bBuffer[GX] = localArr[LX];
    
    if(LX==256-1)
        cBuffer[GROUP_ID] = localArr[LX]+aBuffer[GX];
    
}

__kernel void MergeKernel(__global int* dBuffer, __global int* bBuffer){
    bBuffer[GX]=bBuffer[GX]+dBuffer[GROUP_ID];

}
