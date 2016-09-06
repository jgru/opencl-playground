

#define GX get_global_id(0)
#define GLOBAL_SIZE get_global_size(0)

#define LX get_local_id(0)
#define GROUP_ID get_group_id(0)
#define LOCAL_SIZE get_local_size(0) 

#define BITDEPTH 8

kernel void calcStatistic(__global uchar4 *in, size_t len, __global int *concatenatedHist, size_t pixelPerItem) {
    __local int counts[0x20][1<<BITDEPTH]; //worksize is said to be 0x20
    
    //init with 0
    for(int x=0; x<(1<<BITDEPTH);x++){
        counts[LX][x]=0;
    }
    
    for(int j=0; j<pixelPerItem;j++){
        int idx = GROUP_ID*LOCAL_SIZE*pixelPerItem+LX+j*0x20;
        if(idx<len){
            //G==s1, R==s0, G==s1, B == s2
            float mix =0.299 * in[idx].s0 + 0.587 * in[idx].s1 + 0.114 * in[idx].s2;
            uchar luma = convert_uchar_rtz(mix);
            counts[LX][luma] += 1;
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);
    int m = (1<<BITDEPTH)/LOCAL_SIZE;
    for(int i=0;i<m;i++){
        int sum=0;
        for(int k=0; k<LOCAL_SIZE;k++){
            sum += counts[k][LX+i*0x20];
        }
        concatenatedHist[((1<<BITDEPTH)*GROUP_ID)+(LX+i*LOCAL_SIZE)]=sum;
        
    }
    
}

kernel void calcStatisticAtomic(__global uchar4 *in, size_t len, __global int *concatenatedHist, size_t pixelPerItem) {
    __local int counts[1<<BITDEPTH];
    //init with 0
    int factor = (1<<BITDEPTH)/LOCAL_SIZE;
    for(int x=0; x<factor;x++){
        counts[x*LOCAL_SIZE+LX]=0;
    } 

    
    for(int j=0; j<pixelPerItem;j++){
        int idx = GROUP_ID*0x20*pixelPerItem+LX+j*0x20;
        if(idx<len){
            //G==s1, R==s0, G==s1, B == s2
            float mix =0.299 * in[idx].s0 + 0.587 * in[idx].s1 + 0.114 * in[idx].s2;
            uchar luma = convert_uchar_rtz(mix);
            atomic_inc(&counts[luma]);
        }
    }
    
    barrier(CLK_LOCAL_MEM_FENCE);

    for(int i=0;i<8;i++){
        concatenatedHist[((1<<BITDEPTH)*GROUP_ID)+(LX+i*0x20)]=counts[LX+i*0x20];
    }

    
}

kernel void reduceStatistic(global int* concatenatedHist, int count) {
   int sum = 0;
    for(int i=0; i<count; i++){
        sum += concatenatedHist[LX + i*0x100];

    }
 
    concatenatedHist[LX]=sum;

}



