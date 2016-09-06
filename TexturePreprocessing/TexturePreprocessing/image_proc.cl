#define MIN_VAL (uchar) 0x00
#define MAX_VAL (uchar) 0xFF

#define GX get_global_id(0)
#define GY get_global_id(1)

#define GW get_global_size(0)
#define GH get_global_size(1)


float calculateAlpha(int pos, int len){
    float as =  (-2.0*pos/(len-1.0) + 1.0);
    return fabs(as);
}

uchar4 blendPixel(uchar4 in1,uchar4 in2, float mix_value){
    float4 fin1 = convert_float4(in1);
    float4 fin2 = convert_float4(in2);
    float4 fout = mix(fin1, fin2, mix_value);
    
    uchar4 out = convert_uchar4_rte(fout);
    out = clamp(out, MIN_VAL, MAX_VAL);
    
    return out;

}


/*
 *  Visualization of the indexing in a 3x2-image
 *
 *  |--------------------------------------------------|
 *  |                 |               |                |
 *  |                 |               |                |
 *  |                 |               |                |
 *  |  GY*GW+xshift   |               |  GY*GW+GX      |
 *  |                 |               |                |
 *  |                 |               |                |
 *  |                 |               |                |
 *  |--------------------------------------------------|
 *  |                 |               |                |
 *  |                 |               |                |
 *  |                 |               |                |
 *  |yshift*GW+xshift |               |  yshift*GW+GX  |
 *  |                 |               |                |
 *  |                 |               |                |
 *  |                 |               |                |
 *  |--------------------------------------------------|
 */


kernel void xblend(global uchar4 *in, global uchar4 *out) {
    //horizontal cyclic shift
    int xshift = (GX + GW/2) % GW;
    in += (GY*GW + xshift);
    
    //save shifted pixel
    out += GY*GW + GX;
    *out = *in;

    
    //step back to the address of original input pixel
    in -= xshift;
    in += GX;
    
    //alpha blending
    //calculate a'
    float m = calculateAlpha(GX, GW);
    //blend each channel together
    uchar4 v = blendPixel(*in, *out, m);
    *out = v;
     
}

kernel void yblend(global uchar4 *in, global uchar4 *out) {
    //vertical cyclic shift
    int yshift = (GY + GH/2) % GH;
    in += (yshift * GW + GX);
    
    //save shifted pixel
    out += GY*GW + GX;
    *out = *in;
    
    //step back to the address of original input pixel
    in -= yshift*GW;
    in += GY*GW;
    
    
    
    //alpha blending
    //calculate a'
    float mix = calculateAlpha(GY, GH);
    //blend each channel together
    uchar4 v = blendPixel(*in, *out, mix );
    *out = v;
}


kernel void xyblend(global uchar4 *in, global uchar4 *out) {
    
    /* calc x-shift of pixel */
    /* ====================================================================*/
    
    //calc horizontal cyclic shift
    int xshift = (GX + GW/2) % GW;
    in += (GY*GW + xshift);
    uchar4 xShifted = *in;
    
    //reverse shift; step to the address of original input pixel
    in -= xshift;
    in += GX;
    uchar4 xOrig = *in;
    
    //reset ptr to the first array elem
    in -= (GY*GW+GX);
    
    //alpha blending
    float m = calculateAlpha(GX, GW);
    uchar4 v = blendPixel(xOrig, xShifted, m);
    
    /* calc x-shift of pixel, which will be shifted in vertical direction */
    /* ===================================================================*/
    
    //calc vertical cyclic shift
    int yshift = (GY + GH/2) % GH;
    in += (yshift * GW + GX);
    uchar4 yShifted = *in;
    
    //set ptr to horizontally shifted pos
    in -= GX;
    in += xshift;
    uchar4 xyShifted = *in;
    
    //alpha blending
    m = calculateAlpha(GX, GW);
    uchar4 v2 = blendPixel(yShifted, xyShifted, m);
    
    //alpha blending
    m = calculateAlpha(GY, GH);
    //blend each channel together
    uchar4 v3 = blendPixel(v, v2, m );
    
    //step to the memory address of the computed pixel
    out += GY*GW + GX;
    *out = v3;
}

kernel void testMosaic(global uchar4 *in, global uchar4 *out) {
    //coordination transformation
    int newX = (GX*2)%GW;
    int newY = (GY*2)%GH;
    
    in += newY*GW+newX;
    out += GY*GW+GX;
    *out = *in;
}