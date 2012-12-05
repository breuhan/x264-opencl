#pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable

constant sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP_TO_EDGE | CLK_FILTER_NEAREST;

/* 7.18.1.1  Exact-width integer types */
typedef signed char int8_t;
typedef unsigned char   uint8_t;
typedef short  int16_t;
typedef unsigned short  uint16_t;
typedef int  int32_t;
typedef unsigned   uint32_t;
typedef long  int64_t;
typedef unsigned long uint64_t;

typedef union
{
    uint32_t i;
    uint16_t b[2];
    uint8_t  c[4];
}  x264_union32_t;

typedef uint8_t  pixel;
typedef uint16_t sum_t;
typedef uint32_t sum2_t;

#define LOWRES_COST_MASK ((1<<14)-1)
#define LOWRES_COST_SHIFT 14
#define COST_MAX (1<<28)

#define X264_MIN( a, b ) ((a)<(b) ? (a) : (b))

#define PIXEL_MAX 255
#define BITS_PER_SUM (8 * sizeof(sum_t))

#define COPY2_IF_LT( x, y, a, b )\
    if((y)<(x))\
    {\
        (x) = (y);\
        (a) = (b);\
    }
constant short2 dia_offs[4] =
{
    {0, -1}, {-1, 0}, {1, 0}, {0, 1},
};

inline pixel x264_clip_pixel( int x )
{
    return ((x & ~PIXEL_MAX) ? (-x) >> 31 & PIXEL_MAX : x);
}

inline int x264_clip3( int v, int i_min, int i_max )
{
    return ((v < i_min) ? i_min : (v > i_max) ? i_max : v);
}

inline int x264_median( int a, int b, int c )
{
    int t = (a - b) & ((a - b) >> 31);
    a -= t;
    b += t;
    b -= (b - c) & ((b - c) >> 31);
    b += (a - b) & ((a - b) >> 31);
    return b;
}

inline short2 x264_median_mv( short2 a, short2 b, short2 c )
{
    short2 dst;
    dst.x = x264_median( a.x, b.x, c.x );
    dst.y = x264_median( a.y, b.y, c.y );
    return dst;
}

inline sum2_t abs2( sum2_t a )
{
    sum2_t s = ((a >> (BITS_PER_SUM - 1)) & (((sum2_t)1 << BITS_PER_SUM) + 1)) * ((sum_t)-1);
    return (a + s) ^ s;
}

#define HADAMARD4( d0, d1, d2, d3, s0, s1, s2, s3 ) {\
        sum2_t t0 = s0 + s1;\
        sum2_t t1 = s0 - s1;\
        sum2_t t2 = s2 + s3;\
        sum2_t t3 = s2 - s3;\
        d0 = t0 + t2;\
        d2 = t0 - t2;\
        d1 = t1 + t3;\
        d3 = t1 - t3;\
}

#define HADAMARD4V( d0, d1, d2, d3, s0, s1, s2, s3 ) {\
        int2 t0 = s0 + s1;\
        int2 t1 = s0 - s1;\
        int2 t2 = s2 + s3;\
        int2 t3 = s2 - s3;\
        d0 = t0 + t2;\
        d2 = t0 - t2;\
        d1 = t1 + t3;\
        d3 = t1 - t3;\
}

#define SATD_C_8x4_Q( name, q1, q2 )\
    int name( q1 pixel *pix1, int i_pix1, q2 pixel *pix2, int i_pix2 )\
    {\
        sum2_t tmp[4][4];\
        sum2_t a0, a1, a2, a3;\
        sum2_t sum = 0;\
        for( int i = 0; i < 4; i++, pix1 += i_pix1, pix2 += i_pix2 )\
        {\
            a0 = (pix1[0] - pix2[0]) + ((sum2_t)(pix1[4] - pix2[4]) << BITS_PER_SUM);\
            a1 = (pix1[1] - pix2[1]) + ((sum2_t)(pix1[5] - pix2[5]) << BITS_PER_SUM);\
            a2 = (pix1[2] - pix2[2]) + ((sum2_t)(pix1[6] - pix2[6]) << BITS_PER_SUM);\
            a3 = (pix1[3] - pix2[3]) + ((sum2_t)(pix1[7] - pix2[7]) << BITS_PER_SUM);\
            HADAMARD4( tmp[i][0], tmp[i][1], tmp[i][2], tmp[i][3], a0, a1, a2, a3 );\
        }\
        for( int i = 0; i < 4; i++ )\
        {\
            HADAMARD4( a0, a1, a2, a3, tmp[0][i], tmp[1][i], tmp[2][i], tmp[3][i] );\
            sum += abs2( a0 ) + abs2( a1 ) + abs2( a2 ) + abs2( a3 );\
        }\
        return (((sum_t)sum) + (sum>>BITS_PER_SUM)) >> 1;\
    }

/*
 * Utility function to perform a parallel sum reduction of an array of integers
 */
int parallel_sum( int value, int x, volatile local int *array )
{
    array[x] = value;
    barrier( CLK_LOCAL_MEM_FENCE );

    int dim = get_local_size( 0 );

    while( dim > 1 )
    {
        dim >>= 1;

        if( x < dim )
        {
            array[x] += array[x + dim];
        }

        if( dim > 32 )
        {
            barrier( CLK_LOCAL_MEM_FENCE );
        }
    }

    return array[0];
}

int calc_mv_cost( int dx, int dy )
{
    return round( (log2( (float)(dx + 1) ) * 2.0f + 0.718f + !!dx) ) +
           round( (log2( (float)(dy + 1) ) * 2.0f + 0.718f + !!dy) );
}
