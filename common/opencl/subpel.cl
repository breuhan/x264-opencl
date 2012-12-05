
/* One thread performs 8x8 SAD */
int sad_8x8_ii_hpel( read_only image2d_t fenc, int2 fencpos, read_only image2d_t fref_planes, int2 qpos )
{
    int2 frefpos = (int2)(qpos.x >> 2, qpos.y >> 2);
    int hpel_idx = ((qpos.x & 2) >> 1) + (qpos.y & 2);

    uint mask_shift = 8 * (3 - hpel_idx);
    uint mask = 0x000000ff << mask_shift;

    uint4 cost4 = 0;

    for( int y = 0; y < 8; y++ )
    {
        uint4 enc, val4;
        enc = read_imageui( fenc, sampler, fencpos + (int2)(0, y));
        val4.s0 = (read_imageui( fref_planes, sampler, frefpos + (int2)(0, y)).s0 & mask) >> mask_shift;
        val4.s1 = (read_imageui( fref_planes, sampler, frefpos + (int2)(1, y)).s0 & mask) >> mask_shift;
        val4.s2 = (read_imageui( fref_planes, sampler, frefpos + (int2)(2, y)).s0 & mask) >> mask_shift;
        val4.s3 = (read_imageui( fref_planes, sampler, frefpos + (int2)(3, y)).s0 & mask) >> mask_shift;
        cost4 += abs_diff( enc, val4 );

        enc = read_imageui( fenc, sampler, fencpos + (int2)(4, y));
        val4.s0 = (read_imageui( fref_planes, sampler, frefpos + (int2)(4, y)).s0 & mask) >> mask_shift;
        val4.s1 = (read_imageui( fref_planes, sampler, frefpos + (int2)(5, y)).s0 & mask) >> mask_shift;
        val4.s2 = (read_imageui( fref_planes, sampler, frefpos + (int2)(6, y)).s0 & mask) >> mask_shift;
        val4.s3 = (read_imageui( fref_planes, sampler, frefpos + (int2)(7, y)).s0 & mask) >> mask_shift;
        cost4 += abs_diff( enc, val4 );
    }

    return cost4.s0 + cost4.s1 + cost4.s2 + cost4.s3;
}

/* One thread measures 8x8 SAD cost at a QPEL offset into an HPEL plane */
int sad_8x8_ii_qpel( read_only image2d_t fenc, int2 fencpos, read_only image2d_t fref_planes, int2 qpos )
{
    int2 frefApos = qpos >> 2;
    int hpelA = ((qpos.x & 2) >> 1) + (qpos.y & 2);

    int2 qposB = qpos + ((qpos & 1) << 1);
    int2 frefBpos = qposB >> 2;
    int hpelB = ((qposB.x & 2) >> 1) + (qposB.y & 2);

    uint mask_shift0 = 8 * (3 - hpelA), mask_shift1 = 8 * (3 - hpelB);
    uint mask0 = 0x000000ff << mask_shift0, mask1 = 0x000000ff << mask_shift1;

    int cost = 0;

    for( int y = 0; y < 8; y++ )
    {
        for( int x = 0; x < 8; x++ )
        {
            uint enc = read_imageui( fenc, sampler, fencpos + (int2)(x, y)).s0;
            uint vA = (read_imageui( fref_planes, sampler, frefApos + (int2)(x, y)).s0 & mask0) >> mask_shift0;
            uint vB = (read_imageui( fref_planes, sampler, frefBpos + (int2)(x, y)).s0 & mask1) >> mask_shift1;
            uint ref = rhadd( vA, vB );
            cost += abs_diff( enc, ref );
        }
    }

    return cost;
}

/* Four thread measures 8x8 SATD cost at a QPEL offset into an HPEL plane
 *
 * Each thread collects 1/4 of the rows of diffs and processes one quarter of
 * the transforms
 */
int satd_8x8_ii_qpel_coop4(
    read_only image2d_t fenc, int2 fencpos,
    read_only image2d_t fref_planes, int2 qpos,
    local sum2_t *tmpp,
    int idx )
{
    volatile local sum2_t( *tmp )[4] = (volatile local sum2_t( * )[4])tmpp;
    sum2_t b0, b1, b2, b3;

    // fencpos is full-pel position of original MB
    // qpos is qpel position within reference frame
    int2 frefApos = (int2)(qpos.x>>2, qpos.y>>2);
    int hpelA = ((qpos.x&2)>>1) + (qpos.y&2);

    int2 qposB = (int2)qpos + (int2)(((qpos.x&1)<<1), ((qpos.y&1)<<1));
    int2 frefBpos = (int2)(qposB.x>>2, qposB.y>>2);
    int hpelB = ((qposB.x&2)>>1) + (qposB.y&2);

    uint mask_shift0 = 8 * (3 - hpelA), mask_shift1 = 8 * (3 - hpelB);
    uint mask0 = 0x000000ff << mask_shift0, mask1 = 0x000000ff << mask_shift1;

    uint vA, vB;
    uint a0, a1;
    uint enc;
    sum2_t sum = 0;


#define READ_DIFF( OUT, X )\
    enc = read_imageui( fenc, sampler, fencpos + (int2)(X, idx) ).s0;\
    vA = (read_imageui( fref_planes, sampler, frefApos + (int2)(X, idx) ).s0 & mask0) >> mask_shift0;\
    vB = (read_imageui( fref_planes, sampler, frefBpos + (int2)(X, idx) ).s0 & mask1) >> mask_shift1;\
    OUT = enc - rhadd( vA, vB );

#define READ_DIFF_EX( OUT, a, b )\
    {\
        READ_DIFF( a0, a );\
        READ_DIFF( a1, b );\
        OUT = a0 + (a1<<BITS_PER_SUM);\
    }
#define ROW_8x4_SATD( a, b )\
    {\
        fencpos.y += a;\
        frefApos.y += b;\
        frefBpos.y += b;\
        READ_DIFF_EX( b0, 0, 4 );\
        READ_DIFF_EX( b1, 1, 5 );\
        READ_DIFF_EX( b2, 2, 6 );\
        READ_DIFF_EX( b3, 3, 7 );\
        HADAMARD4( tmp[idx][0], tmp[idx][1], tmp[idx][2], tmp[idx][3], b0, b1, b2, b3 );\
        HADAMARD4( b0, b1, b2, b3, tmp[0][idx], tmp[1][idx], tmp[2][idx], tmp[3][idx] );\
        sum += abs2( b0 ) + abs2( b1 ) + abs2( b2 ) + abs2( b3 );\
    }
    ROW_8x4_SATD( 0, 0 );
    ROW_8x4_SATD( 4, 4 );

#undef READ_DIFF
#undef READ_DIFF_EX
#undef ROW_8x4_SATD
    return (((sum_t)sum) + (sum>>BITS_PER_SUM)) >> 1;
}

constant short2 hpoffs[4] =
{
    {0, -2}, {-2, 0}, {2, 0}, {0, 2}
};

/*
 * sub pixel refinement of motion vectors, output MVs and costs are moved from
 * temporary buffers into final per-frame buffer
 *
 * global launch dimensions:  [mb_width * 4, mb_height]
 *
 * With X being the source 16x16 pixels, F is the lowres pixel used by the
 * motion search.  We will now utilize the H V and C pixels (stored in separate
 * planes) to search at half-pel increments.
 *
 * X X X X X X
 *  F H F H F
 * X X X X X X
 *  V C V C V
 * X X X X X X
 *  F H F H F
 * X X X X X X
 *
 * The YX HPEL bits of the motion vector selects the plane we search in.  The
 * four planes are packed in the fref_planes 2D image buffer.  Each sample
 * returns:  s0 = F, s1 = H, s2 = V, s3 = C
 */
kernel void subpel_refine(
    read_only image2d_t   fenc,
    read_only image2d_t   fref_planes,
    const global short2  *in_mvs,
    const global int16_t *in_sad_mv_costs,
    local int16_t        *cost_local,
    local sum2_t         *satd_local,
    local short2         *mvc_local,
    global short2        *fenc_lowres_mv,
    global int16_t       *fenc_lowres_mv_costs,
    int                   mb_width,
    int                   lambda,
    int                   b,
    int                   ref,
    int                   b_islist1 )
{
    int mb_x = get_global_id( 0 ) >> 2;
    if( mb_x >= mb_width )
        return;
    int mb_height = get_global_size( 1 );

    int mb_i = get_global_id( 0 ) & 3;
    int mb_y = get_global_id( 1 );
    int mb_xy = mb_y * mb_width + mb_x;

    fenc_lowres_mv += ( b_islist1 ? ( ref - b - 1 ) : ( b - ref - 1 ) ) * mb_width * mb_height;
    fenc_lowres_mv_costs += ( b_islist1 ? ( ref - b - 1 ) : ( b - ref - 1 ) ) * mb_width * mb_height;

    int mb_in_group = get_local_id( 1 ) * (get_local_size( 0 ) >> 2) + (get_local_id( 0 ) >> 2);
    cost_local += mb_in_group * 4;
    satd_local += mb_in_group * 16;
    mvc_local += mb_in_group * 4;

    int i_mvc = 0;
    mvc_local[0] = 0;
    mvc_local[1] = 0;
    mvc_local[2] = 0;
    mvc_local[3] = 0;
    short2 mvp;
#define MVC( DX, DY ) mvc_local[i_mvc++] = in_mvs[mb_width * (mb_y + DY) + (mb_x + DX)];
    if( mb_x > 0 )
        MVC( -1, 0 );
    if( mb_y > 0 )
    {
        MVC( 0, -1 );
        if( mb_x < mb_width - 1 )
            MVC( 1, -1 );
        if( mb_x > 0 )
            MVC( -1, -1 );
    }
    if( i_mvc <= 1 )
        mvp = mvc_local[0];
    else
        mvp = x264_median_mv( mvc_local[0], mvc_local[1], mvc_local[2] );
#undef MVC

    int bcost =  in_sad_mv_costs[mb_xy];
    short2 bmv = in_mvs[mb_xy];

    /* Make mvp a QPEL MV */
    mvp.x <<= 2;
    mvp.y <<= 2;

    /* Make bmv a QPEL MV */
    bmv.x <<= 2;
    bmv.y <<= 2;

    int2 coord;
    coord.x = mb_x << 3;
    coord.y = mb_y << 3;
    short2 trymv;
    int2 qpos;
    int cost = 0;

#define HPEL_QPEL( ARR, FUNC )\
    {\
        trymv = bmv + ARR[mb_i];\
        qpos.x = (coord.x << 2) + trymv.x;\
        qpos.y = (coord.y << 2) + trymv.y;\
        cost = FUNC( fenc, coord, fref_planes, qpos ) + lambda * calc_mv_cost( abs_diff( trymv.x, mvp.x ), abs_diff( trymv.y, mvp.y ) );\
        cost_local[mb_i] = (cost<<2) + mb_i;\
        cost = min( cost_local[0], min( cost_local[1], min( cost_local[2], cost_local[3] ) ) );\
        if( (cost>>2) < bcost )\
        {\
            bmv += ARR[cost&3];\
            bcost = cost>>2;\
        }\
    }

    HPEL_QPEL( hpoffs, sad_8x8_ii_hpel );

#define CHEAP_QPEL 1

#if CHEAP_QPEL
    HPEL_QPEL( dia_offs, sad_8x8_ii_qpel );
#endif

    /* remeasure with SATD */
    qpos.x = (coord.x << 2) + bmv.x;
    qpos.y = (coord.y << 2) + bmv.y;
    cost_local[mb_i] = satd_8x8_ii_qpel_coop4( fenc, coord, fref_planes, qpos, satd_local, mb_i );
    bcost = cost_local[0] + cost_local[1] + cost_local[2] + cost_local[3];
    bcost += lambda * calc_mv_cost( abs_diff( bmv.x, mvp.x ), abs_diff( bmv.y, mvp.y ) );
    short2 bestmv = 0;

#if !CHEAP_QPEL
#define QPEL_SATD( ARR, FUNC )\
    {\
        trymv = bmv + ARR;\
        qpos.x = (coord.x << 2) + trymv.x;\
        qpos.y = (coord.y << 2) + trymv.y;\
        cost_local[mb_i] = FUNC( fenc, coord, fref_planes, qpos, satd_local, mb_i );\
        cost = cost_local[0] + cost_local[1] + cost_local[2] + cost_local[3];\
        cost += lambda * calc_mv_cost( abs_diff( trymv.x, mvp.x ), abs_diff( trymv.y, mvp.y ) );\
        if( cost < bcost )\
        {\
            bestmv = ARR;\
            bcost = cost;\
        }\
    }
    for( int i = 0; i<4; i++ )
        QPEL_SATD( dia_offs[i], satd_8x8_ii_qpel_coop4 );
#endif

    fenc_lowres_mv[mb_xy] = bmv+bestmv;
    fenc_lowres_mv_costs[mb_xy] = X264_MIN( bcost, LOWRES_COST_MASK );
}
