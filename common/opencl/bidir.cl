
/* Four threads cooperatively measure 8x8 BIDIR cost with SATD */
int bidir_satd_8x8_ii_coop4(
    read_only image2d_t fenc_lowres,  int2 fencpos,
    read_only image2d_t fref0_planes, int2 qpos0,
    read_only image2d_t fref1_planes, int2 qpos1,
    int weight,
    local sum2_t *tmpp,
    int idx )
{
    volatile local sum2_t( *tmp )[4] = (volatile local sum2_t( * )[4])tmpp;
    sum2_t b0, b1, b2, b3;
    sum2_t sum = 0;

    // fencpos is full-pel position of original MB
    // qpos0 is qpel position within reference frame 0
    // qpos1 is qpel position within reference frame 1

    int2 fref0Apos = (int2)(qpos0.x>>2, qpos0.y>>2);
    int hpel0A = ((qpos0.x&2)>>1) + (qpos0.y&2);

    int2 qpos0B = (int2)qpos0 + (int2)(((qpos0.x&1)<<1), ((qpos0.y&1)<<1));
    int2 fref0Bpos = (int2)(qpos0B.x>>2, qpos0B.y>>2);
    int hpel0B = ((qpos0B.x&2)>>1) + (qpos0B.y&2);

    int2 fref1Apos = (int2)(qpos1.x>>2, qpos1.y>>2);
    int hpel1A = ((qpos1.x&2)>>1) + (qpos1.y&2);

    int2 qpos1B = (int2)qpos1 + (int2)(((qpos1.x&1)<<1), ((qpos1.y&1)<<1));
    int2 fref1Bpos = (int2)(qpos1B.x>>2, qpos1B.y>>2);
    int hpel1B = ((qpos1B.x&2)>>1) + (qpos1B.y&2);

    uint mask_shift0A = 8 * (3 - hpel0A), mask_shift0B = 8 * (3 - hpel0B);
    uint mask0A = 0x000000ff << mask_shift0A, mask0B = 0x000000ff << mask_shift0B;
    uint mask_shift1A = 8 * (3 - hpel1A), mask_shift1B = 8 * (3 - hpel1B);
    uint mask1A = 0x000000ff << mask_shift1A, mask1B = 0x000000ff << mask_shift1B;

    uint vA, vB;
    uint enc, ref0, ref1;
    uint a0, a1;
    const int weight2 = 64 - weight;

#define READ_BIDIR_DIFF( OUT, X )\
    enc = read_imageui( fenc_lowres, sampler, fencpos + (int2)(X, idx) ).s0;\
    vA = (read_imageui( fref0_planes, sampler, fref0Apos + (int2)(X, idx) ).s0 & mask0A) >> mask_shift0A;\
    vB = (read_imageui( fref0_planes, sampler, fref0Bpos + (int2)(X, idx) ).s0 & mask0B) >> mask_shift0B;\
    ref0 = rhadd( vA, vB );\
    vA = (read_imageui( fref1_planes, sampler, fref1Apos + (int2)(X, idx) ).s0 & mask1A) >> mask_shift1A;\
    vB = (read_imageui( fref1_planes, sampler, fref1Bpos + (int2)(X, idx) ).s0 & mask1B) >> mask_shift1B;\
    ref1 = rhadd( vA, vB );\
    OUT = enc - ((ref0 * weight + ref1 * weight2 + (1 << 5)) >> 6);

#define READ_DIFF_EX( OUT, a, b )\
    READ_BIDIR_DIFF( a0, a );\
    READ_BIDIR_DIFF( a1, b );\
    OUT = a0 + (a1<<BITS_PER_SUM);

#define ROW_8x4_SATD( a, b, c )\
    fencpos.y += a;\
    fref0Apos.y += b;\
    fref0Bpos.y += b;\
    fref1Apos.y += c;\
    fref1Bpos.y += c;\
    READ_DIFF_EX( b0, 0, 4 );\
    READ_DIFF_EX( b1, 1, 5 );\
    READ_DIFF_EX( b2, 2, 6 );\
    READ_DIFF_EX( b3, 3, 7 );\
    HADAMARD4( tmp[idx][0], tmp[idx][1], tmp[idx][2], tmp[idx][3], b0, b1, b2, b3 );\
    HADAMARD4( b0, b1, b2, b3, tmp[0][idx], tmp[1][idx], tmp[2][idx], tmp[3][idx] );\
    sum += abs2( b0 ) + abs2( b1 ) + abs2( b2 ) + abs2( b3 );

    ROW_8x4_SATD( 0, 0, 0 );
    ROW_8x4_SATD( 4, 4, 4 );

#undef READ_BIDIR_DIFF
#undef READ_DIFF_EX
#undef ROW_8x4_SATD

    return (((sum_t)sum) + (sum>>BITS_PER_SUM)) >> 1;
}

/*
 * mode selection
 *
 * global launch dimensions:  [mb_width, mb_height]
 *
 * If this is a B frame, launch dims are [mb_width * 4, mb_height]
 */
kernel void mode_selection(
    read_only image2d_t   fenc_lowres,
    read_only image2d_t   fref0_planes,
    read_only image2d_t   fref1_planes,
    const global short2  *fenc_lowres_mvs0,
    const global short2  *fenc_lowres_mvs1,
    const global short2  *fref1_lowres_mvs0,
    const global int16_t *fenc_lowres_mv_costs0,
    const global int16_t *fenc_lowres_mv_costs1,
    const global uint16_t *fenc_intra_cost,
    global uint16_t      *lowres_costs,
    global int           *frame_stats,
    local int16_t        *cost_local,
    local sum2_t         *satd_local,
    int                   mb_width,
    int                   bipred_weight,
    int                   dist_scale_factor,
    int                   b,
    int                   p0,
    int                   p1,
    int                   lambda )
{
    int mb_x = get_global_id( 0 );
    int b_bidir = (b < p1);
    if( b_bidir )
    {
        mb_x >>= 2;
        if( mb_x >= mb_width )
            return;
    }
    int mb_y = get_global_id( 1 );
    int mb_height = get_global_size( 1 );
    int mb_count = mb_width * mb_height;
    int mb_xy = mb_x + mb_y * mb_width;

    /* Initialize frame stats for next kernel (sum_inter_cost) */
    if( mb_x < 4 && mb_y == 0 )
        frame_stats[mb_x] = 0;

    int bcost = COST_MAX;
    int list_used = 0;

    if( !b_bidir )
    {
        int icost = fenc_intra_cost[mb_xy];
        COPY2_IF_LT( bcost, icost, list_used, 0 );
    }
    if( b != p0 )
    {
        int mv_cost0 = fenc_lowres_mv_costs0[(b - p0 - 1) * mb_count + mb_xy];
        COPY2_IF_LT( bcost, mv_cost0, list_used, 1 );
    }
    if( b != p1 )
    {
        int mv_cost1 = fenc_lowres_mv_costs1[(p1 - b - 1) * mb_count + mb_xy];
        COPY2_IF_LT( bcost, mv_cost1, list_used, 2 );
    }

    if( b_bidir )
    {
        int2 coord = (int2)(mb_x << 3, mb_y << 3);
        int mb_i = get_global_id( 0 ) & 3;
        int mb_in_group = get_local_id( 1 ) * (get_local_size( 0 ) >> 2) + (get_local_id( 0 ) >> 2);
        cost_local += mb_in_group * 4;
        satd_local += mb_in_group * 16;

#define TRY_BIDIR( mv0, mv1, penalty )\
    {\
        int2 qpos0 = (int2)((coord.x<<2) + mv0.x, (coord.y<<2) + mv0.y);\
        int2 qpos1 = (int2)((coord.x<<2) + mv1.x, (coord.y<<2) + mv1.y);\
        cost_local[mb_i] = bidir_satd_8x8_ii_coop4( fenc_lowres, coord, fref0_planes, qpos0, fref1_planes, qpos1, bipred_weight, satd_local, mb_i );\
        int cost = cost_local[0] + cost_local[1] + cost_local[2] + cost_local[3];\
        COPY2_IF_LT( bcost, penalty * lambda + cost, list_used, 3 );\
    }
        /* temporal prediction */
        short2 dmv0, dmv1;
        short2 mvr = fref1_lowres_mvs0[mb_xy];
        dmv0.x = (mvr.x * dist_scale_factor + 128) >> 8;
        dmv0.y = (mvr.y * dist_scale_factor + 128) >> 8;
        dmv1.x = dmv0.x - mvr.x;
        dmv1.y = dmv0.y - mvr.y;
        TRY_BIDIR( dmv0, dmv1, 0 )

        if( as_uint( dmv0 ) || as_uint( dmv1 ) )
        {
            /* B-direct prediction */
            dmv0 = 0; dmv1 = 0;
            TRY_BIDIR( dmv0, dmv1, 0 );
        }

        /* L0+L1 prediction */
        dmv0 = fenc_lowres_mvs0[(b - p0 - 1) * mb_count + mb_xy];
        dmv1 = fenc_lowres_mvs1[(p1 - b - 1) * mb_count + mb_xy];
        TRY_BIDIR( dmv0, dmv1, 5 );
#undef TRY_BIDIR
    }

    lowres_costs[mb_xy] = X264_MIN( bcost, LOWRES_COST_MASK ) + (list_used << LOWRES_COST_SHIFT);
}

/*
 * parallel sum inter costs
 *
 * global launch dimensions: [256, mb_height]
 */
kernel void sum_inter_cost(
    const global uint16_t *fenc_lowres_costs,
    const global uint16_t *inv_qscale_factor,
    global int           *fenc_row_satds,
    global int           *frame_stats,
    int                   mb_width,
    int                   bframe_bias,
    int                   b,
    int                   p0,
    int                   p1 )
{
    int y = get_global_id( 1 );
    int mb_height = get_global_size( 1 );

    int row_satds = 0;
    int cost_est = 0;
    int cost_est_aq = 0;
    int intra_mbs = 0;

    for( int x = get_global_id( 0 ); x < mb_width; x += get_global_size( 0 ))
    {
        int mb_xy = x + y * mb_width;
        int cost = fenc_lowres_costs[mb_xy] & LOWRES_COST_MASK;
        int list = fenc_lowres_costs[mb_xy] >> LOWRES_COST_SHIFT;
        int b_frame_score_mb = (x > 0 && x < mb_width - 1 && y > 0 && y < mb_height - 1) || mb_width <= 2 || mb_height <= 2;

        if( list == 0 && b_frame_score_mb )
        {
            intra_mbs++;
        }

        int cost_aq = (cost * inv_qscale_factor[mb_xy] + 128) >> 8;

        row_satds += cost_aq;

        if( b_frame_score_mb )
        {
            cost_est += cost;
            cost_est_aq += cost_aq;
        }
    }

    local int buffer[256];
    int x = get_global_id( 0 );

    row_satds   = parallel_sum( row_satds, x, buffer );
    cost_est    = parallel_sum( cost_est, x, buffer );
    cost_est_aq = parallel_sum( cost_est_aq, x, buffer );
    intra_mbs   = parallel_sum( intra_mbs, x, buffer );

    if( b != p1 )
        cost_est = (int)((float)cost_est * 100 / (120 + bframe_bias));

    if( get_global_id( 0 ) == 0 )
    {
        fenc_row_satds[y] = row_satds;
        atomic_add( frame_stats + 0, cost_est );
        atomic_add( frame_stats + 1, cost_est_aq );
        atomic_add( frame_stats + 2, intra_mbs );
    }
}
