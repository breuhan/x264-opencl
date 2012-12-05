


int find_downscale_mb_xy( int x, int y, int mb_width, int mb_height )
{
    /* edge macroblocks might not have a direct descendant, use nearest */
    x = (x == mb_width-1)  ? (x - (mb_width&1)) >> 1 : x >> 1;
    y = (y == mb_height-1) ? (y - (mb_height&1)) >> 1 : y >> 1;
    return (mb_width>>1) * y + x;
}

/* Four threads calculate an 8x8 SAD.  Each does two rows */
int sad_8x8_ii_coop4( read_only image2d_t fenc, int2 fencpos, read_only image2d_t fref, int2 frefpos, int idx, local int16_t *costs )
{
    frefpos.y += idx << 1;
    fencpos.y += idx << 1;
    int cost = 0;
    if( frefpos.x < 0 )
    {
        /* slow path when MV goes past right edge */
        for( int y = 0; y < 2; y++ )
        {
            for( int x = 0; x < 8; x++ )
            {
                pixel enc = read_imageui( fenc, sampler, fencpos + (int2)(x, y) ).s0;
                pixel ref = read_imageui( fref, sampler, frefpos + (int2)(x, y) ).s0;
                cost += abs_diff( enc, ref );
            }
        }
    }
    else
    {
        uint4 enc, ref, costs = 0;
        enc = read_imageui( fenc, sampler, fencpos );
        ref = read_imageui( fref, sampler, frefpos );
        costs += abs_diff( enc, ref );
        enc = read_imageui( fenc, sampler, fencpos + (int2)(4, 0) );
        ref = read_imageui( fref, sampler, frefpos + (int2)(4, 0) );
        costs += abs_diff( enc, ref );
        enc = read_imageui( fenc, sampler, fencpos + (int2)(0, 1) );
        ref = read_imageui( fref, sampler, frefpos + (int2)(0, 1) );
        costs += abs_diff( enc, ref );
        enc = read_imageui( fenc, sampler, fencpos + (int2)(4, 1) );
        ref = read_imageui( fref, sampler, frefpos + (int2)(4, 1) );
        costs += abs_diff( enc, ref );
        cost = costs.s0 + costs.s1 + costs.s2 + costs.s3;
    }
    costs[idx] = cost;
    return costs[0] + costs[1] + costs[2] + costs[3];
}

/* One thread performs 8x8 SAD */
int sad_8x8_ii( read_only image2d_t fenc, int2 fencpos, read_only image2d_t fref, int2 frefpos )
{
    if( frefpos.x < 0 )
    {
        /* slow path when MV goes past right edge */
        int cost = 0;
        for( int y = 0; y < 8; y++ )
        {
            for( int x = 0; x < 8; x++ )
            {
                uint enc = read_imageui( fenc, sampler, fencpos + (int2)(x, y) ).s0;
                uint ref = read_imageui( fref, sampler, frefpos + (int2)(x, y) ).s0;
                cost += abs_diff( enc, ref );
            }
        }
        return cost;
    }
    else
    {
        uint4 enc, ref, cost = 0;
        for( int y = 0; y < 8; y++ )
        {
            for( int x = 0; x < 8; x += 4 )
            {
                enc = read_imageui( fenc, sampler, fencpos + (int2)(x, y) );
                ref = read_imageui( fref, sampler, frefpos + (int2)(x, y) );
                cost += abs_diff( enc, ref );
            }
        }
        return cost.s0 + cost.s1 + cost.s2 + cost.s3;
    }
}


/*
 * hierarchical motion estimation
 *
 * Each kernel launch is a single iteration
 *
 * MB per work group is determined by lclx / 4 * lcly
 *
 * global launch dimensions:  [mb_width * 4, mb_height]
 */
kernel void hierarchical_motion(
    read_only image2d_t  fenc,
    read_only image2d_t  fref,
    const global short2 *in_mvs,
    global short2       *out_mvs,
    global int16_t      *out_mv_costs,
    global short2       *mvp_buffer,
    local int16_t       *cost_local,
    local short2        *mvc_local,
    int                  mb_width,
    int                  lambda,
    int                  me_range,
    int                  scale,
    int                  b_shift_index,
    int                  b_first_iteration,
    int                  b_reverse_references )
{
    int mb_x = get_global_id( 0 ) >> 2;
    if( mb_x >= mb_width )
        return;
    int mb_height = get_global_size( 1 );
    int mb_i = get_global_id( 0 ) & 3;
    int mb_y = get_global_id( 1 );
    int mb_xy = mb_y * mb_width + mb_x;
    const int mb_size = 8;
    int2 coord;
    coord.x = mb_x * mb_size;
    coord.y = mb_y * mb_size;

    const int mb_in_group = get_local_id( 1 ) * (get_local_size( 0 ) >> 2) + (get_local_id( 0 ) >> 2);
    cost_local += 4 * mb_in_group;

    int i_mvc = 0;
    mvc_local += 4 * mb_in_group;
    mvc_local[mb_i] = 0;
    short2 mvp;

    if( b_first_iteration )
    {
        mvp.x = 0;
        mvp.y = 0;
    }
    else
    {
#define MVC( DX, DY )\
    {\
        int px = mb_x + DX;\
        int py = mb_y + DY;\
        if( b_shift_index )\
            mvc_local[i_mvc] = in_mvs[find_downscale_mb_xy( px, py, mb_width, mb_height )];\
        else\
            mvc_local[i_mvc] = in_mvs[mb_width * py + px];\
        mvc_local[i_mvc].x >>= scale;\
        mvc_local[i_mvc].y >>= scale;\
        i_mvc++;\
    }
        /* Find MVP from median of MVCs */
        if( b_reverse_references )
        {
            /* odd iterations: derive MVP from down and right */
            if( mb_x < mb_width - 1 )
                MVC( 1, 0 );
            if( mb_y < mb_height - 1 )
            {
                MVC( 0, 1 );
                if( mb_x > b_shift_index )
                    MVC( -1, 1 );
                if( mb_x < mb_width - 1 )
                    MVC( 1, 1 );
            }
        }
        else
        {
            /* even iterations: derive MVP from up and left */
            if( mb_x > 0 )
                MVC( -1, 0 );
            if( mb_y > 0 )
            {
                MVC( 0, -1 );
                if( mb_x < mb_width - 1 )
                    MVC( 1, -1 );
                if( mb_x > b_shift_index )
                    MVC( -1, -1 );
            }
        }
        if( i_mvc <= 1 )
        {
            mvp = mvc_local[0];
        }
        else
            mvp = x264_median_mv( mvc_local[0], mvc_local[1], mvc_local[2] );
#undef MVC
    }
    //new mvp == old mvp, copy the input mv to the output mv and exit.
    if( (!b_shift_index) && (mvp.x == mvp_buffer[mb_xy].x) && (mvp.y == mvp_buffer[mb_xy].y) )
    {
        out_mvs[mb_xy] = in_mvs[mb_xy];
        return;
    }
    mvp_buffer[mb_xy] = mvp;
    short2 mv_min;
    short2 mv_max;
    mv_min.x = -mb_size * mb_x - 4;
    mv_max.x = mb_size * (mb_width - mb_x - 1) + 4;
    mv_min.y = -mb_size * mb_y - 4;
    mv_max.y = mb_size * (mb_height - mb_y - 1) + 4;

    short2 bestmv;
    bestmv.x = x264_clip3( mvp.x, mv_min.x, mv_max.x );
    bestmv.y = x264_clip3( mvp.y, mv_min.y, mv_max.y );

    int2 refcrd;
    refcrd.x = coord.x + bestmv.x;
    refcrd.y = coord.y + bestmv.y;
    /* measure cost at bestmv */
    int bcost = sad_8x8_ii_coop4( fenc, coord, fref, refcrd, mb_i, cost_local ) +
                lambda * calc_mv_cost( abs_diff( bestmv.x, mvp.x ) << (2 + scale), abs_diff( bestmv.y, mvp.y ) << (2 + scale) );

    do
    {
        /* measure costs at offsets from bestmv */
        refcrd.x = coord.x + bestmv.x + dia_offs[mb_i].x;
        refcrd.y = coord.y + bestmv.y + dia_offs[mb_i].y;
        short2 trymv = bestmv + dia_offs[mb_i];
        int cost = sad_8x8_ii( fenc, coord, fref, refcrd ) +
                   lambda * calc_mv_cost( abs_diff( trymv.x, mvp.x ) << (2 + scale), abs_diff( trymv.y, mvp.y ) << (2 + scale) );

        cost_local[mb_i] = (cost<<2) | mb_i;
        cost = min( cost_local[0], min( cost_local[1], min( cost_local[2], cost_local[3] ) ) );

        if( (cost >> 2) >= bcost )
            break;

        bestmv += dia_offs[cost&3];
        bcost = cost>>2;

        if( bestmv.x >= mv_max.x || bestmv.x <= mv_min.x || bestmv.y >= mv_max.y || bestmv.y <= mv_min.y )
            break;
    }
    while( --me_range > 0 );

    short2 trymv;

#define COST_MV_NO_PAD( X, Y, L )\
    trymv.x = x264_clip3( X, mv_min.x, mv_max.x );\
    trymv.y = x264_clip3( Y, mv_min.y, mv_max.y );\
    if( abs_diff( mvp.x, trymv.x ) > 1 || abs_diff( mvp.y, trymv.y ) > 1 ) {\
        int2 refcrd = coord; refcrd.x += trymv.x; refcrd.y += trymv.y;\
        int cost = sad_8x8_ii_coop4( fenc, coord, fref, refcrd, mb_i, cost_local ) +\
                   L * calc_mv_cost( abs_diff( trymv.x, mvp.x ) << (2 + scale), abs_diff( trymv.y, mvp.y ) << (2 + scale) );\
        if( cost < bcost ) { bcost = cost; bestmv = trymv; } }

    COST_MV_NO_PAD( 0, 0, 0 );

    if( !b_first_iteration )
    {
        /* try cost at previous iteration's MV, if MVP was too far away */
        short2 prevmv;
        if( b_shift_index )
            prevmv = in_mvs[find_downscale_mb_xy( mb_x, mb_y, mb_width, mb_height )];
        else
            prevmv = in_mvs[mb_xy];
        prevmv.x >>= scale;
        prevmv.y >>= scale;
        COST_MV_NO_PAD( prevmv.x, prevmv.y, lambda );
    }

    for( int i = 0; i < i_mvc; i++ )
    {
        /* try cost at each candidate MV, if MVP was too far away */
        COST_MV_NO_PAD( mvc_local[i].x, mvc_local[i].y, lambda );
    }

    if( mb_i == 0 )
    {
        bestmv.x <<= scale;
        bestmv.y <<= scale;
        out_mvs[mb_xy] = bestmv;
        out_mv_costs[mb_xy] = X264_MIN( bcost, LOWRES_COST_MASK );
    }
}
