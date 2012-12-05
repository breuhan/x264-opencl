/*****************************************************************************
 * opencl.c: OpenCL initialization and kernel compilation
 *****************************************************************************
 * Copyright (C) 2010-2012 x264 project
 *
 * Authors: Steve Borho <sborho@multicorewareinc.com>
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02111, USA.
 *
 * This program is also available under a commercial proprietary license.
 * For more information, contact us at licensing@x264.com.
 *****************************************************************************/

#if HAVE_OPENCL
#include "common/common.h"
#include "encoder/oclobj.h"

/* Try to load the cached compiled program binary, verify the device context is
 * still valid before reuse */
static cl_program x264_opencl_cache_load( x264_t *h, char *devname, char *devvendor, char *driverversion )
{
    cl_program program = NULL;
    cl_int status;

    /* try to load cached program binary */
    FILE *fp = fopen( h->param.psz_clbin_file, "rb" );
    if( !fp )
        return NULL;

    fseek( fp, 0L, SEEK_END );
    size_t size = ftell( fp );
    rewind( fp );
    uint8_t *binary = x264_malloc( size );
    if( !binary )
        goto fail;

    fread( binary, 1, size, fp );
    const uint8_t *ptr = (const uint8_t*)binary;

#define CHECK_STRING( STR )\
    do {\
        size_t len = strlen( STR );\
        if( size <= len || strncmp( (char*)ptr, STR, len ) )\
            goto fail;\
        else {\
            size -= (len+1); ptr += (len+1);\
        }\
    } while( 0 )

    CHECK_STRING( devname );
    CHECK_STRING( devvendor );
    CHECK_STRING( driverversion );
#undef CHECK_STRING

    program = clCreateProgramWithBinary( h->opencl.context, 1, &h->opencl.device, &size, &ptr, NULL, &status );
    if( status != CL_SUCCESS )
        program = NULL;

fail:
    fclose( fp );
    x264_free( binary );
    return program;
}

/* Save the compiled program binary to a file for later reuse.  Device context
 * is also saved in the cache file so we do not reuse stale binaries */
static void x264_opencl_cache_save( x264_t *h, cl_program program, char *devname, char *devvendor, char *driverversion )
{
    FILE *fp = fopen( h->param.psz_clbin_file, "wb" );
    if( !fp )
        return;

    size_t size;
    cl_int status = clGetProgramInfo( program, CL_PROGRAM_BINARY_SIZES, sizeof(size_t), &size, NULL );
    if( status == CL_SUCCESS )
    {
        unsigned char *binary = x264_malloc( size );
        status = clGetProgramInfo( program, CL_PROGRAM_BINARIES, sizeof(unsigned char *), &binary, NULL );
        if( status == CL_SUCCESS )
        {
            fwrite( devname, 1, strlen( devname ), fp );
            fwrite( "\n", 1, 1, fp );
            fwrite( devvendor, 1, strlen( devvendor ), fp );
            fwrite( "\n", 1, 1, fp );
            fwrite( driverversion, 1, strlen( driverversion ), fp );
            fwrite( "\n", 1, 1, fp );
            fwrite( binary, 1, size, fp );
        }
        x264_free( binary );
    }
    fclose( fp );
}

static int x264_detect_AMD_GPU( cl_device_id device, int *b_is_SI )
{
    char extensions[512];
    char boardname[64];
    char devname[64];

    *b_is_SI = 0;

    cl_int status = clGetDeviceInfo( device, CL_DEVICE_EXTENSIONS, sizeof(extensions), extensions, NULL );
    if( status != CL_SUCCESS || !strstr( extensions, "cl_amd_media_ops" ) )
        return 0;

    /* Detect whether GPU is a SouthernIsland based device */
#define CL_DEVICE_BOARD_NAME_AMD                    0x4038
    status = clGetDeviceInfo( device, CL_DEVICE_BOARD_NAME_AMD, sizeof(boardname), boardname, NULL );
    if( status == CL_SUCCESS )
    {
        boardname[15] = boardname[16] = boardname[17] = 'X';
        *b_is_SI = !strcmp( boardname, "AMD Radeon HD 7XXX Series" );
        return 1;
    }

    /* Fall back to checking the device name */
    status = clGetDeviceInfo( device, CL_DEVICE_NAME, sizeof(devname), devname, NULL );
    if( status != CL_SUCCESS )
        return 1;

    const char *tahiti_names[] = { "Tahiti", "Pitcairn", "Capeverde", "Bali", NULL };
    for( int i = 0; tahiti_names[i]; i++ )
        if( !strcmp( devname, tahiti_names[i] ) )
        {
            *b_is_SI = 1;
            return 1;
        }
    return 1;
}

/* The OpenCL source under common/opencl will be merged into encoder/oclobj.h by
 * the Makefile. It defines a x264_opencl_source byte array which we will pass
 * to clCreateProgramWithSource().  We also attempt to use a cache file for the
 * compiled binary, stored in the current working folder.  */
static cl_program x264_opencl_compile( x264_t *h )
{
    cl_program program;
    cl_int status;

    char devname[64];
    char devvendor[64];
    char driverversion[64];
    status  = clGetDeviceInfo( h->opencl.device, CL_DEVICE_NAME,    sizeof(devname), devname, NULL );
    status |= clGetDeviceInfo( h->opencl.device, CL_DEVICE_VENDOR,  sizeof(devvendor), devvendor, NULL );
    status |= clGetDeviceInfo( h->opencl.device, CL_DRIVER_VERSION, sizeof(driverversion), driverversion, NULL );
    if( status != CL_SUCCESS )
        return NULL;

    int b_isamd = x264_detect_AMD_GPU( h->opencl.device, &h->opencl.b_device_AMD_SI );

    x264_log( h, X264_LOG_INFO, "OpenCL: %s %s %s\n", devvendor, devname, h->opencl.b_device_AMD_SI ? "(SI)" : "" );

    program = x264_opencl_cache_load( h, devname, devvendor, driverversion );
    if( !program )
    {
        /* clCreateProgramWithSource() requires a pointer variable, you cannot just use &x264_opencl_source */
        x264_log( h, X264_LOG_INFO, "Compiling OpenCL kernels...\n" );
        const char *strptr = (const char*)x264_opencl_source;
        size_t size = sizeof(x264_opencl_source);
        program = clCreateProgramWithSource( h->opencl.context, 1, &strptr, &size, &status );
        if( status != CL_SUCCESS || !program )
        {
            x264_log( h, X264_LOG_ERROR, "OpenCL: unable to create program\n" );
            return NULL;
        }
    }

    /* Build the program binary for the OpenCL device */
    const char *buildopts = "";
    if( b_isamd && !h->opencl.b_device_AMD_SI )
        buildopts = "-DVECTORIZE=1";
    status = clBuildProgram( program, 1, &h->opencl.device, buildopts, NULL, NULL );
    if( status == CL_SUCCESS )
    {
        x264_opencl_cache_save( h, program, devname, devvendor, driverversion );
        return program;
    }

    /* Compile failure, should not happen with production code. */

    size_t build_log_len = 0;

    status = clGetProgramBuildInfo( program, h->opencl.device, CL_PROGRAM_BUILD_LOG, build_log_len, NULL, &build_log_len );
    if( status != CL_SUCCESS )
    {
        x264_log( h, X264_LOG_ERROR, "OpenCL: Compilation failed, unable to get build log\n" );
        return NULL;
    }

    char *build_log = x264_malloc( build_log_len );
    if( !build_log )
        return NULL;

    status = clGetProgramBuildInfo( program, h->opencl.device, CL_PROGRAM_BUILD_LOG, build_log_len, build_log, NULL );
    if( status != CL_SUCCESS )
    {
        x264_log( h, X264_LOG_ERROR, "OpenCL: Compilation failed, unable to get build log\n" );
        x264_free( build_log );
        return NULL;
    }

    FILE *lg = fopen( "x264_kernel_build_log.txt", "w" );
    if( lg )
    {
        fwrite( build_log, 1, build_log_len, lg );
        fclose( lg );
        x264_log( h, X264_LOG_ERROR, "OpenCL: kernel build errors written to x264_kernel_build_log.txt\n" );
    }

    x264_free( build_log );
    return NULL;
}

static void x264_opencl_free_lookahead( x264_t *h )
{
#define RELEASE( a, f ) if( a ) f( a );
    RELEASE( h->opencl.intra_kernel, clReleaseKernel )
    RELEASE( h->opencl.rowsum_intra_kernel, clReleaseKernel )
    RELEASE( h->opencl.downscale_kernel1, clReleaseKernel )
    RELEASE( h->opencl.downscale_kernel2, clReleaseKernel )
    RELEASE( h->opencl.downscale_hpel_kernel, clReleaseKernel )
    RELEASE( h->opencl.weightp_hpel_kernel, clReleaseKernel )
    RELEASE( h->opencl.weightp_scaled_images_kernel, clReleaseKernel )
    RELEASE( h->opencl.memset_kernel, clReleaseKernel )
    RELEASE( h->opencl.hme_kernel, clReleaseKernel )
    RELEASE( h->opencl.subpel_refine_kernel, clReleaseKernel )
    RELEASE( h->opencl.mode_select_kernel, clReleaseKernel )
    RELEASE( h->opencl.rowsum_inter_kernel, clReleaseKernel )
    RELEASE( h->opencl.lookahead_program, clReleaseProgram )
    RELEASE( h->opencl.row_satds[0], clReleaseMemObject )
    RELEASE( h->opencl.row_satds[1], clReleaseMemObject )
    RELEASE( h->opencl.frame_stats[0], clReleaseMemObject )
    RELEASE( h->opencl.frame_stats[1], clReleaseMemObject )
    RELEASE( h->opencl.mv_buffers[0], clReleaseMemObject )
    RELEASE( h->opencl.mv_buffers[1], clReleaseMemObject )
    RELEASE( h->opencl.mvp_buffer, clReleaseMemObject )
    RELEASE( h->opencl.luma_16x16_image[0], clReleaseMemObject )
    RELEASE( h->opencl.luma_16x16_image[1], clReleaseMemObject )
    RELEASE( h->opencl.lowres_mv_costs, clReleaseMemObject )
    RELEASE( h->opencl.lowres_costs[0], clReleaseMemObject )
    RELEASE( h->opencl.lowres_costs[1], clReleaseMemObject )
    RELEASE( h->opencl.page_locked_buffer, clReleaseMemObject )
    RELEASE( h->opencl.weighted_luma_hpel, clReleaseMemObject )
    for( int i = 0; i < NUM_IMAGE_SCALES; i++ )
    {
        RELEASE( h->opencl.weighted_scaled_images[i], clReleaseMemObject )
    }
#undef RELEASE
}

static int x264_opencl_init_lookahead( x264_t *h )
{
    if( h->param.rc.i_lookahead == 0 )
        return -1;

    char *kernelnames[] = {
        "mb_intra_cost_satd_8x8",
        "sum_intra_cost",
        "downscale_hpel",
        "downscale1",
        "downscale2",
        "memset_int16",
        "weightp_scaled_images",
        "weightp_hpel",
        "hierarchical_motion",
        "subpel_refine",
        "mode_selection",
        "sum_inter_cost"
    };
    cl_kernel *kernels[] = {
        &h->opencl.intra_kernel,
        &h->opencl.rowsum_intra_kernel,
        &h->opencl.downscale_hpel_kernel,
        &h->opencl.downscale_kernel1,
        &h->opencl.downscale_kernel2,
        &h->opencl.memset_kernel,
        &h->opencl.weightp_scaled_images_kernel,
        &h->opencl.weightp_hpel_kernel,
        &h->opencl.hme_kernel,
        &h->opencl.subpel_refine_kernel,
        &h->opencl.mode_select_kernel,
        &h->opencl.rowsum_inter_kernel
    };
    cl_int status;

    h->opencl.lookahead_program = x264_opencl_compile( h );
    if( !h->opencl.lookahead_program )
    {
        x264_opencl_free_lookahead( h );
        return -1;
    }

    for( int i = 0; i < sizeof(kernelnames)/sizeof(char*); i++ )
    {
        *kernels[i] = clCreateKernel( h->opencl.lookahead_program, kernelnames[i], &status );
        if( status != CL_SUCCESS )
        {
            x264_log( h, X264_LOG_ERROR, "Unable to compile kernel '%s' (%d)\n", kernelnames[i], status );
            x264_opencl_free_lookahead( h );
            return -1;
        }
    }

    h->opencl.page_locked_buffer = clCreateBuffer( h->opencl.context, CL_MEM_WRITE_ONLY|CL_MEM_ALLOC_HOST_PTR, PAGE_LOCKED_BUF_SIZE, NULL, &status );
    if( status != CL_SUCCESS )
    {
        x264_log( h, X264_LOG_ERROR, "Unable to allocate page-locked buffer, error '%d'\n", status );
        return -1;
    }
    h->opencl.page_locked_ptr = clEnqueueMapBuffer( h->opencl.queue, h->opencl.page_locked_buffer, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE,
                                                    0, PAGE_LOCKED_BUF_SIZE, 0, NULL, NULL, &status );
    if( status != CL_SUCCESS )
    {
        x264_log( h, X264_LOG_ERROR, "Unable to map page-locked buffer, error '%d'\n", status );
        return -1;
    }

    return 0;
}

static void  x264_opencl_error_notify( const char *errinfo, const void *private_info, size_t cb, void *user_data )
{
    x264_t *h = (x264_t*)user_data;
    x264_log( h, X264_LOG_ERROR, "OpenCL: %s\n", errinfo );
}

static int x264_opencl_init( x264_t *h )
{
    cl_int status;
    cl_uint numPlatforms;
    int ret = -1;

    status = clGetPlatformIDs( 0, NULL, &numPlatforms );
    if( status != CL_SUCCESS || numPlatforms == 0 )
        return -1;

    cl_platform_id *platforms = (cl_platform_id*)x264_malloc( numPlatforms * sizeof(cl_platform_id) );
    status = clGetPlatformIDs( numPlatforms, platforms, NULL );
    if( status != CL_SUCCESS )
    {
        x264_free( platforms );
        return -1;
    }

    /* Select first OpenCL platform that supports a GPU device */
    for( cl_uint i = 0; i < numPlatforms; ++i )
    {
        cl_uint gpu_count;
        status = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, 0, NULL, &gpu_count );
        if( status == CL_SUCCESS && gpu_count > 0 )
        {
            /* take GPU 0 */
            status = clGetDeviceIDs( platforms[i], CL_DEVICE_TYPE_GPU, 1, &h->opencl.device, NULL );
            if( status != CL_SUCCESS )
                continue;

            h->opencl.context = clCreateContext( NULL, 1, &h->opencl.device, (void*)x264_opencl_error_notify, (void*)h, &status );
            if( status != CL_SUCCESS )
                continue;

            cl_bool image_support;
            clGetDeviceInfo( h->opencl.device, CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &image_support, NULL );
            if( !image_support )
                continue;

            cl_uint count = 0;
            cl_image_format imageType[100];
            clGetSupportedImageFormats( h->opencl.context, CL_MEM_READ_WRITE, CL_MEM_OBJECT_IMAGE2D, 100, imageType, &count );
            count = X264_MIN( count, 100 );

            int b_has_r = 0;
            int b_has_rgba = 0;
            for( cl_uint j = 0; j<count; j++ )
            {
                if( imageType[j].image_channel_order == CL_R )
                    b_has_r = 1;
                else if( imageType[j].image_channel_order == CL_RGBA )
                    b_has_rgba = 1;
            }
            if( !b_has_r || !b_has_rgba )
                continue;

            h->opencl.queue = clCreateCommandQueue( h->opencl.context, h->opencl.device, 0, &status );
            if( status != CL_SUCCESS )
                continue;

            ret = 0;
            break;
        }
    }

    x264_free( platforms );

    if( !h->param.psz_clbin_file )
        h->param.psz_clbin_file = "x264_lookahead.clbin";

    if( !ret )
        ret = x264_opencl_init_lookahead( h );

    return ret;
}

void x264_opencl_free( x264_t *h )
{
    x264_opencl_free_lookahead( h );

    if( h->opencl.queue )
        clReleaseCommandQueue( h->opencl.queue );
    if( h->opencl.context )
        clReleaseContext( h->opencl.context );
}
#endif /* HAVE_OPENCL */
