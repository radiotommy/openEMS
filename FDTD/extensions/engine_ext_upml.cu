

#include "engine_ext_upml.h"
#include "operator_ext_upml.h"
#include "FDTD/engine_cuda.h"

#include <cuda_runtime.h>
#include "hemi/hemi_error.h"
#include "hemi/grid_stride_range.h"


void Engine_Ext_UPML::SetEngine(Engine* eng) 
{
    m_Eng = eng;
    if (eng->GetType() != Engine::CUDA) {
        return;
    }

    // call it area to avoid confusion with cuda blocks
    upml_block_t area;

    memcpy(area.start, m_Op_UPML->m_StartPos, sizeof(area.start));
    memcpy(area.lines, m_Op_UPML->m_numLines, sizeof(area.lines));

    //int size = 3 * area.lines[0] * area.lines[1] * area.lines[2] * sizeof(FDTD_FLOAT);
    fprintf(stderr, "preare upml: %d,%d,%d, cells = %d,%d,%d/%d\n", 
            area.start[0], area.start[1], area.start[2],
            area.lines[0],  area.lines[1], area.lines[2],
            m_Op_UPML->vv.size()
        );
    fflush(stderr);

    checkCuda(cudaMalloc(&d_area, sizeof(area)));
    checkCuda(cudaMemcpy(d_area, &area, sizeof(area), cudaMemcpyHostToDevice));

    // load data to gpu
    volt_flux.load();
    curr_flux.load();

    m_Op_UPML->vv.load();
    m_Op_UPML->vvfn.load();
    m_Op_UPML->vvfo.load();

    m_Op_UPML->ii.load();
    m_Op_UPML->iifn.load();
    m_Op_UPML->iifo.load();
}


__device__
int flat_indx_in_full_grid(int i, const upml_block_t *area, const int *dim)
{
    int loc_x = i / (area->lines[1] * area->lines[2]);
    int loc_y = (i / area->lines[2]) % area->lines[1];
    int loc_z = i % area->lines[2];

    int x = loc_x + area->start[0];
    int y = loc_y + area->start[1];
    int z = loc_z + area->start[2];  

    return (x * dim[1] * dim[2] + y * dim[2] + z);
}

__global__ 
void PreVoltageUpdateKernel(FDTD_FLOAT *d_volt, FDTD_FLOAT *d_flux, 
    FDTD_FLOAT *d_vv, FDTD_FLOAT *d_vvfo, 
    const upml_block_t *area, const int *dim, int N)
{
    for (auto i : hemi::grid_stride_range(0, N)) {
        // find the cell location in whole matrix
        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *volt = d_volt + (gi * 3);
        FDTD_FLOAT *vv = d_vv + i * 3;
        FDTD_FLOAT *vvfo = d_vvfo + i * 3;
        FDTD_FLOAT *flux = d_flux + (i * 3);


        FDTD_FLOAT f = vv[0] * volt[0] - vvfo[0] * flux[0];
        volt[0] = flux[0];
        flux[0] = f;

        f = vv[1] * volt[1] - vvfo[1] * flux[1];
        volt[1] = flux[1];
        flux[1] = f;


        f = vv[2] * volt[2] - vvfo[2] * flux[2];
        volt[2] = flux[2];
        flux[2] = f;
    }
}

void Engine_Ext_UPML::DoPreVoltageUpdatesCuda(FDTD_FLOAT *d_volt, const int *d_dim)
{
    int N =  m_Op_UPML->m_numLines[0] * m_Op_UPML->m_numLines[1] * m_Op_UPML->m_numLines[2];
    int blks = (N + 1023) /1024;

    int threads = std::min(1024, N);
    PreVoltageUpdateKernel<<<blks, threads>>>(
        d_volt, 
        volt_flux.device_data(),
        m_Op_UPML->vv.device_data(),
        m_Op_UPML->vvfo.device_data(),
        d_area, d_dim, N);

    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());
}



__global__
void PostVoltageUpdateKernel(FDTD_FLOAT *d_volt, FDTD_FLOAT *d_flux, 
    FDTD_FLOAT *d_vvfn, 
    const upml_block_t *area, const int *dim, int N)
{
    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *volt = d_volt + (gi * 3);
        FDTD_FLOAT *vvfn = d_vvfn + (i * 3);
        FDTD_FLOAT *flux = d_flux + (i * 3);

        FDTD_FLOAT f = flux[0];
        flux[0] = volt[0];
        volt[0] = f + vvfn[0] * flux[0];


        f = flux[1];
        flux[1] = volt[1];
        volt[1] = f + vvfn[1] * flux[1];

        f = flux[2];
        flux[2] = volt[2];
        volt[2] = f + vvfn[2] * flux[2];
    }
}



void Engine_Ext_UPML::DoPostVoltageUpdatesCuda(FDTD_FLOAT *d_volt, const int *d_dim)
{
    int N =  m_Op_UPML->m_numLines[0] * m_Op_UPML->m_numLines[1] * m_Op_UPML->m_numLines[2];
    int blks = (N + 1023) /1024;

    int threads = std::min(1024, N);

    PostVoltageUpdateKernel<<<blks, threads>>>(
        d_volt, 
        volt_flux.device_data(),
        m_Op_UPML->vvfn.device_data(),
        d_area, d_dim, N);

    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());
}


__global__ 
void PreCurrentUpdateKernel(FDTD_FLOAT *d_curr, FDTD_FLOAT *d_flux, 
    FDTD_FLOAT *d_ii, FDTD_FLOAT *d_iifo, 
    const upml_block_t *area, const int *dim, int N)
{
    //printf( "upml pre voltage update at %d,%d,%d\n", area->start[0], area->start[1], area->start[2]);
    for (auto i : hemi::grid_stride_range(0, N)) {
        // find the cell location in whole matrix

        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *curr = d_curr + (gi * 3);
        FDTD_FLOAT *ii = d_ii + i * 3;
        FDTD_FLOAT *iifo = d_iifo + i * 3;
        FDTD_FLOAT *flux = d_flux + (i * 3);


        FDTD_FLOAT f = ii[0] * curr[0] - iifo[0] * flux[0];
        curr[0] = flux[0];
        flux[0] = f;

        f = ii[1] * curr[1] - iifo[1] * flux[1];
        curr[1] = flux[1];
        flux[1] = f;

        f = ii[2] * curr[2] - iifo[2] * flux[2];
        curr[2] = flux[2];  
        flux[2] = f;
    }

}

void Engine_Ext_UPML::DoPreCurrentUpdatesCuda(FDTD_FLOAT *d_curr, const int *d_dim)
{
    int N =  m_Op_UPML->m_numLines[0] * m_Op_UPML->m_numLines[1] * m_Op_UPML->m_numLines[2];
    int blks = (N + 1023) /1024;

    int threads = std::min(1024, N);
    PreCurrentUpdateKernel<<<blks, threads>>>(d_curr,
        curr_flux.device_data(),
        m_Op_UPML->ii.device_data(),
        m_Op_UPML->iifo.device_data(),
        d_area, d_dim, N);
        
    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());
}


__global__
void PostCurrentUpdateKernel(FDTD_FLOAT *d_curr, FDTD_FLOAT *d_flux, 
    FDTD_FLOAT *d_iifn, 
    const upml_block_t *area, const int *dim, int N)
{
    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *curr = d_curr + (gi * 3);
        FDTD_FLOAT *flux = d_flux + (i * 3);
        FDTD_FLOAT *iifn = d_iifn + (i * 3);

        FDTD_FLOAT f = flux[0];
        flux[0] = curr[0];
        curr[0] = f + iifn[0] * flux[0];

        f = flux[1];
        flux[1] = curr[1];
        curr[1] = f + iifn[1] * flux[1];

        f = flux[2];
        flux[2] = curr[2];
        curr[2] = f + iifn[2] * flux[2];
    }
}



void Engine_Ext_UPML::DoPostCurrentUpdatesCuda(FDTD_FLOAT *d_curr, const int *d_dim)
{
    int N =  m_Op_UPML->m_numLines[0] * m_Op_UPML->m_numLines[1] * m_Op_UPML->m_numLines[2];
    int blks = (N + 1023) /1024;

    int threads = std::min(1024, N);
    PostCurrentUpdateKernel<<<blks, threads>>>(d_curr, 
        curr_flux.device_data(),
        m_Op_UPML->iifn.device_data(),
        d_area,
        d_dim,
        N); 
    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());
}



