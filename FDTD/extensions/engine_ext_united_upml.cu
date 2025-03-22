
#include "engine_ext_united_upml.h"
#include "operator_ext_upml.h"
#include "tools/arraylib/array_nijk.h"



#include <cuda_runtime.h>
#include "hemi/hemi_error.h"
#include "hemi/grid_stride_range.h"



void Engine_Ext_United_UPML::SetEngine(Engine* eng)
{
    m_Eng = eng;
    if (eng->GetType() != Engine::CUDA) {
        return;
    }

    int upml_blocks = m_Op_UPML_List->size();

    printf("load united upml blocks %d\n", upml_blocks);
    fflush(stdout);

    checkCuda(cudaMalloc(&m_d_blks, sizeof(upml_block_t) * upml_blocks));

    m_num_of_cells_in_block = new int[upml_blocks];
    m_max_num_cells_in_block = 0;

    for (auto i = 0; i < upml_blocks; i++)
    {
        Operator_Ext_UPML *op = m_Op_UPML_List->at(i);
        upml_block_t block;
        int blk_size = op->m_numLines[0] * op->m_numLines[1] * op->m_numLines[2];


        m_num_of_cells_in_block[i] = blk_size;
        if (blk_size > m_max_num_cells_in_block) {
            m_max_num_cells_in_block = blk_size;
        }


        ArrayLib::ArrayNIJK<FDTD_FLOAT> *volt_flux = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("vflux", op->m_numLines);
        ArrayLib::ArrayNIJK<FDTD_FLOAT> *curr_flux = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("cflow", op->m_numLines);
        m_volt_fluxes.push_back(volt_flux);
        m_curr_fluxes.push_back(curr_flux);

        volt_flux->load();
        curr_flux->load();

        op->vv.load();
        op->vvfn.load();
        op->vvfo.load();

        op->ii.load();
        op->iifn.load();
        op->iifo.load();

        memcpy(block.start, op->m_StartPos, sizeof(block.start));
        memcpy(block.lines, op->m_numLines, sizeof(block.lines));
        block.volt_flux = volt_flux->device_data();
        block.curr_flux = curr_flux->device_data();

        block.vv = op->vv.device_data();
        block.vvfn = op->vvfn.device_data();
        block.vvfo = op->vvfo.device_data();
        block.ii = op->ii.device_data();
        block.iifn = op->iifn.device_data();
        block.iifo = op->iifo.device_data();

        // load all the data point for this block into the device memory
        checkCuda(cudaMemcpy(m_d_blks + i, &block, sizeof(upml_block_t), cudaMemcpyHostToDevice));
    }
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
void PreVoltageUpdateKernel(FDTD_FLOAT *d_volt, const upml_block_t *ublk, const int *dim)
{
    // we use blockIdx.y to determine which block we are in, 
    // blockIdx.x and threadIdx.x to get the data index within that block

    const upml_block_t *area = &ublk[blockIdx.y];
    int N = area->lines[0] * area->lines[1] * area->lines[2];

    for (auto i : hemi::grid_stride_range(0, N)) {
        // find the cell location in whole matrix
        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *volt = d_volt + (gi * 3);
        FDTD_FLOAT *vv = area->vv + i * 3;
        FDTD_FLOAT *vvfo = area->vvfo + i * 3;
        FDTD_FLOAT *flux = area->volt_flux + i * 3;


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

void Engine_Ext_United_UPML::DoPreVoltageUpdates(int threadID)
{
    if (threadID != 0)  return;

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size(); 
    int tBlockSizeX = (m_max_num_cells_in_block + 1023) / 1024;

    dim3 blk(tBlockSizeX, upml_blocks);

    PreVoltageUpdateKernel<<<blk, 1024>>>(eng->GetDeviceVoltData(), m_d_blks, eng->GetDeviceDimData());

}


__global__
void PostVoltageUpdateKernel(FDTD_FLOAT *d_volt, const upml_block_t *ublk, const int *dim) 
{
    const upml_block_t *area = &ublk[blockIdx.y];
    int N = area->lines[0] * area->lines[1] * area->lines[2];

    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *volt = d_volt + (gi * 3);
        FDTD_FLOAT *vvfn = area->vvfn + (i * 3);
        FDTD_FLOAT *flux = area->volt_flux + (i * 3);

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

void Engine_Ext_United_UPML::DoPostVoltageUpdates(int threadID)
{
    if (threadID != 0)  return;

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size(); 
    int tBlockSizeX = (m_max_num_cells_in_block + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    PostVoltageUpdateKernel<<<blk, 1024>>>(eng->GetDeviceVoltData(), m_d_blks, eng->GetDeviceDimData());

}

__global__ 
void PreCurrentUpdateKernel(FDTD_FLOAT *d_curr, const upml_block_t *ublk, const int *dim) 
{
    const upml_block_t *area = &ublk[blockIdx.y];
    int N = area->lines[0] * area->lines[1] * area->lines[2];

    for (auto i : hemi::grid_stride_range(0, N)) {
        // find the cell location in whole matrix

        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *curr = d_curr + (gi * 3);
        FDTD_FLOAT *ii = area->ii + i * 3;
        FDTD_FLOAT *iifo = area->iifo + i * 3;
        FDTD_FLOAT *flux = area->curr_flux + (i * 3);

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


void Engine_Ext_United_UPML::DoPreCurrentUpdates(int threadID)
{
    if (threadID != 0)  return;

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size(); 
    int tBlockSizeX = (m_max_num_cells_in_block + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    PreCurrentUpdateKernel<<<blk, 1024>>>(eng->GetDeviceCurrData(), m_d_blks, eng->GetDeviceDimData());

}


__global__
void PostCurrentUpdateKernel(FDTD_FLOAT *d_curr, const upml_block_t *ublk, const int *dim)
{
    const upml_block_t *area = &ublk[blockIdx.y];
    int N = area->lines[0] * area->lines[1] * area->lines[2];

    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i, area, dim);

        FDTD_FLOAT *curr = d_curr + (gi * 3);
        FDTD_FLOAT *flux = area->curr_flux + (i * 3);
        FDTD_FLOAT *iifn = area->iifn + (i * 3);

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

void Engine_Ext_United_UPML::DoPostCurrentUpdates(int threadID)
{
    if (threadID != 0)  return;

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size(); 
    int tBlockSizeX = (m_max_num_cells_in_block + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    PostCurrentUpdateKernel<<<blk, 1024>>>(eng->GetDeviceCurrData(), m_d_blks, eng->GetDeviceDimData());

}


