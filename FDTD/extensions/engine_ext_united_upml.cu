
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

        // volt and curr flux only needed on device side
        CudaHelper::Array<FDTD_FLOAT> *volt_flux = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, NULL);
        CudaHelper::Array<FDTD_FLOAT> *curr_flux = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, NULL);

        m_volt_fluxes.push_back(volt_flux);
        m_curr_fluxes.push_back(curr_flux);

        volt_flux->clear();
        curr_flux->clear();

        m_op_vv = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, op->vv.data());
        m_op_vvfn = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, op->vvfn.data());
        m_op_vvfo = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, op->vvfo.data());

        m_op_vv->load_to_device();
        m_op_vvfn->load_to_device();
        m_op_vvfo->load_to_device();

        m_op_ii = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, op->ii.data());
        m_op_iifn = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, op->iifn.data());
        m_op_iifo = new CudaHelper::Array<FDTD_FLOAT>(blk_size * 3, op->iifo.data());

        m_op_ii->load_to_device();
        m_op_iifn->load_to_device();
        m_op_iifo->load_to_device();

        block.start = dim3(op->m_StartPos[0], op->m_StartPos[1], op->m_StartPos[2]);
        block.lines = dim3(op->m_numLines[0], op->m_numLines[1], op->m_numLines[2]);
        block.volt_flux = volt_flux->device_data();
        block.curr_flux = curr_flux->device_data();

        block.vv = m_op_vv->device_data();
        block.vvfn = m_op_vvfn->device_data();
        block.vvfo = m_op_vvfo->device_data();
        block.ii = m_op_ii->device_data();
        block.iifn = m_op_iifn->device_data();
        block.iifo = m_op_iifo->device_data();

        // load all the data point for this block into the device memory
        checkCuda(cudaMemcpy(m_d_blks + i, &block, sizeof(upml_block_t), cudaMemcpyHostToDevice));
    }
}

__device__
int flat_indx_in_full_grid(int i, const upml_block_t *area, const dim3 *dim)
{
    int loc_x = i / (area->lines.y * area->lines.z);
    int loc_y = (i / area->lines.z) % area->lines.y;
    int loc_z = i % area->lines.z;

    int x = loc_x + area->start.x;
    int y = loc_y + area->start.y;
    int z = loc_z + area->start.z;  

    return (x * dim->y * dim->z + y * dim->z + z);
}

__global__ 
void PreVoltageUpdateKernel(FDTD_FLOAT *d_volt, const upml_block_t *ublk, const dim3 dim)
{
    // we use blockIdx.y to determine which block we are in, 
    // blockIdx.x and threadIdx.x to get the data index within that block
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z;
    int cell = blockIdx.x * blockDim.x + threadIdx.x;
    if (cell < N) {
        // find the cell location in whole matrix
        int gi = flat_indx_in_full_grid(cell, &area, &dim) * 3 + threadIdx.y;
        int i = cell * 3 + threadIdx.y;

        FDTD_FLOAT *volt = d_volt + gi;
        FDTD_FLOAT *flux = area.volt_flux + i;

        FDTD_FLOAT f = area.vv[i] * volt[0] - area.vvfo[i] * flux[0];
        volt[0] = flux[0];
        flux[0] = f;
    }
}

void Engine_Ext_United_UPML::DoPreVoltageUpdates(int threadID)
{
    if (threadID != 0)  return;

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size(); 
    int tBlockSizeX = (m_max_num_cells_in_block + 340) / 341;

    dim3 blk(tBlockSizeX, upml_blocks);
    dim3 thread(341, 3);

    PreVoltageUpdateKernel<<<blk, thread>>>(eng->GetDeviceVoltData(), m_d_blks, eng->GetDeviceDimData());

}


__global__
void PostVoltageUpdateKernel(FDTD_FLOAT *d_volt, const upml_block_t *ublk, const dim3 dim) 
{
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z * 3;

    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i / 3, &area, &dim) * 3 + (i % 3);

        FDTD_FLOAT *volt = d_volt + gi;
        FDTD_FLOAT vvfn = area.vvfn[i];
        FDTD_FLOAT *flux = area.volt_flux + i;

        FDTD_FLOAT f = flux[0];
        flux[0] = volt[0];
        volt[0] = f + vvfn * volt[0];
    }
}

void Engine_Ext_United_UPML::DoPostVoltageUpdates(int threadID)
{
    if (threadID != 0)  return;

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size(); 
    int tBlockSizeX = (m_max_num_cells_in_block * 3 + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    PostVoltageUpdateKernel<<<blk, 1024>>>(eng->GetDeviceVoltData(), m_d_blks, eng->GetDeviceDimData());
}

__global__ 
void PreCurrentUpdateKernel(FDTD_FLOAT *d_curr, const upml_block_t *ublk, const dim3 dim) 
{
    // we use blockIdx.y to determine which block we are in, 
    // blockIdx.x and threadIdx.x to get the data index within that block
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z * 3;
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        // find the cell location in whole matrix
        int gi = flat_indx_in_full_grid(i / 3, &area, &dim) * 3 + (i % 3);

        FDTD_FLOAT *curr = d_curr + gi;
        FDTD_FLOAT *flux = area.curr_flux + i;

        FDTD_FLOAT f = area.ii[i] * curr[0] - area.iifo[i] * flux[0];
        curr[0] = flux[0];
        flux[0] = f;
    }
}


void Engine_Ext_United_UPML::DoPreCurrentUpdates(int threadID)
{
    if (threadID != 0)  return;

    Engine_cuda *eng = static_cast<Engine_cuda*>(m_Eng);
    int upml_blocks = m_Op_UPML_List->size(); 

    int tBlockSizeX = (m_max_num_cells_in_block * 3 + 1023) / 1024;

    dim3 blk (tBlockSizeX, upml_blocks);

    PreCurrentUpdateKernel<<<blk, 1024>>>(eng->GetDeviceCurrData(), m_d_blks, eng->GetDeviceDimData());

}


__global__
void PostCurrentUpdateKernel(FDTD_FLOAT *d_curr, const upml_block_t *ublk, const dim3 dim)
{
    const upml_block_t area = ublk[blockIdx.y];
    int N = area.lines.x * area.lines.y * area.lines.z;

    for (auto i : hemi::grid_stride_range(0, N)) {
        int gi = flat_indx_in_full_grid(i, &area, &dim);

        FDTD_FLOAT *curr = d_curr + (gi * 3);
        FDTD_FLOAT *flux = area.curr_flux + (i * 3);
        FDTD_FLOAT *iifn = area.iifn + (i * 3);

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


