
/*
 * Copyright (C) 2010 Thorsten Liebig (Thorsten.Liebig@gmx.de)
 * [License text omitted for brevity, remains unchanged]
 */

#include "engine_cuda.h"
#include "extensions/engine_extension.h"
#include "extensions/operator_extension.h"
#include "tools/array_ops.h"

#include <cooperative_groups.h>

#include <cuda_runtime.h>
#include <iostream>

#include "hemi/grid_stride_range.h"
#include "hemi/launch.h"

#include "tools/cuda/check.h"

using namespace std;


#define USE_UNIFIED_MEM

#define flat_index(X, Y, Z, D)     ((X) * D[1] * D[2] + ((Y) * D[2]) + (Z))
#define THREADS     1024



// Kernel for voltage updates with flat arrays
__global__
void updateVoltagesKernel(FDTD_FLOAT *volt, const FDTD_FLOAT *curr, const FDTD_FLOAT *op_vv_vi, int N, dim3 dim)
{
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = cell * 3;
        int x = cell / (dim.y * dim.z);
        int y = (cell / dim.z) % dim.y;
        int z = cell % dim.z;

        const FDTD_FLOAT* ix = curr + ((x != 0) ? offs - dim.y * dim.z * 3 : offs);
        const FDTD_FLOAT* iy = curr + ((y != 0) ? offs - dim.z * 3 : offs);
        const FDTD_FLOAT* iz = curr + ((z != 0) ? offs - 3 : offs);

        float2 vvvi[3];
        op_vv_vi += offs * 2;

        vvvi[0] = *(float2*)(op_vv_vi);
        vvvi[1] = *(float2*)(op_vv_vi + 2);
        vvvi[2] = *(float2*)(op_vv_vi + 4);

        FDTD_FLOAT i[3];
        const FDTD_FLOAT *p = curr + offs;
        i[0]= *p++;
        i[1] = *p++;
        i[2] = *p++;

        // nbr cells in x, y, z direction
        vvvi[0].y *= (i[2] - iy[2] - i[1] + iz[1]);
        vvvi[1].y *= (i[0] - iz[0] - i[2] + ix[2]);
        vvvi[2].y *= (i[1] -ix[1] - i[0] + iy[0]); 

        // update x
        FDTD_FLOAT* v= volt + offs;
        v[0]  = v[0] * vvvi[0].x + vvvi[0].y; 
        // update y
        v[1] = v[1] * vvvi[1].x + vvvi[1].y;
        // update z
        v[2] = v[2] * vvvi[2].x +  vvvi[2].y;
    }
}


// Kernel for current updates
__global__
void updateCurrentsKernel(FDTD_FLOAT *curr, const FDTD_FLOAT *volt, const FDTD_FLOAT *op_ii_iv, int N, dim3 dim)
{
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = cell * 3;
        int y = (cell / dim.z) % dim.y;
        int z = cell % dim.z;

        if ((y < dim.y - 1) && (z < dim.z - 1)) {

            volt += offs;
            // next nbr cells in x, y, z direction
            const FDTD_FLOAT* vx = volt + (dim.y * dim.z * 3);
            const FDTD_FLOAT* vy = volt + (3 * dim.z);
            const FDTD_FLOAT* vz = volt + 3;

            float2 iiiv[3];
            op_ii_iv += offs * 2;

            iiiv[0] = *(float2 *)(&op_ii_iv[0]);
            iiiv[1] = *(float2 *)(&op_ii_iv[2]);
            iiiv[2] = *(float2 *)(&op_ii_iv[4]);

            FDTD_FLOAT v[3];
            v[0] = volt[0];
            v[1] = volt[1];
            v[2] = volt[2];

            iiiv[0].y *= (v[2] - vy[2] - v[1] + vz[1]);
            iiiv[1].y *= (v[0] - vz[0] - v[2] + vx[2]);
            iiiv[2].y *= (v[1] - vx[1] - v[0] + vy[0]);

            FDTD_FLOAT* i = curr + offs;
            // update x
            i[0] = i[0] * iiiv[0].x + iiiv[0].y;
            // update y
            i[1] = i[1] * iiiv[1].x + iiiv[1].y;
            // update z
            i[2] = i[2] * iiiv[2].x + iiiv[2].y;
        }
    }
}

__global__
void addInKernel(FDTD_FLOAT *p, int cell, int n, FDTD_FLOAT val)
{
    p[cell * 3 + n] += val;
}

__device__ void wrapReduce(volatile double *vv, volatile double *ii, int tid)
{
    vv[tid] += vv[(tid + 32)];
    ii[tid] += ii[(tid + 32)];
    vv[tid] += vv[(tid + 16)];
    ii[tid] += ii[(tid + 16)];
    vv[tid] += vv[(tid + 8)];
    ii[tid] += ii[(tid + 8)];
    vv[tid] += vv[(tid + 4)];
    ii[tid] += ii[(tid + 4)];
    vv[tid] += vv[(tid + 2)];
    ii[tid] += ii[(tid + 2)];
    vv[tid] += vv[(tid + 1)];
    ii[tid] += ii[(tid + 1)];
}

__global__ void calcFastEnergyKernel(FDTD_FLOAT *volt, FDTD_FLOAT *curr, double *p_sum, dim3 dim)
{
    __shared__ double vv[THREADS]; // shared memory for energy calculation, between threads in a block
    __shared__ double ii[THREADS]; // shared memory for energy calculation, between threads in a block

    double local_vv = 0.0;
    double local_ii = 0.0;
    int tid = threadIdx.x;

    // Process all cells in strides
    int total = dim.x * dim.y * dim.z;
    for (int cell = tid; cell < total; cell += THREADS) {
        int x = cell / (dim.y * dim.z);
        int y = (cell / dim.z) % dim.y;
        int z = cell % dim.z;
        if (x < dim.x - 1 && y < dim.y - 1 && z < dim.z - 1) {
            int offs = cell * 3;
            FDTD_FLOAT *v = volt + offs;
            FDTD_FLOAT *i = curr + offs;

            local_vv += v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
            local_ii += i[0] * i[0] + i[1] * i[1] + i[2] * i[2];
        }
    }
    vv[tid] = local_vv;
    ii[tid] = local_ii;
    __syncthreads();

    // Reduction
    for (int s = THREADS / 2; s > 32; s >>= 1) {
        if (tid < s) {
            vv[tid] += vv[tid + s];
            ii[tid] += ii[tid + s];
        }
        __syncthreads();
    }
    if (tid < 32) {
        wrapReduce(vv, ii, tid);
    }
    __syncthreads();

    if (tid == 0) {
        p_sum[0] = vv[0] * FLT_EPSILON + ii[0] * MUE0;
    }
}


Engine_cuda* Engine_cuda::New(const Operator_CUDA* op, unsigned int cuda_device_number) {
    cout << "Create CUDA FDTD engine" << endl;
    Engine_cuda* e = new Engine_cuda(op);
    e->setCUDAdevice(cuda_device_number);
    e->Init();
    return e;
}

Engine_cuda::Engine_cuda(const Operator_CUDA* op)  : Engine(op)
{
    m_type = CUDA;
    numTS = 0;
    Op = op;

    cout << "Engine CUDA construct type=" << m_type << endl;
}

void Engine_cuda::setCUDAdevice(unsigned int cuda_device_number)	   
{
    m_cuda_device_number = cuda_device_number;
}


Engine_cuda::~Engine_cuda()
{
    this->Reset();
}



static void load_data_pair(FDTD_FLOAT *dest, const FDTD_FLOAT *a, const FDTD_FLOAT *b, int num_cells)
{
    FDTD_FLOAT *buf;

    checkCuda(cudaMallocHost(&buf, num_cells * 6 * sizeof(FDTD_FLOAT)));

    FDTD_FLOAT *p = buf;
    for (int i = 0; i < num_cells; i++) {
        *p++ = *a++; *p++ = *b++; 
        *p++ = *a++; *p++ = *b++; 
        *p++ = *a++; *p++ = *b++;
    }

    checkCuda(cudaMemcpy(dest, buf, num_cells * 6 * sizeof(FDTD_FLOAT), cudaMemcpyHostToDevice));

    cudaFreeHost(buf);

}

void Engine_cuda::Init() {
	   

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices <= 0) {
        throw std::runtime_error("NO CUDA device found");
    }
    if (m_cuda_device_number >= nDevices) {
        cout << "cuda device out of range " << m_cuda_device_number << " / " << nDevices << endl;
        throw std::runtime_error("CUDA device number out of range");
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_cuda_device_number);
    cout << "  Running on device: " << prop.name << endl;
    cout << "    max block dimensions " \
        << prop.maxThreadsDim[0] << "," \
        << prop.maxThreadsDim[1] << "," \
        << prop.maxThreadsDim[2] << endl;

    cout << "    max block dimensions " \
        << prop.maxGridSize[0] << "," \
        << prop.maxGridSize[1] << "," \
        << prop.maxGridSize[2] << endl;
   
    cudaSetDevice(m_cuda_device_number);
    cudaDeviceGetAttribute(&m_supports_coop_launch, cudaDevAttrCooperativeLaunch, m_cuda_device_number);

    m_dim = dim3(numLines[0], numLines[1], numLines[2]);

	numTS = 0;
    int num_cells = numLines[0] * numLines[1] * numLines[2];

    printf("simulation dim: %d, %d, %d\n", m_dim.x, m_dim.y, m_dim.z); fflush(stdout);

    // Allocate GPU memory
	volt_array = new CudaHelper::Array<FDTD_FLOAT>(num_cells * 3);
	curr_array = new CudaHelper::Array<FDTD_FLOAT>(num_cells * 3);
    m_energy_sum =  new CudaHelper::Array<double>(1);

    volt_array->clear();
    curr_array->clear();

    checkCuda(cudaMalloc(&d_op_vv_vi, 6 * num_cells * sizeof(FDTD_FLOAT)));
    checkCuda(cudaMalloc(&d_op_ii_iv, 6 * num_cells * sizeof(FDTD_FLOAT)));

    load_data_pair(d_op_vv_vi, Op->vv_ptr->data(), Op->vi_ptr->data(), num_cells);
    load_data_pair(d_op_ii_iv, Op->ii_ptr->data(), Op->iv_ptr->data(), num_cells);

    InitExtensions();
    SortExtensionByPriority();
}

void Engine_cuda::Reset() {
    if (volt_array)     delete volt_array;
    if (curr_array)     delete curr_array;
    if (m_energy_sum)   delete m_energy_sum;

    ClearExtensions();
}


void Engine_cuda::UpdateVoltages(unsigned int startX, unsigned int numX) {
    // Copy current data to GPU
    int N = numX * numLines[1] * numLines[2];

    // Launch kernel
    int blocks = (N  + THREADS - 1) / THREADS;
    dim3 dim(numLines[0], numLines[1], numLines[2]);
    updateVoltagesKernel<<< blocks, THREADS >>>(volt_array->device_data(), (const FDTD_FLOAT*)curr_array->device_data(), 
            d_op_vv_vi, N, dim);

    //checkCudaErrors();
}

void Engine_cuda::UpdateCurrents(unsigned int startX, unsigned int numX) {
    // Copy voltage data to GPU

    int N = numX * numLines[1] * numLines[2];

    // Launch kernel
    int blocks = (N  + THREADS - 1) / THREADS;
    dim3 dim(numLines[0], numLines[1], numLines[2]);

    updateCurrentsKernel<<< blocks, THREADS >>>(curr_array->device_data(), (const FDTD_FLOAT*)volt_array->device_data(), 
            d_op_ii_iv, N, dim);
    //checkCudaErrors();
}

void Engine_cuda::AddVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int cell = flat_index(pos[0], pos[1], pos[2], numLines);
    addInKernel<<< 1, 1>>>(volt_array->device_data(), cell, n, value);

    //checkCudaErrors();
}

void Engine_cuda::AddCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int cell = flat_index(pos[0], pos[1], pos[2], numLines);
    addInKernel<<<1, 1>>>(curr_array->device_data(), cell, n, value);

    //checkCudaErrors();
}


double Engine_cuda::CalcFastEnergy()
{
    if (m_volt_updated || m_curr_updated) {
        printf("volt & curr updated: %d,%d\n", m_volt_updated, m_curr_updated);
    }
    calcFastEnergyKernel<<<1, THREADS>>>(volt_array->device_data(), curr_array->device_data(), m_energy_sum->device_data(), m_dim);

    CudaHelper::check_cuda();
    //checkCuda(cudaDeviceSynchronize());

    m_energy_sum->load_to_host();
    return m_energy_sum->host_data()[0];
}

// Other methods (InitExtensions, SortExtensionByPriority, etc.) remain unchanged unless extensions need CUDA support

bool Engine_cuda::IterateTS(unsigned int iterTS) {
    // TODO: run all extentions in cuda
    // load data to cuda
    // set extension like a pipeline?
    // load data back

    m_volt_updated_by_host = 1;
    m_curr_updated_by_host = 1;

    if (m_volt_updated) {
        printf("load volt to cuda\n"); fflush(stdout);
        volt_array->load_to_device_async();
    }
    if (m_curr_updated) {
        printf("load curr to cuda\n"); fflush(stdout);
        curr_array->load_to_device_async();
    }
    m_host_data_locked = true;

    for (unsigned int iter = 0; iter < iterTS; ++iter) {
        DoPreVoltageUpdates();
        UpdateVoltages(0, numLines[0]);

        DoPostVoltageUpdates();
        Apply2Voltages();

        if (iter == iterTS - 1) {
            volt_array->load_to_host_async();
        }

        DoPreCurrentUpdates();
        UpdateCurrents(0, numLines[0] - 1);
        DoPostCurrentUpdates();
        Apply2Current();

        ++numTS;

        if (iter == iterTS - 1) {
            curr_array->load_to_host_async();
        }
    }


    m_host_data_locked = false;
    m_volt_updated = 0;
    m_curr_updated = 0;
    return true;
}

