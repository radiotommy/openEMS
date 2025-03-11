
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

using namespace std;


#define USE_UNIFIED_MEM

#define flat_index(X, Y, Z, D)     ((X) * D[1] * D[2] + ((Y) * D[2]) + (Z))
#define THREADS     1024


// Kernel for voltage updates with flat arrays
__global__
void updateVoltagesKernel(FDTD_FLOAT *volt, const FDTD_FLOAT *curr,
									 const FDTD_FLOAT *opvv, const FDTD_FLOAT *opvi,
                                     int *dim, int N)
{
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = cell * 3;
        int x = cell / (dim[1] * dim[2]);
        int y = (cell / dim[2]) % dim[1];
        int z = cell % dim[2];

        FDTD_FLOAT* v= volt + offs;
        const FDTD_FLOAT* i= curr + offs;
        const FDTD_FLOAT* vv = opvv + offs;
        const FDTD_FLOAT* vi = opvi + offs;

        // nbr cells in x, y, z direction
        const FDTD_FLOAT* ix = curr + (3 * flat_index(x - ((x!=0)), y, z, dim));
        const FDTD_FLOAT* iy = curr + (3 * flat_index(x, (y - (y!=0)), z, dim));
        const FDTD_FLOAT* iz = curr + (3 * flat_index(x, y, (z - (z!=0)), dim));

        // update x
        v[0]  = v[0] * vv[0] + vi[0] * (i[2] - iy[2] - i[1] + iz[1]);
        // update y
        v[1] = v[1] * vv[1] + vi[1] * (i[0] - iz[0] - i[2] + ix[2]);
        // update z
        v[2] = v[2] * vv[2] + vi[2] * (i[1] -ix[1] - i[0] + iy[0]); 

        //if (cell == 171199) {
        //    printf("%d: %f %f %f\n", blockIdx.x, v[0], v[1], v[2]);
        //}
    }
}


// Kernel for current updates
__global__
void updateCurrentsKernel(FDTD_FLOAT *curr, const FDTD_FLOAT *volt,
									 const FDTD_FLOAT *opii, const FDTD_FLOAT *opiv,
                                     int *dim, int N)
{
    for (auto cell : hemi::grid_stride_range(0, N)) {

        int offs = cell * 3;
        int x = cell / (dim[1] * dim[2]);
        int y = (cell / dim[2]) % dim[1];
        int z = cell % dim[2];

        if ((y < dim[1] - 1) && (z < dim[2] - 1)) {
            const FDTD_FLOAT* v = volt + offs;
            FDTD_FLOAT* i = curr + offs;
            const FDTD_FLOAT* ii = opii + offs;
            const FDTD_FLOAT* iv = opiv + offs;

            // next nbr cells in x, y, z direction
            const FDTD_FLOAT* vx = volt + (3 * flat_index(x + 1, y, z, dim));
            const FDTD_FLOAT* vy = volt + (3 * flat_index(x, y + 1, z, dim));
            const FDTD_FLOAT* vz = volt + (3 * flat_index(x, y,  z + 1, dim));

            // update x
            i[0] = i[0] * ii[0] + iv[0] * (v[2] - vy[2] - v[1] + vz[1]);
            // update y
            i[1] = i[1] * ii[1] + iv[1] * (v[0] - vz[0] - v[2] + vx[2]);
            // update z
            i[2] = i[2] * ii[2] + iv[2] * (v[1] - vx[1] - v[0] + vy[0]);
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

__global__ void calcFastEnergyKernel(FDTD_FLOAT *volt, FDTD_FLOAT *curr, double *p_sum, int *dim)
{
    __shared__ double vv[THREADS]; // shared memory for energy calculation, between threads in a block
    __shared__ double ii[THREADS]; // shared memory for energy calculation, between threads in a block

    double local_vv = 0.0;
    double local_ii = 0.0;
    int tid = threadIdx.x;

    // Process all cells in strides
    int total = dim[0] * dim[1] * dim[2];
    for (int cell = tid; cell < total; cell += THREADS) {
        int x = cell / (dim[1] * dim[2]);
        int y = (cell / dim[2]) % dim[1];
        int z = cell % dim[2];
        if (x < dim[0] - 1 && y < dim[1] - 1 && z < dim[2] - 1) {
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
        p_sum[0] = vv[0] * __EPS0__ + ii[0] * __MUE0__;
    }
}


Engine_cuda* Engine_cuda::New(const Operator_CUDA* op, unsigned int cuda_device_number) {
    cout << "Create CUDA FDTD engine" << endl;
    Engine_cuda* e = new Engine_cuda(op);
    e->setCUDAdevice(cuda_device_number);
    e->Init();
    fprintf(stderr, "engen cuda at %p\n", e);
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

void Engine_cuda::Init() {
	   

    int nDevices;
    cudaGetDeviceCount(&nDevices);
    if (nDevices <= 0) {
        throw std::runtime_error("NO CUDA device found");
    }
    if (m_cuda_device_number >= nDevices) {
        fprintf(stderr, "cuda_device: %d/%d\n", m_cuda_device_number, nDevices);
        throw std::runtime_error("CUDA device number out of range");
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, m_cuda_device_number);
    cout << "  Running on device: " << prop.name << endl;
    cout << "    max block dimensions" \
        << prop.maxThreadsDim[0] << "," \
        << prop.maxThreadsDim[1] << "," \
        << prop.maxThreadsDim[2] << endl;

    cout << "    max block dimensions" \
        << prop.maxGridSize[0] << "," \
        << prop.maxGridSize[1] << "," \
        << prop.maxGridSize[2] << endl;
   
    cudaSetDevice(m_cuda_device_number);
    cudaDeviceGetAttribute(&m_supports_coop_launch, cudaDevAttrCooperativeLaunch, m_cuda_device_number);

	numTS = 0;
	volt_ptr = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("volt", numLines);
	curr_ptr = new ArrayLib::ArrayNIJK<FDTD_FLOAT>("curr", numLines);

    volt_ptr->load();
    curr_ptr->load();

    // Allocate GPU memory
    Op->vv_ptr->load();
    Op->vi_ptr->load();
    Op->iv_ptr->load();
    Op->ii_ptr->load();

    checkCuda(cudaMalloc(&d_energy_sum, sizeof(double)));

    checkCuda(cudaMalloc(&d_dim, 3 * sizeof(int)));
    checkCuda(cudaMemcpy(d_dim, numLines, 3 * sizeof(int), cudaMemcpyHostToDevice));

    InitExtensions();
    SortExtensionByPriority();
}

void Engine_cuda::Reset() {
	delete volt_ptr;
    volt_ptr = NULL;
	delete curr_ptr;
    curr_ptr = NULL;

    // Free GPU memory
    if (d_dim)          checkCuda(cudaFree(d_dim));
    if (d_energy_sum)   checkCuda(cudaFree(d_energy_sum));

    ClearExtensions();
}


void Engine_cuda::UpdateVoltages(unsigned int startX, unsigned int numX) {
    // Copy current data to GPU
    int N = numX * numLines[1] * numLines[2];

    volt_ptr->load();
    curr_ptr->load();

    // Launch kernel
    int blocks = (N  + THREADS - 1) / THREADS;
    updateVoltagesKernel<<< blocks, THREADS >>>(volt_ptr->device_data(), (const FDTD_FLOAT*)curr_ptr->device_data(), 
            Op->vv_ptr->device_data(), Op->vi_ptr->device_data(), d_dim, N);

    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());

    volt_ptr->unload();
}

void Engine_cuda::UpdateCurrents(unsigned int startX, unsigned int numX) {
    // Copy voltage data to GPU
    volt_ptr->load();
    curr_ptr->load();

    int N = numX * numLines[1] * numLines[2];

    // Launch kernel
    int blocks = (N  + THREADS - 1) / THREADS;
    updateCurrentsKernel<<< blocks, THREADS >>>(curr_ptr->device_data(), (const FDTD_FLOAT*)volt_ptr->device_data(), 
            Op->ii_ptr->device_data(), Op->iv_ptr->device_data(), d_dim, N);
    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());

    curr_ptr->unload();
}

void Engine_cuda::AddVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int cell = flat_index(pos[0], pos[1], pos[2], numLines);
    addInKernel<<< 1, 1>>>(volt_ptr->device_data(), cell, n, value);

    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());

	ArrayLib::ArrayNIJK<FDTD_FLOAT>& volt = *volt_ptr;
    volt[n][pos[0]][pos[1]][pos[2]] += value;
}

void Engine_cuda::AddCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value)
{
    int cell = flat_index(pos[0], pos[1], pos[2], numLines);
    addInKernel<<<1, 1>>>(curr_ptr->device_data(), cell, n, value);

    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());

	ArrayLib::ArrayNIJK<FDTD_FLOAT>& curr = *curr_ptr;
    curr[n][pos[0]][pos[1]][pos[2]] += value;
}


double Engine_cuda::CalcFastEnergy()
{
    calcFastEnergyKernel<<<1, THREADS>>>(volt_ptr->device_data(), curr_ptr->device_data(), d_energy_sum, d_dim);

    checkCudaErrors();
    checkCuda(cudaDeviceSynchronize());

    double energy_sum;
    checkCuda(cudaMemcpy(&energy_sum, d_energy_sum, sizeof(energy_sum), cudaMemcpyDeviceToHost));
    return energy_sum;
}

// Other methods (InitExtensions, SortExtensionByPriority, etc.) remain unchanged unless extensions need CUDA support

bool Engine_cuda::IterateTS(unsigned int iterTS) {
    // TODO: run all extentions in cuda
    // load data to cuda
    // set extension like a pipeline?
    // load data back

    m_volt_updated_by_host = 1;
    m_curr_updated_by_host = 1;

    for (unsigned int iter = 0; iter < iterTS; ++iter) {
        DoPreVoltageUpdates();
        UpdateVoltages(0, numLines[0]);

        DoPostVoltageUpdates();
        Apply2Voltages();

        DoPreCurrentUpdates();
        UpdateCurrents(0, numLines[0] - 1);
        DoPostCurrentUpdates();
        Apply2Current();

        ++numTS;
    }
    return true;
}