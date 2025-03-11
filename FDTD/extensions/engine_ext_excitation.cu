
#include "engine_ext_excitation.h"
#include "operator_ext_excitation.h"

#include <cuda_runtime.h>

#include "hemi/hemi.h"
#include "hemi/launch.h"

#if 1

__global__
void kernelApply2VA(volatile FDTD_FLOAT *d_va, const exitation_point *ep, const FDTD_FLOAT* sig_va, const int *d_dim, const int numTS, const int p, const int N)
{
    int n = threadIdx.x + blockIdx.x * blockDim.x;

    if (n >= N) return;

    ep = ep + n;

	int exc_pos = numTS - ep->delay;
	exc_pos *= (exc_pos>0);
	exc_pos %= p;
	exc_pos *= (exc_pos<(int)ep->length);
	int ny = ep->dir;

    int pos = ep->x * d_dim[1] * d_dim[2] + ep->y * d_dim[2] + ep->z;

    volatile FDTD_FLOAT* va = d_va + (3 * pos);

    va[ny] = va[ny] + ep->amp * sig_va[exc_pos];
    //printf("update %d,%d, %f\n", pos, ny, va[ny]);
}


void Engine_Ext_Excitation::Apply2VoltagesCuda(Engine_cuda *eng)
{
    int N = m_Op_Exc->Volt_Count;
    if (N <= 0 || this->d_ep_v == NULL)     return;

    int numTS = m_Eng->GetNumberOfTimesteps();
    int p = numTS + 1;
    if (m_Op_Exc->m_Exc->GetSignalPeriod() > 0) {
        p = int(m_Op_Exc->m_Exc->GetSignalPeriod() / m_Op_Exc->m_Exc->GetTimestep());
    }

    int threads = std::min(1024, N);
    int blocks = (N + threads - 1) / threads;


    kernelApply2VA<<<blocks, threads>>>(
        eng->GetDeviceVoltData(),
        this->d_ep_v,
        this->d_signal_v,
        eng->GetDeviceDimData(), 
        numTS,
        p, 
        N
    );
    checkCuda(cudaDeviceSynchronize());
    checkCudaErrors();

    unsigned int pos[3];
    for (int n = 0; n < N; n++) {
		pos[0]=m_Op_Exc->Volt_index[0][n];
		pos[1]=m_Op_Exc->Volt_index[1][n];
		pos[2]=m_Op_Exc->Volt_index[2][n];
        eng->UnloadVoltData(m_Op_Exc->Volt_dir[n], pos);
    }
}


void Engine_Ext_Excitation::Apply2CurrentCuda(Engine_cuda *eng)
{
    int N = m_Op_Exc->Curr_Count;
    if (N <= 0 || this->d_ep_a == NULL)     return;

    printf("Applying currents to CUDA engine...\n");
    int numTS = eng->GetNumberOfTimesteps();
    int p = numTS + 1;
    if (m_Op_Exc->m_Exc->GetSignalPeriod() > 0) {
        p = int(m_Op_Exc->m_Exc->GetSignalPeriod() / m_Op_Exc->m_Exc->GetTimestep());
    }

    int threads = std::min(1024, N);
    int blocks = (N + threads - 1) / threads;

    kernelApply2VA<<<blocks, threads>>>(
        eng->GetDeviceCurrData(), 
        this->d_ep_a,
        this->d_signal_a,
        eng->GetDeviceDimData(), 
        numTS,
        p, 
        N
    );

    checkCuda(cudaDeviceSynchronize());
    checkCudaErrors();

    unsigned int pos[3];
    for (int n = 0; n < N; n++) {
		pos[0]=m_Op_Exc->Volt_index[0][n];
		pos[1]=m_Op_Exc->Volt_index[1][n];
		pos[2]=m_Op_Exc->Volt_index[2][n];
        eng->UnloadCurrData(m_Op_Exc->Volt_dir[n], pos);
    }
}





void  Engine_Ext_Excitation::SetEngine(Engine* eng) 
{
    m_Eng = eng;

    if (eng->GetType() != Engine::CUDA) {
        return;
    }

    int n_volt = m_Op_Exc->Volt_Count;
    int n_curr = m_Op_Exc->Curr_Count;

    if (n_volt + n_curr <= 0) {
        return;
    }

	unsigned int length = m_Op_Exc->m_Exc->GetLength();
	FDTD_FLOAT* exc_volt =  m_Op_Exc->m_Exc->GetVoltageSignal();
	FDTD_FLOAT* exc_curr =  m_Op_Exc->m_Exc->GetCurrentSignal();

    exitation_point *ep_all = new exitation_point[n_volt + n_curr];

    printf(">>>>>>>>>>>>>>> load excitation signal to device memory: %d,%d\n", n_volt, n_curr);
    exitation_point *ep = ep_all;
    for (int i = 0; i < n_volt; i++) {
        ep->delay = m_Op_Exc->Volt_delay[i];
        ep->dir = m_Op_Exc->Volt_dir[i];
        ep->amp = m_Op_Exc->Volt_amp[i];
        ep->x = m_Op_Exc->Volt_index[0][i];
        ep->y = m_Op_Exc->Volt_index[1][i];
        ep->z = m_Op_Exc->Volt_index[2][i];
        ep->length = length;


        printf("load volt signal: %d,%d,%d, %f\n", ep->x, ep->y, ep->z, ep->amp);

        ep++;
    }
    for (int i = 0; i < n_curr; i++) {
        ep->delay = m_Op_Exc->Curr_delay[i];
        ep->dir = m_Op_Exc->Curr_dir[i];
        ep->amp = m_Op_Exc->Curr_amp[i];
        ep->x = m_Op_Exc->Curr_index[0][i];
        ep->y = m_Op_Exc->Curr_index[1][i];
        ep->z = m_Op_Exc->Curr_index[2][i];
        ep->length = length;

        printf("load curr signal: %d,%d,%d, %f\n", ep->x, ep->y, ep->z, ep->amp);

        ep++;
    }

    // not sure this is a good practice, by combining two arrays to avoid too much cudaMalloc calls.
    if (n_volt > 0) {
        checkCuda(cudaMalloc(&d_ep_v, sizeof(exitation_point) * n_volt));
        checkCuda(cudaMemcpy(d_ep_v, ep_all, sizeof(exitation_point) * n_volt, cudaMemcpyHostToDevice));
    }

    if (n_curr > 0) {
        checkCuda(cudaMalloc(&d_ep_a, sizeof(exitation_point) * n_curr));
        checkCuda(cudaMemcpy(d_ep_a, ep_all + n_volt, sizeof(exitation_point) * n_curr, cudaMemcpyHostToDevice));
    }


    checkCuda(cudaMalloc(&d_signal_v, sizeof(FDTD_FLOAT) * length));
    checkCuda(cudaMalloc(&d_signal_a, sizeof(FDTD_FLOAT) * length));


    checkCuda(cudaMemcpy(d_signal_v, exc_volt, sizeof(FDTD_FLOAT) * length, cudaMemcpyHostToDevice));
    checkCuda(cudaMemcpy(d_signal_a, exc_curr, sizeof(FDTD_FLOAT) * length, cudaMemcpyHostToDevice));

    delete ep_all;
}

#endif