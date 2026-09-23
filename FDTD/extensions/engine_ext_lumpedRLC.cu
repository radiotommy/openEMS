#include "engine_ext_lumpedRLC.h"
#include "operator_ext_lumpedRLC.h"

#include <cuda_runtime.h>
#include "hemi/hemi_error.h"

__global__
void LumpedRLCPreVoltageKernel(lumpedRLC_cuda_point* rlc, const unsigned int N)
{
        unsigned int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pIdx >= N)
                return;

        lumpedRLC_cuda_point* p = rlc + pIdx;

        // CPU equivalent:
        //   Vdn[2] <- Vdn[1]
        //   Vdn[1] <- Vdn[0]
        // followed by the parallel-inductor ADE update.
        p->Vdn2 = p->Vdn1;
        p->Vdn1 = p->Vdn0;

        p->Il += p->i2v * p->ilv * p->Vdn1;
}

__global__
void LumpedRLCApplyVoltageKernel(FDTD_FLOAT* volt,
                                 lumpedRLC_cuda_point* rlc,
                                 const dim3 dim,
                                 const unsigned int N)
{
        unsigned int pIdx = blockIdx.x * blockDim.x + threadIdx.x;
        if (pIdx >= N)
                return;

        lumpedRLC_cuda_point* p = rlc + pIdx;

        // CPU equivalent:
        //   Jn[2] <- Jn[1]
        //   Jn[1] <- Jn[0]
        p->Jn2 = p->Jn1;
        p->Jn1 = p->Jn0;

        const unsigned int cell =
                p->x * dim.y * dim.z +
                p->y * dim.z +
                p->z;

        FDTD_FLOAT* nodeVolt = volt + cell * 3 + p->dir;

        // Read the voltage just calculated by the CUDA FDTD engine.
        p->Vdn0 = *nodeVolt;

        // Preserve the CPU ADE equations exactly.
        p->Vdn0 = p->vvd * (
                p->Vdn0 - p->Il
                + p->vv2 * p->Vdn2
                + p->vj1 * p->Jn1
                + p->vj2 * p->Jn2);

        p->Jn0 =
                p->ib0 * (p->Vdn0 - p->Vdn2)
                - (p->b1 * p->ib0) * p->Jn1
                - (p->b2 * p->ib0) * p->Jn2;

        // Write the corrected voltage directly to the live device field.
        *nodeVolt = p->Vdn0;
}

void Engine_Ext_LumpedRLC::FreeCudaData()
{
        if (d_RLC != NULL)
        {
                cudaFree(d_RLC);
                d_RLC = NULL;
        }
}

void Engine_Ext_LumpedRLC::SetEngine(Engine* eng)
{
        m_Eng = eng;

        FreeCudaData();

        if (eng->GetType() != Engine::CUDA)
                return;

        const unsigned int N = m_Op_Ext_RLC->RLC_count;
        if (N == 0)
                return;

        lumpedRLC_cuda_point* host =
                new lumpedRLC_cuda_point[N];

        for (unsigned int pIdx = 0; pIdx < N; ++pIdx)
        {
                lumpedRLC_cuda_point& p = host[pIdx];

                p.x = m_Op_Ext_RLC->v_RLC_pos[0][pIdx];
                p.y = m_Op_Ext_RLC->v_RLC_pos[1][pIdx];
                p.z = m_Op_Ext_RLC->v_RLC_pos[2][pIdx];
                p.dir = m_Op_Ext_RLC->v_RLC_dir[pIdx];

                p.ilv = m_Op_Ext_RLC->v_RLC_ilv[pIdx];
                p.i2v = m_Op_Ext_RLC->v_RLC_i2v[pIdx];

                p.vv2 = m_Op_Ext_RLC->v_RLC_vv2[pIdx];
                p.vj1 = m_Op_Ext_RLC->v_RLC_vj1[pIdx];
                p.vj2 = m_Op_Ext_RLC->v_RLC_vj2[pIdx];
                p.vvd = m_Op_Ext_RLC->v_RLC_vvd[pIdx];
                p.ib0 = m_Op_Ext_RLC->v_RLC_ib0[pIdx];
                p.b1  = m_Op_Ext_RLC->v_RLC_b1[pIdx];
                p.b2  = m_Op_Ext_RLC->v_RLC_b2[pIdx];

                p.Il = 0.0;
                p.Vdn0 = 0.0;
                p.Vdn1 = 0.0;
                p.Vdn2 = 0.0;
                p.Jn0 = 0.0;
                p.Jn1 = 0.0;
                p.Jn2 = 0.0;
        }

        checkCuda(cudaMalloc(&d_RLC, sizeof(lumpedRLC_cuda_point) * N));
        checkCuda(cudaMemcpy(d_RLC,
                             host,
                             sizeof(lumpedRLC_cuda_point) * N,
                             cudaMemcpyHostToDevice));

        delete[] host;
}

void Engine_Ext_LumpedRLC::DoPreVoltageUpdatesCuda(Engine_cuda* eng)
{
        (void)eng;

        const unsigned int N = m_Op_Ext_RLC->RLC_count;
        if (N == 0 || d_RLC == NULL)
                return;

        const unsigned int threads = 256;
        const unsigned int blocks = (N + threads - 1) / threads;

        LumpedRLCPreVoltageKernel<<<blocks, threads>>>(d_RLC, N);
}

void Engine_Ext_LumpedRLC::Apply2VoltagesCuda(Engine_cuda* eng)
{
        const unsigned int N = m_Op_Ext_RLC->RLC_count;
        if (N == 0 || d_RLC == NULL)
                return;

        const unsigned int threads = 256;
        const unsigned int blocks = (N + threads - 1) / threads;

        LumpedRLCApplyVoltageKernel<<<blocks, threads>>>(
                eng->GetDeviceVoltData(),
                d_RLC,
                eng->GetDeviceDimData(),
                N);
}
