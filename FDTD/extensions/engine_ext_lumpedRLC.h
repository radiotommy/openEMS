/*
*       Copyright (C) 2023 Gadi Lahav (gadi@rfwithcare.com)
*
*       This program is free software: you can redistribute it and/or modify
*       it under the terms of the GNU General Public License as published by
*       the Free Software Foundation, either version 3 of the License, or
*       (at your option) any later version.
*
*       This program is distributed in the hope that it will be useful,
*       but WITHOUT ANY WARRANTY; without even the implied warranty of
*       MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*       GNU General Public License for more details.
*
*       You should have received a copy of the GNU General Public License
*       along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_EXT_LUMPEDRLC_H
#define ENGINE_EXT_LUMPEDRLC_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

class Operator_Ext_LumpedRLC;

#if WITH_CUDA
struct lumpedRLC_cuda_point
{
        unsigned int x;
        unsigned int y;
        unsigned int z;
        int dir;

        FDTD_FLOAT ilv;
        FDTD_FLOAT i2v;
        FDTD_FLOAT vv2;
        FDTD_FLOAT vj1;
        FDTD_FLOAT vj2;
        FDTD_FLOAT vvd;
        FDTD_FLOAT ib0;
        FDTD_FLOAT b1;
        FDTD_FLOAT b2;

        FDTD_FLOAT Il;
        FDTD_FLOAT Vdn0;
        FDTD_FLOAT Vdn1;
        FDTD_FLOAT Vdn2;
        FDTD_FLOAT Jn0;
        FDTD_FLOAT Jn1;
        FDTD_FLOAT Jn2;
};
#endif

class Engine_Ext_LumpedRLC : public Engine_Extension
{
        friend class Operator_Ext_LumpedRLC;
        friend class Operator;
        friend class ContinuousStructure;

public:

        Engine_Ext_LumpedRLC(Operator_Ext_LumpedRLC *op_ext_RLC);
        virtual ~Engine_Ext_LumpedRLC();

        virtual void DoPreVoltageUpdates();
        virtual void Apply2Voltages();

#if WITH_CUDA
        virtual void SetEngine(Engine* eng);
#endif

protected:
        template <typename EngType>
        void Apply2VoltagesImpl(EngType* eng);

#if WITH_CUDA
        void DoPreVoltageUpdatesCuda(Engine_cuda* eng);
        void Apply2VoltagesCuda(Engine_cuda* eng);
        void FreeCudaData();

        lumpedRLC_cuda_point* d_RLC;
#endif

        Operator_Ext_LumpedRLC* m_Op_Ext_RLC;

        // Auxilliary containers

        // Array setup: volt_C_ADE[mesh_pos]
        FDTD_FLOAT *v_Il;               // Container for current on inductor- Parallel RLC

        FDTD_FLOAT **v_Vdn;             // Container for nodal vd at [n],[n-1],[n-2]
        FDTD_FLOAT **v_Jn;              // Container for nodal J at [n],[n-1],[n-2]

};

#endif // ENGINE_EXT_LUMPEDRLC_H
