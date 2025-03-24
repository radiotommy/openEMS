/*
*	Copyright (C) 2025 Tommy Gu (radiotommy@gmail.com)
*
*	This program is free software: you can redistribute it and/or modify
*	it under the terms of the GNU General Public License as published by
*	the Free Software Foundation, either version 3 of the License, or
*	(at your option) any later version.
*
*	This program is distributed in the hope that it will be useful,
*	but WITHOUT ANY WARRANTY; without even the implied warranty of
*	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
*	GNU General Public License for more details.
*
*	You should have received a copy of the GNU General Public License
*	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef ENGINE_EXT_UNITED_UPML_H
#define ENGINE_EXT_UNITED_UPML_H

#include "engine_extension.h"
#include "FDTD/engine.h"
#include "FDTD/operator.h"
#include "engine_extension_dispatcher.h"

struct upml_block_t {
	dim3 start;
	dim3 lines;
    FDTD_FLOAT *volt_flux;
    FDTD_FLOAT *curr_flux;
    FDTD_FLOAT *vv;
    FDTD_FLOAT *vvfn;
    FDTD_FLOAT *vvfo;
    FDTD_FLOAT *ii;
    FDTD_FLOAT *iifn;
    FDTD_FLOAT *iifo;
};


class Operator_Ext_UPML;

class Engine_Ext_United_UPML : public Engine_Extension
{
public:
    Engine_Ext_United_UPML(std::vector<Operator_Ext_UPML *> *op_ext_upml_list);
	virtual ~Engine_Ext_United_UPML();

	virtual void SetEngine(Engine* eng);

	virtual void DoPreVoltageUpdates() {Engine_Ext_United_UPML::DoPreVoltageUpdates(0);};
	virtual void DoPreVoltageUpdates(int threadID);
	virtual void DoPostVoltageUpdates() {Engine_Ext_United_UPML::DoPostVoltageUpdates(0);};
	virtual void DoPostVoltageUpdates(int threadID);

	virtual void DoPreCurrentUpdates() {Engine_Ext_United_UPML::DoPreCurrentUpdates(0);};
	virtual void DoPreCurrentUpdates(int threadID);
	virtual void DoPostCurrentUpdates() {Engine_Ext_United_UPML::DoPostCurrentUpdates(0);};
	virtual void DoPostCurrentUpdates(int threadID);


protected:
    template <typename EngType>
    void DoPreVoltageUpdatesImpl(EngType* eng, int threadID);

    template <typename EngType>
    void DoPostVoltageUpdatesImpl(EngType* eng, int threadID);

    template <typename EngType>
    void DoPreCurrentUpdatesImpl(EngType* eng, int threadID);

    template <typename EngType>
    void DoPostCurrentUpdatesImpl(EngType* eng, int threadID);

private:
	std::vector<Operator_Ext_UPML *> *m_Op_UPML_List;

    // location and size of each UPML block
    upml_block_t *m_d_blks;
    // pre calculate how many cells are in each block
    int *m_num_of_cells_in_block;
    int m_max_num_cells_in_block;

    std::vector<ArrayLib::ArrayNIJK<FDTD_FLOAT> *> m_volt_fluxes;
    std::vector<ArrayLib::ArrayNIJK<FDTD_FLOAT> *> m_curr_fluxes;

};

#endif