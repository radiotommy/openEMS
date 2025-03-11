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

#include "engine_interface_cuda_fdtd.h"

Engine_Interface_CUDA_FDTD::Engine_Interface_CUDA_FDTD(Operator_CUDA* op) : Engine_Interface_FDTD(op)
{
	m_Op_CUDA= op;
	m_Eng_CUDA = dynamic_cast<Engine_cuda*>(m_Op_CUDA->GetEngine());
	if (m_Eng_CUDA==NULL)
	{
		cerr << "Engine_Interface_SSE_FDTD::Engine_Interface_SSE_FDTD: Error: SSE-Engine is not set! Exit!" << endl;
		exit(1);
	}
}

Engine_Interface_CUDA_FDTD::~Engine_Interface_CUDA_FDTD()
{
	m_Op_CUDA = NULL;
	m_Eng_CUDA = NULL;
}

double Engine_Interface_CUDA_FDTD::CalcFastEnergy() const
{
#if 1
	double E_energy=0.0;
	double H_energy=0.0;

	unsigned int pos[3];
	for (pos[0]=0; pos[0]<m_Op->GetNumberOfLines(0)-1; ++pos[0])
	{
		for (pos[1]=0; pos[1]<m_Op->GetNumberOfLines(1)-1; ++pos[1])
		{
			for (pos[2]=0; pos[2]<m_Op->GetNumberOfLines(2)-1; ++pos[2])
			{
				E_energy+=m_Eng->Engine::GetVolt(0,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetVolt(0,pos[0],pos[1],pos[2]);
				E_energy+=m_Eng->Engine::GetVolt(1,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetVolt(1,pos[0],pos[1],pos[2]);
				E_energy+=m_Eng->Engine::GetVolt(2,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetVolt(2,pos[0],pos[1],pos[2]);

				H_energy+=m_Eng->Engine::GetCurr(0,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetCurr(0,pos[0],pos[1],pos[2]);
				H_energy+=m_Eng->Engine::GetCurr(1,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetCurr(1,pos[0],pos[1],pos[2]);
				H_energy+=m_Eng->Engine::GetCurr(2,pos[0],pos[1],pos[2]) * m_Eng->Engine::GetCurr(2,pos[0],pos[1],pos[2]);
			}
		}
	}
	return __EPS0__*E_energy + __MUE0__*H_energy;
#else
	return m_Eng_CUDA->CalcFastEnergy();
#endif


}
