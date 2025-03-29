#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#include "engine.h"
#include "operator_cuda.h"

#include "tools/cuda/array.h"


class Operator_CUDA;

class Engine_cuda : public Engine
{
public:
	static Engine_cuda* New(const Operator_CUDA* op, unsigned int cuda_device_number);
	virtual ~Engine_cuda();

	virtual void Init();
	virtual void Reset();

	virtual void setCUDAdevice(unsigned int cuda_device_number);

	//!Iterate a number of timesteps
	virtual bool IterateTS(unsigned int iterTS);

	int inline FlatIndex(int x, int y, int z) const { 
		return x * numLines[1] * numLines[2] + y * numLines[2] + z; 
	}


	unsigned int m_cuda_device_number;
	int m_supports_coop_launch;

	virtual double CalcFastEnergy();

	virtual void AddVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);
	virtual void AddCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT value);

	virtual inline FDTD_FLOAT* GetDeviceVoltData() { return volt_array->device_data(); }
	virtual inline FDTD_FLOAT* GetDeviceCurrData() { return curr_array->device_data(); }
	virtual inline dim3 GetDeviceDimData() { return m_dim; }

	//this access functions muss be overloaded by any new engine using a different storage model
#if 1
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		assert(!m_host_data_locked);
		FDTD_FLOAT *volt = volt_array->host_data();
		return volt[FlatIndex(x, y, z) * 3 + n];
	}

	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{
		return GetVolt(n, pos[0], pos[1], pos[2]);
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		assert(!m_host_data_locked);

		return curr_array->host_data()[FlatIndex(x, y, z) * 3 + n];
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{
		return GetCurr(n, pos[0], pos[1], pos[2]);
	}

	inline virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
	{
		assert(!m_host_data_locked);
		volt_array->host_data()[FlatIndex(x, y, z) * 3 + n] = val;
		m_volt_updated += 1;
	}

	inline virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
	{
		SetVolt(n, pos[0], pos[1], pos[2], val);
	}

	inline virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
	{
		assert(!m_host_data_locked);
		curr_array->host_data()[FlatIndex(x, y, z) * 3 + n] = val;
		m_curr_updated += 1;
	}

	inline virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
	{
		SetCurr(n, pos[0], pos[1], pos[2], val);
	}
#endif


protected:

	Engine_cuda(const Operator_CUDA* op);
	const Operator_CUDA* Op;

	virtual void UpdateVoltages(unsigned int startX, unsigned int numX);
	virtual void UpdateCurrents(unsigned int startX, unsigned int numX);

	inline unsigned int getLinearIndex(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		return (x * (numLines[1] * numLines[2]) + y * numLines[2] + z) * 3 + n;
	}

	inline unsigned int getLinearIndex(unsigned int n, const unsigned int pos[3]) const
	{
		return getLinearIndex(n, pos[0], pos[1], pos[2]);
	}

	dim3 m_dim;

	double *d_energy_sum;
	FDTD_FLOAT *d_fastEnergy;

	FDTD_FLOAT *d_op_vv_vi;
	FDTD_FLOAT *d_op_ii_iv;

private:

	CudaHelper::Array<FDTD_FLOAT> *volt_array;
	CudaHelper::Array<FDTD_FLOAT> *curr_array;

	int m_volt_updated;
	uint32_t m_volt_updated_by_host;

	int m_curr_updated;
	uint32_t m_curr_updated_by_host;

	bool m_host_data_locked;

	void checkZero();

};

#endif // ENGINE_CUDA_H
