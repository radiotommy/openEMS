#ifndef ENGINE_CUDA_H
#define ENGINE_CUDA_H

#include "engine.h"
#include "operator_cuda.h"


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

	virtual inline FDTD_FLOAT* GetDeviceVoltData() { return volt_ptr->device_data(); }
	virtual inline FDTD_FLOAT* GetDeviceCurrData() { return volt_ptr->device_data(); }
	virtual inline int* GetDeviceDimData() { return d_dim; }
	virtual inline void UnloadVoltData() { volt_ptr->unload(); }
	virtual inline void UnloadCurrData() { volt_ptr->unload(); }

	virtual inline void UnloadVoltData(unsigned int n, const unsigned int pos[3]) {
		int i = getLinearIndex(n, pos[0], pos[1], pos[2]);
		checkCuda(cudaMemcpy(volt_ptr->data() + i, volt_ptr->device_data() + i, sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost));
	}


	virtual inline void UnloadCurrData(unsigned int n, const unsigned int pos[3]) {
		int i = getLinearIndex(n, pos[0], pos[1], pos[2]);
		checkCuda(cudaMemcpy(curr_ptr->data() + i, curr_ptr->device_data() + i, sizeof(FDTD_FLOAT), cudaMemcpyDeviceToHost));
	}

	//this access functions muss be overloaded by any new engine using a different storage model
#if 0
	inline virtual FDTD_FLOAT GetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		return d_volt[getLinearIndex(n, x, y, z)];
	}

	inline virtual FDTD_FLOAT GetVolt(unsigned int n, const unsigned int pos[3]) const
	{
		return d_volt[getLinearIndex(n, pos[0], pos[1], pos[2])];
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z) const
	{
		return d_curr[getLinearIndex(n, x, y, z)];
	}

	inline virtual FDTD_FLOAT GetCurr(unsigned int n, const unsigned int pos[3]) const
	{
		return d_curr[getLinearIndex(n, pos[0], pos[1], pos[2])];
	}

	inline virtual void SetVolt(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
	{
		d_volt[getLinearIndex(n, x, y, z)] = val;
	}

	inline virtual void SetVolt(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
	{
		d_volt[getLinearIndex(n, pos[0], pos[1], pos[2])] = val;
	}

	inline virtual void SetCurr(unsigned int n, unsigned int x, unsigned int y, unsigned int z, FDTD_FLOAT val)
	{
		d_curr[getLinearIndex(n, x, y, z)] = val;
	}

	inline virtual void SetCurr(unsigned int n, const unsigned int pos[3], FDTD_FLOAT val)
	{
		d_curr[getLinearIndex(n, pos[0], pos[1], pos[2])] = val;
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



	int *d_dim;

	double *d_energy_sum;
	FDTD_FLOAT *d_fastEnergy;

private:
	bool m_volt_updated;
	uint32_t m_volt_updated_by_host;

	bool m_curr_updated;
	uint32_t m_curr_updated_by_host;

	void checkZero();

};

#endif // ENGINE_CUDA_H
