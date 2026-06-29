#include "operator_cuda.h"
#include "engine_cuda.h"

#include "tools/array_ops.h"

#include <assert.h>

Operator_CUDA* Operator_CUDA::New(unsigned int cuda_device_number)
{
	std::cout << "Create FDTD operator (CUDA-" << cuda_device_number << ")" << std::endl;
	Operator_CUDA* op = new Operator_CUDA();
	op->setCUDAdevice(cuda_device_number);
	op->Init();
	return op;
}

Engine* Operator_CUDA::CreateEngine()
{
	m_Engine = Engine_cuda::New(this, m_cuda_device_number);
	std::cerr << "create cuda engin for cuda operator at " << m_Engine << std::endl;
	return m_Engine;
}

void Operator_CUDA::setCUDAdevice(unsigned int cuda_device_number) {
	m_cuda_device_number = cuda_device_number;
}

