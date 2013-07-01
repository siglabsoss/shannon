/******************************************************************************
* Copyright 2013 PopWi Technology Group, Inc. (PTG)
*
* This file is proprietary and exclusively owned by PTG or its associates.
* This document is protected by international and domestic patents where
* applicable. All rights reserved.
*
******************************************************************************/

#include "popgpu.hpp"

#include <memory>
#include <iostream>
#include <complex>

#include <cuda.h>
#include <cuda_runtime.h>

#include "cuda/helper_cuda.h"

using namespace std;

/**************************************************************************
 * CUDA Function Prototypes
 *************************************************************************/
extern "C" void start_deconvolve(std::complex<float> *pn,
							     std::complex<float> *data,
							     std::complex<float> *product,
                                 int len);
namespace pop
{
	PopGpu::PopGpu()
	{
	    int ret;
	    int deviceCount = 0;

	    mp_barrier = new boost::barrier(2);

	    cout << "initializing graphics card(s)...." << endl;

	    cudaError_t error_id = cudaGetDeviceCount(&deviceCount);

	    if (error_id != cudaSuccess)
	    {
	        printf("cudaGetDeviceCount returned %d\n-> %s\n", (int)error_id, cudaGetErrorString(error_id));
	        exit(EXIT_FAILURE);
	    }

	    // This function call returns 0 if there are no CUDA capable devices.
	    if (deviceCount == 0)
	    {
	        printf("There are no available device(s) that support CUDA\n");
	    }
	    else
	    {
	        printf("Detected %d CUDA Capable device(s)\n", deviceCount);
	    }

	    int dev, driverVersion = 0, runtimeVersion = 0;

	    for (dev = 0; dev < deviceCount; ++dev)
	    {
	        cudaSetDevice(dev);
	        cudaDeviceProp deviceProp;
	        cudaGetDeviceProperties(&deviceProp, dev);

	        printf("\nDevice %d: \"%s\"\n", dev, deviceProp.name);

	        // Console log
	        cudaDriverGetVersion(&driverVersion);
	        cudaRuntimeGetVersion(&runtimeVersion);
	        printf("  CUDA Driver Version / Runtime Version          %d.%d / %d.%d\n", driverVersion/1000, (driverVersion%100)/10, runtimeVersion/1000, (runtimeVersion%100)/10);
	        printf("  CUDA Capability Major/Minor version number:    %d.%d\n", deviceProp.major, deviceProp.minor);

	        char msg[256];
	        sprintf(msg, "  Total amount of global memory:                 %.0f MBytes (%llu bytes)\n",
	                (float)deviceProp.totalGlobalMem/1048576.0f, (unsigned long long) deviceProp.totalGlobalMem);
	        printf("%s", msg);

	        printf("  (%2d) Multiprocessors x (%3d) CUDA Cores/MP:    %d CUDA Cores\n",
	               deviceProp.multiProcessorCount,
	               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor),
	               _ConvertSMVer2Cores(deviceProp.major, deviceProp.minor) * deviceProp.multiProcessorCount);
	        printf("  GPU Clock rate:                                %.0f MHz (%0.2f GHz)\n", deviceProp.clockRate * 1e-3f, deviceProp.clockRate * 1e-6f);


	        printf("  Memory Clock rate:                             %.0f Mhz\n", deviceProp.memoryClockRate * 1e-3f);
	        printf("  Memory Bus Width:                              %d-bit\n",   deviceProp.memoryBusWidth);

	        if (deviceProp.l2CacheSize)
	        {
	            printf("  L2 Cache Size:                                 %d bytes\n", deviceProp.l2CacheSize);
	        }

	        printf("  Max Texture Dimension Size (x,y,z)             1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
	               deviceProp.maxTexture1D   , deviceProp.maxTexture2D[0], deviceProp.maxTexture2D[1],
	               deviceProp.maxTexture3D[0], deviceProp.maxTexture3D[1], deviceProp.maxTexture3D[2]);
	        printf("  Max Layered Texture Size (dim) x layers        1D=(%d) x %d, 2D=(%d,%d) x %d\n",
	               deviceProp.maxTexture1DLayered[0], deviceProp.maxTexture1DLayered[1],
	               deviceProp.maxTexture2DLayered[0], deviceProp.maxTexture2DLayered[1], deviceProp.maxTexture2DLayered[2]);

	        printf("  Total amount of constant memory:               %lu bytes\n", deviceProp.totalConstMem);
	        printf("  Total amount of shared memory per block:       %lu bytes\n", deviceProp.sharedMemPerBlock);
	        printf("  Total number of registers available per block: %d\n", deviceProp.regsPerBlock);
	        printf("  Warp size:                                     %d\n", deviceProp.warpSize);
	        printf("  Maximum number of threads per multiprocessor:  %d\n", deviceProp.maxThreadsPerMultiProcessor);
	        printf("  Maximum number of threads per block:           %d\n", deviceProp.maxThreadsPerBlock);
	        printf("  Maximum sizes of each dimension of a block:    %d x %d x %d\n",
	               deviceProp.maxThreadsDim[0],
	               deviceProp.maxThreadsDim[1],
	               deviceProp.maxThreadsDim[2]);
	        printf("  Maximum sizes of each dimension of a grid:     %d x %d x %d\n",
	               deviceProp.maxGridSize[0],
	               deviceProp.maxGridSize[1],
	               deviceProp.maxGridSize[2]);
	        printf("  Maximum memory pitch:                          %lu bytes\n", deviceProp.memPitch);
	        printf("  Texture alignment:                             %lu bytes\n", deviceProp.textureAlignment);
	        printf("  Concurrent copy and kernel execution:          %s with %d copy engine(s)\n", (deviceProp.deviceOverlap ? "Yes" : "No"), deviceProp.asyncEngineCount);
	        printf("  Run time limit on kernels:                     %s\n", deviceProp.kernelExecTimeoutEnabled ? "Yes" : "No");
	        printf("  Integrated GPU sharing Host Memory:            %s\n", deviceProp.integrated ? "Yes" : "No");
	        printf("  Support host page-locked memory mapping:       %s\n", deviceProp.canMapHostMemory ? "Yes" : "No");
	        printf("  Alignment requirement for Surfaces:            %s\n", deviceProp.surfaceAlignment ? "Yes" : "No");
	        printf("  Device has ECC support:                        %s\n", deviceProp.ECCEnabled ? "Enabled" : "Disabled");
	        printf("  Device supports Unified Addressing (UVA):      %s\n", deviceProp.unifiedAddressing ? "Yes" : "No");
	        printf("  Device PCI Bus ID / PCI location ID:           %d / %d\n", deviceProp.pciBusID, deviceProp.pciDeviceID);

	        const char *sComputeMode[] =
	        {
	            "Default (multiple host threads can use ::cudaSetDevice() with device simultaneously)",
	            "Exclusive (only one host thread in one process is able to use ::cudaSetDevice() with this device)",
	            "Prohibited (no host thread can use ::cudaSetDevice() with this device)",
	            "Exclusive Process (many threads in one process is able to use ::cudaSetDevice() with this device)",
	            "Unknown",
	            NULL
	        };
	        printf("  Compute Mode:\n");
	        printf("     < %s >\n", sComputeMode[deviceProp.computeMode]);
	    }

	    init();
	}


	/**
	 * Initialize GPU hardware.
	 */
	void PopGpu::init()
	{
		// start new GPU process I/O thread
		mp_thread = new boost::thread(boost::bind(&PopGpu::run, this));
	}


	/**
	 * Push receive buffer and recompute. Data should be recast to std::complex
	 */
	void PopGpu::import(void* data, std::size_t len)
	{
		std::complex<float> *cdata = (std::complex<float>*)data;

		// TODO: copy new data into buffer

		// wait for process I/O to complete or start new process I/O
		mp_barrier->wait();
	}

	void PopGpu::crunch()
	{
		// wait for new data to arrive
		mp_barrier->wait();

		// TODO: call the GPU to process work
		// temporary, sleep for 10 milliseconds
		//boost::this_thread::sleep(boost::posix_time::milliseconds(10));

		static std::complex<float> a[10];
		static std::complex<float> b[10];
		static std::complex<float> c[10];
		start_deconvolve(a, b, c, 0);
	}


	/**
	 * Thread loop for synchronous GPU communcation
	 */
	 void PopGpu::run()
	 {
	 	while(1)
	 	{
	 		crunch();
	 	}
	 }


	 /**
	  * Standard class deconstructor.
	  */
	PopGpu::~PopGpu()
	{
		delete mp_barrier;
	}
}
