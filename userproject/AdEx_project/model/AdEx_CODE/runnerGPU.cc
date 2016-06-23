
//-------------------------------------------------------------------------
/*! \file runnerGPU.cc

\brief File generated from GeNN for the model AdEx containing the host side code for a GPU simulator version.
*/
//-------------------------------------------------------------------------


__device__ double atomicAdd(double* address, double val)
{
    unsigned long long int* address_as_ull =
                                          (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed, 
                        __double_as_longlong(val + 
                        __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}

#include "neuronKrnl.cc"
// ------------------------------------------------------------------------
// copying things to device

void pushAdEx1StateToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_VAdEx1, VAdEx1, 1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_wAdEx1, wAdEx1, 1 * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_I0AdEx1, I0AdEx1, 1 * sizeof(float), cudaMemcpyHostToDevice));
    }

void pushAdEx1SpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntAdEx1, glbSpkCntAdEx1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkAdEx1, glbSpkAdEx1, 1 * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushAdEx1SpikeEventsToDevice()
 {
    }

void pushAdEx1CurrentSpikesToDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkCntAdEx1, glbSpkCntAdEx1, sizeof(unsigned int), cudaMemcpyHostToDevice));
    CHECK_CUDA_ERRORS(cudaMemcpy(d_glbSpkAdEx1, glbSpkAdEx1, glbSpkCntAdEx1[0] * sizeof(unsigned int), cudaMemcpyHostToDevice));
    }

void pushAdEx1CurrentSpikeEventsToDevice()
 {
    }

// ------------------------------------------------------------------------
// copying things from device

void pullAdEx1StateFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(VAdEx1, d_VAdEx1, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(wAdEx1, d_wAdEx1, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(I0AdEx1, d_I0AdEx1, 1 * sizeof(float), cudaMemcpyDeviceToHost));
    }

void pullAdEx1SpikeEventsFromDevice()
 {
    }

void pullAdEx1SpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntAdEx1, d_glbSpkCntAdEx1, 1 * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkAdEx1, d_glbSpkAdEx1, glbSpkCntAdEx1 [0]* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullAdEx1SpikeTimesFromDevice()
 {
    //Assumes that spike numbers are already copied back from the device
    }

void pullAdEx1CurrentSpikesFromDevice()
 {
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntAdEx1, d_glbSpkCntAdEx1, sizeof(unsigned int), cudaMemcpyDeviceToHost));
    CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkAdEx1, d_glbSpkAdEx1, glbSpkCntAdEx1[0] * sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

void pullAdEx1CurrentSpikeEventsFromDevice()
 {
    }

// ------------------------------------------------------------------------
// global copying values to device
void copyStateToDevice()
 {
    pushAdEx1StateToDevice();
    pushAdEx1SpikesToDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes to device
void copySpikesToDevice()
 {
    pushAdEx1SpikesToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikesToDevice()
 {
    pushAdEx1CurrentSpikesToDevice();
    }
// ------------------------------------------------------------------------
// global copying spike events to device
void copySpikeEventsToDevice()
 {
    pushAdEx1SpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// copying current spikes to device
void copyCurrentSpikeEventsToDevice()
 {
    pushAdEx1CurrentSpikeEventsToDevice();
    }
// ------------------------------------------------------------------------
// global copying values from device
void copyStateFromDevice()
 {
    pullAdEx1StateFromDevice();
    pullAdEx1SpikesFromDevice();
    }

// ------------------------------------------------------------------------
// global copying spikes from device
void copySpikesFromDevice()
 {
    
pullAdEx1SpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikes from device
void copyCurrentSpikesFromDevice()
 {
    
pullAdEx1CurrentSpikesFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying spike numbers from device (note, only use when only interested
// in spike numbers; copySpikesFromDevice() already includes this)
void copySpikeNFromDevice()
 {
    
CHECK_CUDA_ERRORS(cudaMemcpy(glbSpkCntAdEx1, d_glbSpkCntAdEx1, 1* sizeof(unsigned int), cudaMemcpyDeviceToHost));
    }

    
// ------------------------------------------------------------------------
// global copying spikeEvents from device
void copySpikeEventsFromDevice()
 {
    
pullAdEx1SpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// copying current spikeEvents from device
void copyCurrentSpikeEventsFromDevice()
 {
    
pullAdEx1CurrentSpikeEventsFromDevice();
    }

    
// ------------------------------------------------------------------------
// global copying spike event numbers from device (note, only use when only interested
// in spike numbers; copySpikeEventsFromDevice() already includes this)
void copySpikeEventNFromDevice()
 {
    
}

    
// ------------------------------------------------------------------------
// the time stepping procedure
void stepTimeGPU()
 {
    
dim3 nThreads(32, 1);
    dim3 nGrid(1, 1);
    
    calcNeurons <<< nGrid, nThreads >>> (t);
    iT++;
    t= iT*DT;
    }

    