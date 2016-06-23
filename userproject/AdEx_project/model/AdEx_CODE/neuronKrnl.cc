

#ifndef _AdEx_neuronKrnl_cc
#define _AdEx_neuronKrnl_cc

//-------------------------------------------------------------------------
/*! \file neuronKrnl.cc

\brief File generated from GeNN for the model AdEx containing the neuron kernel function.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

extern "C" __global__ void calcNeurons(float t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[32];
    __shared__ volatile unsigned int posSpk;
    unsigned int spkIdx;
    __shared__ volatile unsigned int spkCount;
    
    if (id == 0) {
        dd_glbSpkCntAdEx1[0] = 0;
        }
    __threadfence();
    
    if (threadIdx.x == 0) {
        spkCount = 0;
        }
    __syncthreads();
    
    // neuron group AdEx1
    if (id < 32) {
        
        // only do this for existing neurons
        if (id < 1) {
            // pull neuron variables in a coalesced access
            float lV = dd_VAdEx1[id];
            float lw = dd_wAdEx1[id];
            float lI0 = dd_I0AdEx1[id];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= ((lV > 29.99f));
            // calculate membrane potential
            
	double curr_v = lV;
    if (lV > 0.0f){
		lV=(-48.0000f);
		lw+=(30.0000f);
		} 
	else {
		lV+=((lI0 - ((12.0000f)*(lV - (-60.0000f))) + ((12.0000f)*(2.00000f)*expf((lV - (-50.0000f)) / (2.00000f))) - lw)/(100.000f))*DT;
		lw+= ((((-11.0000f)*(curr_v - (-60.0000f))) - lw) / (130.000f))*DT; 
				if (lV > 30.0f){
					lV=30.0f;
					} 
		}
		
            // test for and register a true spike
            if (((lV > 29.99f)) && !(oldSpike))  {
                spkIdx = atomicAdd((unsigned int *) &spkCount, 1);
                shSpk[spkIdx] = id;
                }
            dd_VAdEx1[id] = lV;
            dd_wAdEx1[id] = lw;
            dd_I0AdEx1[id] = lI0;
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (spkCount > 0) posSpk = atomicAdd((unsigned int *) &dd_glbSpkCntAdEx1[0], spkCount);
            }
        __syncthreads();
        if (threadIdx.x < spkCount) {
            dd_glbSpkAdEx1[posSpk + threadIdx.x] = shSpk[threadIdx.x];
            }
        }
    
    }

    #endif
