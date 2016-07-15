

#ifndef _AdEx_neuronFnct_cc
#define _AdEx_neuronFnct_cc

//-------------------------------------------------------------------------
/*! \file neuronFnct.cc

\brief File generated from GeNN for the model AdEx containing the the equivalent of neuron kernel function for the CPU-only version.
*/
//-------------------------------------------------------------------------

// include the support codes provided by the user for neuron or synaptic models
#include "support_code.h"

void calcNeuronsCPU(float t)
 {
    // neuron group AdEx1
     {
        glbSpkCntAdEx1[0] = 0;
        
        for (int n = 0; n < 1; n++) {
            float lV = VAdEx1[n];
            float lw = wAdEx1[n];
            float lI0 = I0AdEx1[n];
            
            // test whether spike condition was fulfilled previously
            bool oldSpike= ((lV > 29.99f));
            // calculate membrane potential
            
	double curr_v = lV;
    if (lV > 0.0f){
		lV=(-47.0000f);
		lw+=(120.000f);
		} 
	else {
		lV+=((lI0 - ((20.0000f)*(lV - (-70.0000f))) + ((20.0000f)*(2.00000f)*expf((lV - (-50.0000f)) / (2.00000f))) - lw)/(100.000f))*DT;
		lw+= ((((1.00000f)*(curr_v - (-70.0000f))) - lw) / (90.0000f))*DT; 
				if (lV > 30.0f){
					lV=30.0f;
					} 
		}
		
            // test for and register a true spike
            if (((lV > 29.99f)) && !(oldSpike)) {
                glbSpkAdEx1[glbSpkCntAdEx1[0]++] = n;
                sTAdEx1[n] = t;
                }
            VAdEx1[n] = lV;
            wAdEx1[n] = lw;
            I0AdEx1[n] = lI0;
            }
        }
    
    // neuron group AdEx2
     {
        glbSpkCntAdEx2[0] = 0;
        
        for (int n = 0; n < 1; n++) {
            float lV = VAdEx2[n];
            float lw = wAdEx2[n];
            float lI0 = I0AdEx2[n];
            
            float Isyn = 0;
            Isyn += inSynAdEx1AdEx2[n]; inSynAdEx1AdEx2[n]= 0;
            // test whether spike condition was fulfilled previously
            bool oldSpike= ((lV > 29.99f));
            // calculate membrane potential
            
	double curr_v = lV;
    if (lV > 0.0f){
		lV=(-47.0000f);
		lw+=(120.000f);
		} 
	else {
		lV+=((lI0 - ((20.0000f)*(lV - (-70.0000f))) + ((20.0000f)*(2.00000f)*expf((lV - (-50.0000f)) / (2.00000f))) - lw)/(100.000f))*DT;
		lw+= ((((1.00000f)*(curr_v - (-70.0000f))) - lw) / (90.0000f))*DT; 
				if (lV > 30.0f){
					lV=30.0f;
					} 
		}
		
            // test for and register a true spike
            if (((lV > 29.99f)) && !(oldSpike)) {
                glbSpkAdEx2[glbSpkCntAdEx2[0]++] = n;
                }
            VAdEx2[n] = lV;
            wAdEx2[n] = lw;
            I0AdEx2[n] = lI0;
            // the post-synaptic dynamics
            
            }
        }
    
    }

    #endif
