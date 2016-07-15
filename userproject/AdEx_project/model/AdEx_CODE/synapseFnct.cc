

#ifndef _AdEx_synapseFnct_cc
#define _AdEx_synapseFnct_cc

//-------------------------------------------------------------------------
/*! \file synapseFnct.cc

\brief File generated from GeNN for the model AdEx containing the equivalent of the synapse kernel and learning kernel functions for the CPU only version.
*/
//-------------------------------------------------------------------------

void calcSynapseDynamicsCPU(float t)
 {
    // execute internal synapse dynamics if any
    }
void calcSynapsesCPU(float t)
 {
    unsigned int ipost;
    unsigned int ipre;
    float addtoinSyn;
    
    // synapse group AdEx1AdEx2
     {
        // process presynaptic events: True Spikes
        for (int i = 0; i < glbSpkCntAdEx1[0]; i++) {
            ipre = glbSpkAdEx1[i];
            for (ipost = 0; ipost < 1; ipost++) {
                  addtoinSyn = gAdEx1AdEx2[ipre * 1 + ipost];
  inSynAdEx1AdEx2[ipost] += addtoinSyn;

                }
            }
        }
    
    }


#endif
