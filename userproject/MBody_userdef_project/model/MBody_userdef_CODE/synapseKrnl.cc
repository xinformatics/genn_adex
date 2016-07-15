

#ifndef _MBody_userdef_synapseKrnl_cc
#define _MBody_userdef_synapseKrnl_cc
#define BLOCKSZ_SYN 96

//-------------------------------------------------------------------------
/*! \file synapseKrnl.cc

\brief File generated from GeNN for the model MBody_userdef containing the synapse kernel and learning kernel functions.
*/
//-------------------------------------------------------------------------

extern "C" __global__ void calcSynapses(float t)
 {
    unsigned int id = BLOCKSZ_SYN * blockIdx.x + threadIdx.x;
    unsigned int lmax, j, r;
    float addtoinSyn;
    volatile __shared__ float shLg[BLOCKSZ_SYN];
    float linSyn;
    unsigned int ipost;
    unsigned int prePos; 
    unsigned int npost; 
    __shared__ unsigned int shSpk[BLOCKSZ_SYN];
    unsigned int lscnt, numSpikeSubsets;
    __shared__ unsigned int shSpkEvnt[BLOCKSZ_SYN];
    unsigned int lscntEvnt, numSpikeSubsetsEvnt;
    
    // synapse group PNKC
    if (id < 1056) {
        lscnt = dd_glbSpkCntPN[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkPN[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (id < 1000) {
                    prePos = dd_indInGPNKC[shSpk[j]];
                    npost = dd_indInGPNKC[shSpk[j] + 1] - prePos;
                    if (id < npost) {
                        prePos += id;
                        ipost = dd_indPNKC[prePos];
                        addtoinSyn = dd_gPNKC[prePos];
  atomicAdd(&dd_inSynPNKC[ipost], addtoinSyn);
  
                        }
                    }
                
                    }
            
                }
        
            
        }
    
    // synapse group PNLHI
    if ((id >= 1056) && (id < 1152)) {
        unsigned int lid = id - 1056;
        // only do this for existing neurons
        if (lid < 20) {
            linSyn = dd_inSynPNLHI[lid];
            }
        lscnt = dd_glbSpkCntPN[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: True Spikes
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkPN[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 20) {
                    ipost = lid;
                    addtoinSyn = dd_gPNLHI[shSpk[j] * 20+ ipost];
  linSyn += addtoinSyn;
  
                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 20) {
            dd_inSynPNLHI[lid] = linSyn;
            }
        }
    
    // synapse group LHIKC
    if ((id >= 1152) && (id < 2208)) {
        unsigned int lid = id - 1152;
        // only do this for existing neurons
        if (lid < 1000) {
            linSyn = dd_inSynLHIKC[lid];
            }
        lscntEvnt = dd_glbSpkCntEvntLHI[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntLHI[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 1000) {
                    ipost = lid;
                    addtoinSyn = (0.0200000f) * tanhf((dd_VLHI[shSpkEvnt[j]] - (-40.0000f)) / (50.0000f))* DT;
    if (addtoinSyn < 0) addtoinSyn = 0.0f;
    linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 1000) {
            dd_inSynLHIKC[lid] = linSyn;
            }
        }
    
    // synapse group KCDN
    if ((id >= 2208) && (id < 3264)) {
        unsigned int lid = id - 2208;
        lscnt = dd_glbSpkCntKC[0];
        numSpikeSubsets = (lscnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        if (lid < dd_glbSpkCntKC[0]) {
            int preInd = dd_glbSpkKC[lid];
            prePos = dd_indInGKCDN[preInd];
            npost = dd_indInGKCDN[preInd + 1] - prePos;
            for (int i = 0; i < npost; ++i) {
                	ipost = dd_indKCDN[prePos];
                addtoinSyn = dd_gKCDN[prePos];
					atomicAdd(&dd_inSynKCDN[ipost], addtoinSyn); 
					float dt = dd_sTDN[ipost] - t - ((10.0000f)); 
					float dg = 0;
					if (dt > (31.2500f))  
					dg = -((7.50000e-005f)) ; 
					else if (dt > 0.0f)  
					dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
					else if (dt > (-25.0125f))  
					dg = (1.20000e-005f) * dt + ((0.000300000f)); 
					else dg = - ((1.50000e-007f)) ; 
					dd_gRawKCDN[prePos] += dg; 
					dd_gKCDN[prePos]=(0.0150000f)/2.0f *(tanhf((33.3300f)*(dd_gRawKCDN[prePos] - ((0.00750000f))))+1.0f); 
					
                prePos += 1;
                }
            }
        
        }
    
    // synapse group DNDN
    if ((id >= 3264) && (id < 3456)) {
        unsigned int lid = id - 3264;
        // only do this for existing neurons
        if (lid < 100) {
            linSyn = dd_inSynDNDN[lid];
            }
        lscntEvnt = dd_glbSpkCntEvntDN[0];
        numSpikeSubsetsEvnt = (lscntEvnt+BLOCKSZ_SYN-1) / BLOCKSZ_SYN;
        // process presynaptic events: Spike type events
        for (r = 0; r < numSpikeSubsetsEvnt; r++) {
            if (r == numSpikeSubsetsEvnt - 1) lmax = ((lscntEvnt-1) % BLOCKSZ_SYN) +1;
            else lmax = BLOCKSZ_SYN;
            __syncthreads();
            if (threadIdx.x < lmax) {
                shSpkEvnt[threadIdx.x] = dd_glbSpkEvntDN[(r * BLOCKSZ_SYN) + threadIdx.x];
                }
            __syncthreads();
            // loop through all incoming spikes
            for (j = 0; j < lmax; j++) {
                // only work on existing neurons
                if (lid < 100) {
                    ipost = lid;
                    addtoinSyn = (0.0500000f) * tanhf((dd_VDN[shSpkEvnt[j]] - (-30.0000f)) / (50.0000f))* DT;
    if (addtoinSyn < 0) addtoinSyn = 0.0f;
    linSyn += addtoinSyn;

                    }
                
                    }
            
                }
        
            
        // only do this for existing neurons
        if (lid < 100) {
            dd_inSynDNDN[lid] = linSyn;
            }
        }
    
    }

extern "C" __global__ void learnSynapsesPost(float t)
 {
    unsigned int id = 32 * blockIdx.x + threadIdx.x;
    __shared__ unsigned int shSpk[32];
    unsigned int lscnt, numSpikeSubsets, lmax, j, r;
    
    // synapse group KCDN
    if (id < 1024) {
        lscnt = dd_glbSpkCntDN[0];
        numSpikeSubsets = (lscnt+31) / 32;
        for (r = 0; r < numSpikeSubsets; r++) {
            if (r == numSpikeSubsets - 1) lmax = ((lscnt-1) % 32)+1;
            else lmax = 32;
            if (threadIdx.x < lmax) {
                shSpk[threadIdx.x] = dd_glbSpkDN[(r * 32) + threadIdx.x];
                }
            __syncthreads();
            // only work on existing neurons
            if (id < 1000) {
                // loop through all incoming spikes for learning
                for (j = 0; j < lmax; j++) {
                    
                unsigned int iprePos = dd_revIndInGKCDN[shSpk[j]];
                    unsigned int npre = dd_revIndInGKCDN[shSpk[j] + 1] - iprePos;
                    if (id < npre) {
                        iprePos += id;
                        float dt = t - (dd_sTKC[dd_revIndKCDN[iprePos]]) - ((10.0000f)); 
				   float dg =0; 
				   if (dt > (31.2500f))  
				   dg = -((7.50000e-005f)) ; 
 				   else if (dt > 0.0f)  
			           dg = (-1.20000e-005f) * dt + ((0.000300000f)); 
				   else if (dt > (-25.0125f))  
				   dg = (1.20000e-005f) * dt + ((0.000300000f)); 
				   else dg = -((1.50000e-007f)) ; 
				   dd_gRawKCDN[dd_remapKCDN[iprePos]] += dg; 
				   dd_gKCDN[dd_remapKCDN[iprePos]]=(0.0150000f)/2.0f *(tanhf((33.3300f)*(dd_gRawKCDN[dd_remapKCDN[iprePos]] - ((0.00750000f))))+1.0f); 
				  
                        }
                    }
                }
            }
        __syncthreads();
        if (threadIdx.x == 0) {
            j = atomicAdd((unsigned int *) &d_done, 1);
            if (j == 31) {
                dd_glbSpkCntPN[0] = 0;
                dd_glbSpkCntKC[0] = 0;
                dd_glbSpkCntEvntLHI[0] = 0;
                dd_glbSpkCntLHI[0] = 0;
                dd_glbSpkCntEvntDN[0] = 0;
                dd_glbSpkCntDN[0] = 0;
                d_done = 0;
                }
            }
        }
    }

#endif
