

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model MBody_userdef containing general control code.
*/
//-------------------------------------------------------------------------

#include <cstdio>
#include <cassert>
#include <ctime>
#include <stdint.h>
#include "utils.h"

#include "numlib/simpleBit.h"

#include "hr_time.cpp"
#define RUNNER_CC_COMPILE
#include "definitions.h"

#define Conductance SparseProjection
/*struct Conductance is deprecated. 
  By GeNN 2.0, Conductance is renamed as SparseProjection and contains only indexing values. 
  Please consider updating your user code by renaming Conductance as SparseProjection 
  and making g member a synapse variable.*/
#ifndef MYRAND
#define MYRAND(Y,X) Y = Y * 1103515245 + 12345; X = (Y >> 16);
#endif

#ifndef MYRAND_MAX
#define MYRAND_MAX 0x0000FFFFFFFFFFFFLL
#endif
cudaEvent_t neuronStart, neuronStop;
double neuron_tme;
CStopWatch neuron_timer;
cudaEvent_t synapseStart, synapseStop;
double synapse_tme;
CStopWatch synapse_timer;
cudaEvent_t learningStart, learningStop;
double learning_tme;
CStopWatch learning_timer;
#ifndef CHECK_CUDA_ERRORS
#define CHECK_CUDA_ERRORS(call) {\
    cudaError_t error = call;\
    if (error != cudaSuccess) {\
        fprintf(stderr, "%s: %i: cuda error %i: %s\n", __FILE__, __LINE__, (int) error, cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}
#endif

template<class T>
void deviceMemAllocate(T* hostPtr, const T &devSymbol, size_t size)
{
    void *devptr;
    CHECK_CUDA_ERRORS(cudaMalloc(hostPtr, size));
    CHECK_CUDA_ERRORS(cudaGetSymbolAddress(&devptr, devSymbol));
    CHECK_CUDA_ERRORS(cudaMemcpy(devptr, hostPtr, sizeof(void*), cudaMemcpyHostToDevice));
}

//-------------------------------------------------------------------------
/*! \brief Function to convert a firing probability (per time step) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given probability.
*/
//-------------------------------------------------------------------------

void convertProbabilityToRandomNumberThreshold(float *p_pattern, uint64_t *pattern, int N)
{
    float fac= pow(2.0, (double) sizeof(uint64_t)*8-16);
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (p_pattern[i]*fac);
    }
}

//-------------------------------------------------------------------------
/*! \brief Function to convert a firing rate (in kHz) 
to an integer of type uint64_t that can be used as a threshold for the GeNN random number generator to generate events with the given rate.
*/
//-------------------------------------------------------------------------

void convertRateToRandomNumberThreshold(float *rateKHz_pattern, uint64_t *pattern, int N)
{
    float fac= pow(2.0, (double) sizeof(uint64_t)*8-16)*DT;
    for (int i= 0; i < N; i++) {
        pattern[i]= (uint64_t) (rateKHz_pattern[i]*fac);
    }
}

// global variables
unsigned long long iT= 0;
float t;
// neuron variables
__device__ volatile unsigned int d_done;
unsigned int * glbSpkCntPN;
unsigned int * d_glbSpkCntPN;
__device__ unsigned int * dd_glbSpkCntPN;
unsigned int * glbSpkPN;
unsigned int * d_glbSpkPN;
__device__ unsigned int * dd_glbSpkPN;
float * sTPN;
float * d_sTPN;
__device__ float * dd_sTPN;
float * VPN;
float * d_VPN;
__device__ float * dd_VPN;
uint64_t * seedPN;
uint64_t * d_seedPN;
__device__ uint64_t * dd_seedPN;
float * spikeTimePN;
float * d_spikeTimePN;
__device__ float * dd_spikeTimePN;
uint64_t * ratesPN;
unsigned int offsetPN;
unsigned int * glbSpkCntKC;
unsigned int * d_glbSpkCntKC;
__device__ unsigned int * dd_glbSpkCntKC;
unsigned int * glbSpkKC;
unsigned int * d_glbSpkKC;
__device__ unsigned int * dd_glbSpkKC;
float * sTKC;
float * d_sTKC;
__device__ float * dd_sTKC;
float * VKC;
float * d_VKC;
__device__ float * dd_VKC;
float * wKC;
float * d_wKC;
__device__ float * dd_wKC;
unsigned int * glbSpkCntLHI;
unsigned int * d_glbSpkCntLHI;
__device__ unsigned int * dd_glbSpkCntLHI;
unsigned int * glbSpkLHI;
unsigned int * d_glbSpkLHI;
__device__ unsigned int * dd_glbSpkLHI;
unsigned int * glbSpkCntEvntLHI;
unsigned int * d_glbSpkCntEvntLHI;
__device__ unsigned int * dd_glbSpkCntEvntLHI;
unsigned int * glbSpkEvntLHI;
unsigned int * d_glbSpkEvntLHI;
__device__ unsigned int * dd_glbSpkEvntLHI;
float * sTLHI;
float * d_sTLHI;
__device__ float * dd_sTLHI;
float * VLHI;
float * d_VLHI;
__device__ float * dd_VLHI;
float * mLHI;
float * d_mLHI;
__device__ float * dd_mLHI;
float * hLHI;
float * d_hLHI;
__device__ float * dd_hLHI;
float * nLHI;
float * d_nLHI;
__device__ float * dd_nLHI;
unsigned int * glbSpkCntDN;
unsigned int * d_glbSpkCntDN;
__device__ unsigned int * dd_glbSpkCntDN;
unsigned int * glbSpkDN;
unsigned int * d_glbSpkDN;
__device__ unsigned int * dd_glbSpkDN;
unsigned int * glbSpkCntEvntDN;
unsigned int * d_glbSpkCntEvntDN;
__device__ unsigned int * dd_glbSpkCntEvntDN;
unsigned int * glbSpkEvntDN;
unsigned int * d_glbSpkEvntDN;
__device__ unsigned int * dd_glbSpkEvntDN;
float * sTDN;
float * d_sTDN;
__device__ float * dd_sTDN;
float * VDN;
float * d_VDN;
__device__ float * dd_VDN;
float * mDN;
float * d_mDN;
__device__ float * dd_mDN;
float * hDN;
float * d_hDN;
__device__ float * dd_hDN;
float * nDN;
float * d_nDN;
__device__ float * dd_nDN;

// synapse variables
float * inSynPNKC;
float * d_inSynPNKC;
__device__ float * dd_inSynPNKC;
SparseProjection CPNKC;
unsigned int *d_indInGPNKC;
__device__ unsigned int *dd_indInGPNKC;
unsigned int *d_indPNKC;
__device__ unsigned int *dd_indPNKC;
float * gPNKC;
float * d_gPNKC;
__device__ float * dd_gPNKC;
float * EEEEPNKC;
float * d_EEEEPNKC;
__device__ float * dd_EEEEPNKC;
float * inSynPNLHI;
float * d_inSynPNLHI;
__device__ float * dd_inSynPNLHI;
float * gPNLHI;
float * d_gPNLHI;
__device__ float * dd_gPNLHI;
float * inSynLHIKC;
float * d_inSynLHIKC;
__device__ float * dd_inSynLHIKC;
float * inSynKCDN;
float * d_inSynKCDN;
__device__ float * dd_inSynKCDN;
SparseProjection CKCDN;
unsigned int *d_indInGKCDN;
__device__ unsigned int *dd_indInGKCDN;
unsigned int *d_indKCDN;
__device__ unsigned int *dd_indKCDN;
unsigned int *d_revIndInGKCDN;
__device__ unsigned int *dd_revIndInGKCDN;
unsigned int *d_revIndKCDN;
__device__ unsigned int *dd_revIndKCDN;
unsigned int *d_remapKCDN;
__device__ unsigned int *dd_remapKCDN;
float * gKCDN;
float * d_gKCDN;
__device__ float * dd_gKCDN;
float * gRawKCDN;
float * d_gRawKCDN;
__device__ float * dd_gRawKCDN;
float * inSynDNDN;
float * d_inSynDNDN;
__device__ float * dd_inSynDNDN;

#include "sparseUtils.cc"

#include "runnerGPU.cc"

#include "neuronFnct.cc"
#include "synapseFnct.cc"
void allocateMem()
{
    CHECK_CUDA_ERRORS(cudaSetDevice(0));
    cudaEventCreate(&neuronStart);
    cudaEventCreate(&neuronStop);
    neuron_tme= 0.0;
    cudaEventCreate(&synapseStart);
    cudaEventCreate(&synapseStop);
    synapse_tme= 0.0;
    cudaEventCreate(&learningStart);
    cudaEventCreate(&learningStop);
    learning_tme= 0.0;
cudaHostAlloc(&glbSpkCntPN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntPN, dd_glbSpkCntPN, 1 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkPN, 100 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkPN, dd_glbSpkPN, 100 * sizeof(unsigned int));
cudaHostAlloc(&sTPN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_sTPN, dd_sTPN, 100 * sizeof(float));
cudaHostAlloc(&VPN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_VPN, dd_VPN, 100 * sizeof(float));
cudaHostAlloc(&seedPN, 100 * sizeof(uint64_t), cudaHostAllocPortable);
    deviceMemAllocate(&d_seedPN, dd_seedPN, 100 * sizeof(uint64_t));
cudaHostAlloc(&spikeTimePN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_spikeTimePN, dd_spikeTimePN, 100 * sizeof(float));

cudaHostAlloc(&glbSpkCntKC, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntKC, dd_glbSpkCntKC, 1 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkKC, 1000 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkKC, dd_glbSpkKC, 1000 * sizeof(unsigned int));
cudaHostAlloc(&sTKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_sTKC, dd_sTKC, 1000 * sizeof(float));
cudaHostAlloc(&VKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_VKC, dd_VKC, 1000 * sizeof(float));
cudaHostAlloc(&wKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_wKC, dd_wKC, 1000 * sizeof(float));

cudaHostAlloc(&glbSpkCntLHI, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntLHI, dd_glbSpkCntLHI, 1 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkLHI, 20 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkLHI, dd_glbSpkLHI, 20 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkCntEvntLHI, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntEvntLHI, dd_glbSpkCntEvntLHI, 1 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkEvntLHI, 20 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkEvntLHI, dd_glbSpkEvntLHI, 20 * sizeof(unsigned int));
cudaHostAlloc(&sTLHI, 20 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_sTLHI, dd_sTLHI, 20 * sizeof(float));
cudaHostAlloc(&VLHI, 20 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_VLHI, dd_VLHI, 20 * sizeof(float));
cudaHostAlloc(&mLHI, 20 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_mLHI, dd_mLHI, 20 * sizeof(float));
cudaHostAlloc(&hLHI, 20 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_hLHI, dd_hLHI, 20 * sizeof(float));
cudaHostAlloc(&nLHI, 20 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_nLHI, dd_nLHI, 20 * sizeof(float));

cudaHostAlloc(&glbSpkCntDN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntDN, dd_glbSpkCntDN, 1 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkDN, 100 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkDN, dd_glbSpkDN, 100 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkCntEvntDN, 1 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkCntEvntDN, dd_glbSpkCntEvntDN, 1 * sizeof(unsigned int));
cudaHostAlloc(&glbSpkEvntDN, 100 * sizeof(unsigned int), cudaHostAllocPortable);
    deviceMemAllocate(&d_glbSpkEvntDN, dd_glbSpkEvntDN, 100 * sizeof(unsigned int));
cudaHostAlloc(&sTDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_sTDN, dd_sTDN, 100 * sizeof(float));
cudaHostAlloc(&VDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_VDN, dd_VDN, 100 * sizeof(float));
cudaHostAlloc(&mDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_mDN, dd_mDN, 100 * sizeof(float));
cudaHostAlloc(&hDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_hDN, dd_hDN, 100 * sizeof(float));
cudaHostAlloc(&nDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_nDN, dd_nDN, 100 * sizeof(float));

cudaHostAlloc(&inSynPNKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynPNKC, dd_inSynPNKC, 1000 * sizeof(float));
cudaHostAlloc(&EEEEPNKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_EEEEPNKC, dd_EEEEPNKC, 1000 * sizeof(float));

cudaHostAlloc(&inSynPNLHI, 20 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynPNLHI, dd_inSynPNLHI, 20 * sizeof(float));
cudaHostAlloc(&gPNLHI, 2000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_gPNLHI, dd_gPNLHI, 2000 * sizeof(float));

cudaHostAlloc(&inSynLHIKC, 1000 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynLHIKC, dd_inSynLHIKC, 1000 * sizeof(float));

cudaHostAlloc(&inSynKCDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynKCDN, dd_inSynKCDN, 100 * sizeof(float));

cudaHostAlloc(&inSynDNDN, 100 * sizeof(float), cudaHostAllocPortable);
    deviceMemAllocate(&d_inSynDNDN, dd_inSynDNDN, 100 * sizeof(float));

}

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize()
{
    srand((unsigned int) 1234);

    // neuron variables
    glbSpkCntPN[0] = 0;
    for (int i = 0; i < 100; i++) {
        glbSpkPN[i] = 0;
    }
    for (int i = 0; i < 100; i++) {
        sTPN[i] = -10.0;
    }
    for (int i = 0; i < 100; i++) {
        VPN[i] = -60.0000f;
    }
    for (int i = 0; i < 100; i++) {
        seedPN[i] = 0;
    }
    for (int i = 0; i < 100; i++) {
        spikeTimePN[i] = -10.0000f;
    }
    for (int i = 0; i < 100; i++) {
        seedPN[i] = rand();
    }
    glbSpkCntKC[0] = 0;
    for (int i = 0; i < 1000; i++) {
        glbSpkKC[i] = 0;
    }
    for (int i = 0; i < 1000; i++) {
        sTKC[i] = -10.0;
    }
    for (int i = 0; i < 1000; i++) {
        VKC[i] = -70.0000f;
    }
    for (int i = 0; i < 1000; i++) {
        wKC[i] = 0.000000f;
    }
    glbSpkCntLHI[0] = 0;
    for (int i = 0; i < 20; i++) {
        glbSpkLHI[i] = 0;
    }
    glbSpkCntEvntLHI[0] = 0;
    for (int i = 0; i < 20; i++) {
        glbSpkEvntLHI[i] = 0;
    }
    for (int i = 0; i < 20; i++) {
        sTLHI[i] = -10.0;
    }
    for (int i = 0; i < 20; i++) {
        VLHI[i] = -60.0000f;
    }
    for (int i = 0; i < 20; i++) {
        mLHI[i] = 0.0529324f;
    }
    for (int i = 0; i < 20; i++) {
        hLHI[i] = 0.317677f;
    }
    for (int i = 0; i < 20; i++) {
        nLHI[i] = 0.596121f;
    }
    glbSpkCntDN[0] = 0;
    for (int i = 0; i < 100; i++) {
        glbSpkDN[i] = 0;
    }
    glbSpkCntEvntDN[0] = 0;
    for (int i = 0; i < 100; i++) {
        glbSpkEvntDN[i] = 0;
    }
    for (int i = 0; i < 100; i++) {
        sTDN[i] = -10.0;
    }
    for (int i = 0; i < 100; i++) {
        VDN[i] = -60.0000f;
    }
    for (int i = 0; i < 100; i++) {
        mDN[i] = 0.0529324f;
    }
    for (int i = 0; i < 100; i++) {
        hDN[i] = 0.317677f;
    }
    for (int i = 0; i < 100; i++) {
        nDN[i] = 0.596121f;
    }

    // synapse variables
    for (int i = 0; i < 1000; i++) {
        inSynPNKC[i] = 0.000000f;
    }
    for (int i = 0; i < 1000; i++) {
        EEEEPNKC[i] = 0.000000f;
    }
    for (int i = 0; i < 20; i++) {
        inSynPNLHI[i] = 0.000000f;
    }
    for (int i = 0; i < 2000; i++) {
        gPNLHI[i] = 0.000000f;
    }
    for (int i = 0; i < 1000; i++) {
        inSynLHIKC[i] = 0.000000f;
    }
    for (int i = 0; i < 100; i++) {
        inSynKCDN[i] = 0.000000f;
    }
    for (int i = 0; i < 100; i++) {
        inSynDNDN[i] = 0.000000f;
    }


    copyStateToDevice();

    //initializeAllSparseArrays(); //I comment this out instead of removing to keep in mind that sparse arrays need to be initialised manually by hand later
}

void allocatePNKC(unsigned int connN){
// Allocate host side variables
  CPNKC.connN= connN;
cudaHostAlloc(&CPNKC.indInG, 101 * sizeof(unsigned int), cudaHostAllocPortable);
cudaHostAlloc(&CPNKC.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CPNKC.preInd= NULL;
  CPNKC.revIndInG= NULL;
  CPNKC.revInd= NULL;
  CPNKC.remap= NULL;
cudaHostAlloc(&gPNKC, CPNKC.connN * sizeof(float), cudaHostAllocPortable);
// Allocate device side variables
  deviceMemAllocate( &d_indInGPNKC, dd_indInGPNKC, sizeof(unsigned int) * (101));
  deviceMemAllocate( &d_indPNKC, dd_indPNKC, sizeof(unsigned int) * (CPNKC.connN));
deviceMemAllocate(&d_gPNKC, dd_gPNKC, sizeof(float)*(CPNKC.connN));
}

void createSparseConnectivityFromDensePNKC(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDensePNKC() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocatePNKC(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void allocateKCDN(unsigned int connN){
// Allocate host side variables
  CKCDN.connN= connN;
cudaHostAlloc(&CKCDN.indInG, 1001 * sizeof(unsigned int), cudaHostAllocPortable);
cudaHostAlloc(&CKCDN.ind, connN * sizeof(unsigned int), cudaHostAllocPortable);
  CKCDN.preInd= NULL;
cudaHostAlloc(&CKCDN.revIndInG, 101 * sizeof(unsigned int), cudaHostAllocPortable);
cudaHostAlloc(&CKCDN.revInd, connN * sizeof(unsigned int), cudaHostAllocPortable);
cudaHostAlloc(&CKCDN.remap, connN * sizeof(unsigned int), cudaHostAllocPortable);
cudaHostAlloc(&gKCDN, CKCDN.connN * sizeof(float), cudaHostAllocPortable);
cudaHostAlloc(&gRawKCDN, CKCDN.connN * sizeof(float), cudaHostAllocPortable);
// Allocate device side variables
  deviceMemAllocate( &d_indInGKCDN, dd_indInGKCDN, sizeof(unsigned int) * (1001));
  deviceMemAllocate( &d_indKCDN, dd_indKCDN, sizeof(unsigned int) * (CKCDN.connN));
  deviceMemAllocate( &d_revIndInGKCDN, dd_revIndInGKCDN, sizeof(unsigned int) * (101));
  deviceMemAllocate( &d_revIndKCDN, dd_revIndKCDN, sizeof(unsigned int) * (CKCDN.connN));
  deviceMemAllocate( &d_remapKCDN, dd_remapKCDN, sizeof(unsigned int) * (CKCDN.connN));
deviceMemAllocate(&d_gKCDN, dd_gKCDN, sizeof(float)*(CKCDN.connN));
deviceMemAllocate(&d_gRawKCDN, dd_gRawKCDN, sizeof(float)*(CKCDN.connN));
}

void createSparseConnectivityFromDenseKCDN(int preN,int postN, float *denseMatrix){
    gennError("The function createSparseConnectivityFromDenseKCDN() has been deprecated because with the introduction of synapse models that can be fully user-defined and may not contain a conductance variable g the existence condition for synapses has become ill-defined. \n Please use your own logic and use the general tools allocateKCDN(), countEntriesAbove(), and setSparseConnectivityFromDense().");
}

void initializeAllSparseArrays() {
size_t size;
size = CPNKC.connN;
  initializeSparseArray(CPNKC, d_indPNKC, d_indInGPNKC,100);
CHECK_CUDA_ERRORS(cudaMemcpy(d_gPNKC, gPNKC, sizeof(float) * size , cudaMemcpyHostToDevice));
size = CKCDN.connN;
  initializeSparseArray(CKCDN, d_indKCDN, d_indInGKCDN,1000);
  initializeSparseArrayRev(CKCDN,  d_revIndKCDN,  d_revIndInGKCDN,  d_remapKCDN,100);
CHECK_CUDA_ERRORS(cudaMemcpy(d_gKCDN, gKCDN, sizeof(float) * size , cudaMemcpyHostToDevice));
CHECK_CUDA_ERRORS(cudaMemcpy(d_gRawKCDN, gRawKCDN, sizeof(float) * size , cudaMemcpyHostToDevice));
}

void initMBody_userdef()
 {
    
createPosttoPreArray(1000, 100, &CKCDN);
    initializeAllSparseArrays();
    }

    void freeMem()
{
cudaFreeHost(glbSpkCntPN);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntPN));
cudaFreeHost(glbSpkPN);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkPN));
cudaFreeHost(sTPN);
    CHECK_CUDA_ERRORS(cudaFree(d_sTPN));
cudaFreeHost(VPN);
    CHECK_CUDA_ERRORS(cudaFree(d_VPN));
cudaFreeHost(seedPN);
    CHECK_CUDA_ERRORS(cudaFree(d_seedPN));
cudaFreeHost(spikeTimePN);
    CHECK_CUDA_ERRORS(cudaFree(d_spikeTimePN));
cudaFreeHost(glbSpkCntKC);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntKC));
cudaFreeHost(glbSpkKC);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkKC));
cudaFreeHost(sTKC);
    CHECK_CUDA_ERRORS(cudaFree(d_sTKC));
cudaFreeHost(VKC);
    CHECK_CUDA_ERRORS(cudaFree(d_VKC));
cudaFreeHost(wKC);
    CHECK_CUDA_ERRORS(cudaFree(d_wKC));
cudaFreeHost(glbSpkCntLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntLHI));
cudaFreeHost(glbSpkLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkLHI));
cudaFreeHost(glbSpkCntEvntLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntLHI));
cudaFreeHost(glbSpkEvntLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntLHI));
cudaFreeHost(sTLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_sTLHI));
cudaFreeHost(VLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_VLHI));
cudaFreeHost(mLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_mLHI));
cudaFreeHost(hLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_hLHI));
cudaFreeHost(nLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_nLHI));
cudaFreeHost(glbSpkCntDN);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntDN));
cudaFreeHost(glbSpkDN);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkDN));
cudaFreeHost(glbSpkCntEvntDN);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkCntEvntDN));
cudaFreeHost(glbSpkEvntDN);
    CHECK_CUDA_ERRORS(cudaFree(d_glbSpkEvntDN));
cudaFreeHost(sTDN);
    CHECK_CUDA_ERRORS(cudaFree(d_sTDN));
cudaFreeHost(VDN);
    CHECK_CUDA_ERRORS(cudaFree(d_VDN));
cudaFreeHost(mDN);
    CHECK_CUDA_ERRORS(cudaFree(d_mDN));
cudaFreeHost(hDN);
    CHECK_CUDA_ERRORS(cudaFree(d_hDN));
cudaFreeHost(nDN);
    CHECK_CUDA_ERRORS(cudaFree(d_nDN));
cudaFreeHost(inSynPNKC);
    CHECK_CUDA_ERRORS(cudaFree(d_inSynPNKC));
    CPNKC.connN= 0;
cudaFreeHost(CPNKC.indInG);
cudaFreeHost(CPNKC.ind);
cudaFreeHost(gPNKC);
    CHECK_CUDA_ERRORS(cudaFree(d_gPNKC));
cudaFreeHost(EEEEPNKC);
    CHECK_CUDA_ERRORS(cudaFree(d_EEEEPNKC));
cudaFreeHost(inSynPNLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_inSynPNLHI));
cudaFreeHost(gPNLHI);
    CHECK_CUDA_ERRORS(cudaFree(d_gPNLHI));
cudaFreeHost(inSynLHIKC);
    CHECK_CUDA_ERRORS(cudaFree(d_inSynLHIKC));
cudaFreeHost(inSynKCDN);
    CHECK_CUDA_ERRORS(cudaFree(d_inSynKCDN));
    CKCDN.connN= 0;
cudaFreeHost(CKCDN.indInG);
cudaFreeHost(CKCDN.ind);
cudaFreeHost(CKCDN.revIndInG);
cudaFreeHost(CKCDN.revInd);
cudaFreeHost(CKCDN.remap);
cudaFreeHost(gKCDN);
    CHECK_CUDA_ERRORS(cudaFree(d_gKCDN));
cudaFreeHost(gRawKCDN);
    CHECK_CUDA_ERRORS(cudaFree(d_gRawKCDN));
cudaFreeHost(inSynDNDN);
    CHECK_CUDA_ERRORS(cudaFree(d_inSynDNDN));
}

void exitGeNN(){
  freeMem();
  cudaDeviceReset();
}
// ------------------------------------------------------------------------
// Throw an error for "old style" time stepping calls
template <class T>
void stepTimeCPU(T arg1, ...)
 {
    
gennError("Since GeNN 2.2 the call to step time has changed to not take any arguments. You appear to attempt to pass arguments. This is no longer supported. See the GeNN 2.2. release notes and the manual for examples how to pass data like, e.g., Poisson rates and direct inputs, that were previously handled through arguments.");
    }

    // ------------------------------------------------------------------------
// the actual time stepping procedure
void stepTimeCPU()
{
        synapse_timer.startTimer();
        calcSynapsesCPU(t);
        synapse_timer.stopTimer();
        synapse_tme+= synapse_timer.getElapsedTime();
        learning_timer.startTimer();
        learnSynapsesPostHost(t);
        learning_timer.stopTimer();
        learning_tme+= learning_timer.getElapsedTime();
    neuron_timer.startTimer();
    calcNeuronsCPU(t);
    neuron_timer.stopTimer();
    neuron_tme+= neuron_timer.getElapsedTime();
iT++;
t= iT*DT;
}
