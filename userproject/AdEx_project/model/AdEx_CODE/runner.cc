

//-------------------------------------------------------------------------
/*! \file runner.cc

\brief File generated from GeNN for the model AdEx containing general control code.
*/
//-------------------------------------------------------------------------

#include <cstdio>
#include <cassert>
#include <ctime>
#include <stdint.h>
#include "utils.h"

#include "numlib/simpleBit.h"

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
unsigned int * glbSpkCntAdEx1;
unsigned int * glbSpkAdEx1;
float * VAdEx1;
float * wAdEx1;
float * I0AdEx1;

// synapse variables

#include "sparseUtils.cc"

#include "neuronFnct.cc"
void allocateMem()
{
glbSpkCntAdEx1 = new unsigned int[1];
glbSpkAdEx1 = new unsigned int[1];
VAdEx1 = new float[1];
wAdEx1 = new float[1];
I0AdEx1 = new float[1];

}

//-------------------------------------------------------------------------
/*! \brief Function to (re)set all model variables to their compile-time, homogeneous initial values.
 Note that this typically includes synaptic weight values. The function (re)sets host side variables and copies them to the GPU device.
*/
//-------------------------------------------------------------------------

void initialize()
{
    srand((unsigned int) time(NULL));

    // neuron variables
    glbSpkCntAdEx1[0] = 0;
    for (int i = 0; i < 1; i++) {
        glbSpkAdEx1[i] = 0;
    }
    for (int i = 0; i < 1; i++) {
        VAdEx1[i] = -70.0000f;
    }
    for (int i = 0; i < 1; i++) {
        wAdEx1[i] = 0.000000f;
    }
    for (int i = 0; i < 1; i++) {
        I0AdEx1[i] = 0.000000f;
    }

    // synapse variables


}

void initAdEx()
 {
    
}

    void freeMem()
{
    delete[] glbSpkCntAdEx1;
    delete[] glbSpkAdEx1;
    delete[] VAdEx1;
    delete[] wAdEx1;
    delete[] I0AdEx1;
}

void exitGeNN(){
  freeMem();
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
    calcNeuronsCPU(t);
iT++;
t= iT*DT;
}
