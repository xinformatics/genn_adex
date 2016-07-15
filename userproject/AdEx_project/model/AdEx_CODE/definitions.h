

//-------------------------------------------------------------------------
/*! \file definitions.h

\brief File generated from GeNN for the model AdEx containing useful Macros used for both GPU amd CPU versions.
*/
//-------------------------------------------------------------------------

#ifndef DEFINITIONS_H
#define DEFINITIONS_H
#ifndef DT
#define DT 0.01
#endif
#ifndef scalar
typedef float scalar;
#endif
#ifndef SCALAR_MIN
#define SCALAR_MIN 1.17549e-038f
#endif
#ifndef SCALAR_MAX
#define SCALAR_MAX 3.40282e+038f
#endif
#define glbSpkShiftAdEx1 0
#define glbSpkShiftAdEx2 0
#define spikeCount_AdEx1 glbSpkCntAdEx1[0]
#define spike_AdEx1 glbSpkAdEx1
#define spikeCount_AdEx2 glbSpkCntAdEx2[0]
#define spike_AdEx2 glbSpkAdEx2
// neuron variables
extern unsigned int * glbSpkCntAdEx1;
extern unsigned int * glbSpkAdEx1;
extern float * sTAdEx1;
extern float * VAdEx1;
extern float * wAdEx1;
extern float * I0AdEx1;
extern unsigned int * glbSpkCntAdEx2;
extern unsigned int * glbSpkAdEx2;
extern float * VAdEx2;
extern float * wAdEx2;
extern float * I0AdEx2;

// synapse variables
extern float * inSynAdEx1AdEx2;
extern float * gAdEx1AdEx2;

#endif
