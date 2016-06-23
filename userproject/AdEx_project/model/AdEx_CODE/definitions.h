

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
#define spikeCount_AdEx1 glbSpkCntAdEx1[0]
#define spike_AdEx1 glbSpkAdEx1
// neuron variables
extern unsigned int * glbSpkCntAdEx1;
extern unsigned int * glbSpkAdEx1;
extern float * VAdEx1;
extern float * wAdEx1;
extern float * I0AdEx1;

// synapse variables

#endif
