/*--------------------------------------------------------------------------
Author: Shashank









--------------------------------------------------------------------------*/


#include <cassert>
using namespace std;
#include "hr_time.cpp"

#include "utils.h" // for CHECK_CUDA_ERRORS
#ifndef CPU_ONLY
#include <cuda_runtime.h>
#endif

// we will hard-code some stuff ... because at the end of the day that is 
// what we will do for the CUDA version

#define DBG_SIZE 10000

//----------------------------------------------------------------------
// other stuff:
#define T_REPORT_TME 100.0
#define TOTAL_TME 5000

CStopWatch timer;

#include "AdEx_model.h"
#include "AdEx_model.cc"

