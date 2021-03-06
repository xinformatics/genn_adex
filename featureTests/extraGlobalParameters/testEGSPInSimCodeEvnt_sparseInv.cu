
#ifndef TESTEGSPINSIMCODEEVNT_CU
#define TESTEGSPINSIMCODEEVNT_CU

#include <cstdlib>
#include <cfloat>
#include <iostream>
#include <fstream>

using namespace std;

#include "hr_time.cpp"
#include "utils.h"
#include "testHelper.h"

#include "testEGSPInSimCodeEvnt_sparseInv.h"
#include "EGSPInSimCodeEvnt_sparseInv_CODE/definitions.h"
#include "EGSPInSimCodeEvnt_sparseInv_CODE/runner.cc"



EGSPInSimCodeEvnt_sparseInv::EGSPInSimCodeEvnt_sparseInv()
{
  allocateMem();
  initialize();
  init_synapses();
  init_neurons();
}

EGSPInSimCodeEvnt_sparseInv::~EGSPInSimCodeEvnt_sparseInv()
{
  freeMem();
  delete[] theW;
}

void EGSPInSimCodeEvnt_sparseInv::init_synapses() {
    // fill the sparse connections with a cyclic 1-to-1 scheme
    SparseProjection  *theC;
    for (int i= 0; i < 10; i++) { // all different delay groups get same connectivity
	switch (i) {
	case 0:
	    allocatesyn0(10);
	    theC= &Csyn0;
	    break;
	case 1:
	    allocatesyn1(10);
	    theC= &Csyn1;
	    break;
	case 2:
	    allocatesyn2(10);
	    theC= &Csyn2;
	    break;
	case 3:
	    allocatesyn3(10);
	    theC= &Csyn3;
	    break;
	case 4:
	    allocatesyn4(10);
	    theC= &Csyn4;
	    break;
	case 5:
	    allocatesyn5(10);
	    theC= &Csyn5;
	    break;
	case 6:
	    allocatesyn6(10);
	    theC= &Csyn6;
	    break;
	case 7:
	    allocatesyn7(10);
	    theC= &Csyn7;
	    break;
	case 8:
	    allocatesyn8(10);
	    theC= &Csyn8;
	    break;
	case 9:
	    allocatesyn9(10);
	    theC= &Csyn9;
	    break;
	}
	for (int j= 0; j < 10; j++) { // loop through pre-synaptic neurons
	    // each pre-synatic neuron gets one target neuron
	    unsigned int trg= (j+1)%10;
	    theC->indInG[j]= j;
	    theC->ind[j]= trg;
	}
	theC->indInG[10]= 10;
    }	
    theW= new float*[10];
    theW[0]= wsyn0;
    theW[1]= wsyn1;
    theW[2]= wsyn2;
    theW[3]= wsyn3;
    theW[4]= wsyn4;
    theW[5]= wsyn5;
    theW[6]= wsyn6;
    theW[7]= wsyn7;
    theW[8]= wsyn8;
    theW[9]= wsyn9;
    for (int i= 0; i < 10; i++) { // for all synapse groups
	for (int j= 0; j < 10; j++) { // for all synapses
	    theW[i][j]= 0.0f;
	} 
    }
    theThresh= new float*[10];
    theThresh[0]= &threshsyn0;
    theThresh[1]= &threshsyn1;
    theThresh[2]= &threshsyn2;
    theThresh[3]= &threshsyn3;
    theThresh[4]= &threshsyn4;
    theThresh[5]= &threshsyn5;
    theThresh[6]= &threshsyn6;
    theThresh[7]= &threshsyn7;
    theThresh[8]= &threshsyn8;
    theThresh[9]= &threshsyn9;
    initializeAllSparseArrays();
}


void EGSPInSimCodeEvnt_sparseInv::init_neurons() {
    for (int i= 0; i < 10; i++) {
	shiftpre[i]= i*10.0f;
    }
    copyStateToDevice();
}

void EGSPInSimCodeEvnt_sparseInv::run(int which)
{
  if (which == GPU)
  {
    stepTimeGPU();
    copyStateFromDevice();
  }
  else
  {
    stepTimeCPU();
  }
}


/*====================================================================
--------------------------- MAIN FUNCTION ----------------------------
====================================================================*/

int main(int argc, char *argv[])
{
  if (argc != 4)
  {
    cerr << "usage: testEGSPInSimCodeEvnt_sparseInv <GPU = 1, CPU = 0> <output label> <write output files? 0/1>" << endl;
    return EXIT_FAILURE;
  }

  EGSPInSimCodeEvnt_sparseInv *sim = new EGSPInSimCodeEvnt_sparseInv();
  int which= atoi(argv[1]);
  int write= atoi(argv[3]);
  CStopWatch *timer = new CStopWatch();
  string outLabel = toString(argv[2]);
  ofstream timeOs;
  ofstream neurOs;
  ofstream synOs;
  ofstream expSynOs;
  if (write) {
      timeOs.open((outLabel + "_time.dat").c_str(), ios::app);
      neurOs.open((outLabel + "_neur.dat").c_str());
      synOs.open((outLabel + "_syn.dat").c_str());
      expSynOs.open((outLabel + "_expSyn.dat").c_str());
  }
  float x[10][10];
  if (write) {
      cout << "# DT " << DT << endl;
      cout << "# TOTAL_TIME " << TOTAL_TIME << endl;
      cout << "# REPORT_TIME " << REPORT_TIME << endl;
      cout << "# begin simulating on " << ((which) ? "GPU" : "CPU") << endl;
  }
  timer->startTimer();
  float err= 0.0f;
  for (int d= 0; d < 10; d++) {
      for (int j= 0; j < 10; j++) {
	  x[d][j]= 0.0f;
      }
  }
  float swapT= TOTAL_TIME/2.0f;
  float theSwap= TOTAL_TIME;

  for (int i= 0; i < 10; i++) {
      // different steps in the second half
      *(sim->theThresh[i])= 2.0f*(i+1);
  }
  for (int i = 0; i < (TOTAL_TIME / DT); i++)
  {      
      t = i*DT;
      if (write) {
	  neurOs << t << " ";
	  synOs << t << " ";
	  expSynOs << t << " ";
      }
      if (abs(t- swapT) < DT) {
	  for (int i= 0; i < 10; i++) {
	      // different steps in the second half
	      *(sim->theThresh[i])= 2.0f+3*i;
	  }
	  theSwap= t;
      }
      for (int d= 0; d < 10; d++) { // for each delay
	  for (int j= 0; j < 10; j++) { // for all pre-synaptic neurons 
              // generate expected values
	      float evntT= t-2*DT-d*DT+5e-5;
	      if (evntT < theSwap) { // before the swap
		  if ((t > d*DT+0.1001) && (fmod(evntT+10*j,(float) (2*(d+1))) < 1e-4)) {
		      x[d][j]= t-2*DT-d*DT+10*j;
		  }
	      }
	      else { // after the swap
		  if ((t > d*DT+0.1001) && (fmod(evntT+10*j,(float) (2.0f+3*d)) < 1e-4)) {
		      x[d][j]= t-2*DT-d*DT+10*j;
		  }
	      }
	      if (write) {
		  synOs << sim->theW[d][j] << " ";
		  expSynOs << x[d][j] << " ";
	      }
	  }		  
	  err+= absDiff(x[d], sim->theW[d], 10);
	  if (write) {
	      synOs << "    ";
	      expSynOs << "    ";
	  }
      }
      if (write) {
	  for (int j= 0; j < 10; j++) {
	      neurOs << xpre[spkQuePtrpre*10+j] << " ";
	  }
	  neurOs << "    ";
      }
      neurOs << endl;
      synOs << endl;
      expSynOs << endl;
      sim->run(which);
      if (fmod(t+5e-5, REPORT_TIME) < 1e-4)
      {
	  cout << "\r" << t;
      }
  }
  cout << "\r";
  timer->stopTimer();
  cout << "# done in " << timer->getElapsedTime() << " seconds" << endl;
  if (write) {
      timeOs << timer->getElapsedTime() << endl;
      timeOs.close();
      neurOs.close();
      synOs.close();
      expSynOs.close();
  }

  delete sim;
  delete timer;
  
  float tolerance= 1e-3;
  int success;
  string result;
  if (abs(err) < tolerance) {
      success= 1;
      result= tS("\033[1;32m PASS \033[0m");
  } else {
      success= 0;
      result= tS("\033[1;31m FAIL \033[0m");
  }
  cout << "# test EGSPInSimCodeEvnt_sparseInv: Result " << result << endl;
  cout << "# the error was: " << err << " against tolerance " << tolerance << endl;
  cout << "#-----------------------------------------------------------" << endl;
  if (success)
      return EXIT_SUCCESS;
  else 
      return EXIT_FAILURE;
}

#endif // TESTEGSPINSIMCODEEVNT_CU
