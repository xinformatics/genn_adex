/*--------------------------------------------------------------------------
   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  
--------------------------------------------------------------------------*/

#include "AdEx_sim.h"

int main(int argc, char *argv[])
{
  if (argc != 3)
  {
    fprintf(stderr, "usage: AdEx_sim <basename> <CPU=0, GPU=1> \n");
    return 1;
  }
  int which= atoi(argv[2]);
  string OutDir = toString(argv[1]) +toString("_output");
  string name, name2; 

  name= OutDir+ toString("/") + toString(argv[1])+ toString(".time");
  FILE *timef= fopen(name.c_str(),"a"); 

  timer.startTimer();
  fprintf(stderr, "# DT %f \n", DT);
  fprintf(stderr, "# T_REPORT_TME %f \n", T_REPORT_TME);
  fprintf(stderr, "# TOTAL_TME %d \n", TOTAL_TME);
 
  name= OutDir+ toString("/") + toString(argv[1]) + toString(".out.Vm"); 
  FILE *osf= fopen(name.c_str(),"w");
  name2= OutDir+ toString("/") + toString(argv[1]) + toString(".explinp"); 
  FILE *osf2= fopen(name2.c_str(),"w");
  //-----------------------------------------------------------------
  // build the neuronal circuitry
  neuronpop AdExPop;
    
  AdExPop.init(which);         // this includes copying g's for the GPU version

  fprintf(stderr, "# neuronal circuitry built, start computation ... \n\n");
  unsigned int outno;
  if (AdExPop.model.neuronN[0]>10) 
  outno=10;
  else outno=AdExPop.model.neuronN[0];

  //------------------------------------------------------------------
  // output general parameters to output file and start the simulation

  fprintf(stderr, "# We are running with fixed time step %f \n", DT);
  fprintf(stderr, "# initial wait time execution ... \n");

  t= 0.0;
  int done= 0;
  float last_t_report=  t;
  while (!done) 
  {
	  if (t>1000.0 && t<3000.0){
	  /*if ((t>0.0 && t<5.0) || (t>1000.0 && t<1005.0) || (t>2000.0 && t<2005.0) || (t>3000.0 && t<3005.0) || (t>4000.0 && t<4005.0)){*/
		I0AdEx1[0]=380.0;
	  }
	  else {
		I0AdEx1[0]=0.0;
	  }
#ifndef CPU_ONLY
	  if (which == GPU) {
		copyStateToDevice();  
	  }
#endif
      AdExPop.run(DT, which); // run next batch
#ifndef CPU_ONLY        
      if (which == GPU) { 
	  AdExPop.getSpikeNumbersFromGPU();
	  //CHECK_CUDA_ERRORS(cudaMemcpy(VAdEx1, d_VAdEx1, outno*sizeof(float), cudaMemcpyDeviceToHost));
	  pullAdEx1StateFromDevice();
      } 
#endif
      AdExPop.sum_spikes();
      fprintf(osf, "%f ", t);
      
      for(int i=0;i<outno;i++) {
	  fprintf(osf, "%f ", VAdEx1[i]);
	  //fprintf(osf, "%f ", wAdEx1[i]);
	  fprintf(osf, "%f ", I0AdEx1[i]);
      }
      fprintf(osf, "\n");
      
      // report progress
      if (t - last_t_report >= T_REPORT_TME)
      {
	  fprintf(stderr, "time %f, V: %f \n", t, VAdEx1[0]);
	  last_t_report= t;
      }
      
      done= (t >= TOTAL_TME);
  }

  timer.stopTimer();
  fprintf(timef, "%d %d %u %f %f \n",which, AdExPop.model.neuronN[0], AdExPop.sumAdEx1, timer.getElapsedTime(),VAdEx1[0]);
//  cerr << "Output files are created under the current directory." << endl;
    cout << timer.getElapsedTime()  << endl;
  fclose(osf);
  fclose(osf2);
  fclose(timef);

  return 0;
}
