/*--------------------------------------------------------------------------
Author: Shashank









--------------------------------------------------------------------------*/


#ifndef ADEX_H 
#define ADEX_H

#include "AdEx.cc"

class neuronpop
{
public:
	NNmodel model;
	unsigned int sumAdEx1;
	neuronpop();
	~neuronpop();
	void init(unsigned int);
	void run(float, unsigned int);
#ifndef CPU_ONLY
	void getSpikesFromGPU();
	void getSpikeNumbersFromGPU();
#endif 
	void output_state(FILE *, unsigned int);
	void output_spikes(FILE *, unsigned int);
	void sum_spikes();
};

#endif
