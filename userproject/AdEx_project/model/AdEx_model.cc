/*--------------------------------------------------------------------------
Author: Shashank









--------------------------------------------------------------------------*/

#ifndef _ADEX_MODEL_CC_
#define _ADEX_MODEL_CC_

#include "AdEx_CODE/runner.cc"

neuronpop::neuronpop()
{
	modelDefinition(model);
	allocateMem();
	initialize();
	sumAdEx1 = 0;
}

void neuronpop::init(unsigned int which)
{
	if (which == CPU) {
	}
	if (which == GPU) {
#ifndef CPU_ONLY
		copyStateToDevice();
#endif
	}
}


neuronpop::~neuronpop()
{
	freeMem();
}

void neuronpop::run(float runtime, unsigned int which)
{
	int riT = (int)(runtime / DT + 1e-6);

	for (int i = 0; i < riT; i++) {
		if (which == GPU){
#ifndef CPU_ONLY
			stepTimeGPU();
#endif
		}
		if (which == CPU)
			stepTimeCPU();
	}
}

void neuronpop::sum_spikes()
{
	sumAdEx1 += glbSpkCntAdEx1[0];
}

//--------------------------------------------------------------------------
// output functions

void neuronpop::output_state(FILE *f, unsigned int which)
{
	if (which == GPU)
#ifndef CPU_ONLY
		copyStateFromDevice();
#endif

	fprintf(f, "%f ", t);

	for (int i = 0; i < model.neuronN[0] - 1; i++) {
		fprintf(f, "%f ", VAdEx1[i]);
	}

	fprintf(f, "\n");
}

#ifndef CPU_ONLY
//--------------------------------------------------------------------------
/*! \brief Method for copying all spikes of the last time step from the GPU

This is a simple wrapper for the convenience function copySpikesFromDevice() which is provided by GeNN.
*/
//--------------------------------------------------------------------------

void neuronpop::getSpikesFromGPU()
{
	copySpikesFromDevice();
}

//--------------------------------------------------------------------------
/*! \brief Method for copying the number of spikes in all neuron populations that have occurred during the last time step

This method is a simple wrapper for the convenience function copySpikeNFromDevice() provided by GeNN.
*/
//--------------------------------------------------------------------------

void neuronpop::getSpikeNumbersFromGPU()
{
	copySpikeNFromDevice();
}
#endif

void neuronpop::output_spikes(FILE *f, unsigned int which)
{

	for (int i = 0; i < glbSpkCntAdEx1[0]; i++) {
		fprintf(f, "%f %d\n", t, glbSpkAdEx1[i]);
	}

}
#endif	

