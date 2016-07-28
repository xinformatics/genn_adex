/*--------------------------------------------------------------------------

   Author: Thomas Nowotny
  
   Institute: Institute for Nonlinear Dynamics
              University of California San Diego
              La Jolla, CA 92093-0402
  
   email to:  tnowotny@ucsd.edu
  
   initial version: 2002-09-26
  -----------------------------------------------------------------------
  Additonal Author: Shashank Yadav, IIT Delhi
  I implemented the AdEx model for faster simulations
--------------------------------------------------------------------------*/
#define DT 0.01
#include "modelSpec.h"
#include "modelSpec.cc"
#include "sizes.h"

/* //AdEx model parameters
// original 281.0, 30.0, -70.6, -50.4, 2, 144.0, 4.0, 80.5, -67.35
//  200 12 −70 −50 2 2 300 60 −58 – 500
naud et. al simulation
% 200 10 -70 -50 2 30  2    0 -58  500 4a
% 200 12 -70 -50 2 300 2   60 -58  500 4b
% 130 18 -58 -50 2 150 4  120 -50  400 4c
% 200 10 -58 -50 2 120 2  100 -46  210 4d
% 200 12 -70 -50 2 300 -10  0 -58  300 4e
% 200 12 -70 -50 2 300 -06  0  -58  110 4f
% 100 10 -65 -50 2  90 -10 30  -47  350 4g
% 100 12 -60 -50 2 130 -11 30  -48  160 4h
*/
// current state 4f
double AdEx_p[9] = {
	100.0,		// 0 - c (capacitance)
	20.0,		// 1 - g_l (conductance)
	-70.0, 	// 2 - l_v (leak_potential)
	-50.0,	// 3 - s_t (spike thresold)
	2.0,		// 4 - s_f (slope_factor)
	90.0,	// 5 - t_w (tau_w)
	1.0,	// 6 - a
	120.0,	// 7 - b
	-47,	// 8 - v_r (reset potential)
};

double AdExvar_ini[3] = {
	//AdEx model initial conditions
	-70.0,	//0 - V
	0.0,	//1 - w
	0.0	//2 - I0
};

// synapse parameters

double mySyn_p[3] = {
	0.0,           // 0 - Erev: Reversal potential
	-20.0,         // 1 - Epre: Presynaptic threshold potential
	1.0            // 2 - tau_S: decay time constant for S [ms]
};

double mySyn_ini[1] = {
	0.0 //initial values of g
};

double postExp[2] = {
	1.0,            // 0 - tau_S: decay time constant for S [ms]
	0.0		  // 1 - Erev: Reversal potential
};
double *postSynV = NULL;

// synapse parameters end-------------------------------------

void modelDefinition(NNmodel &model)
{
	initGeNN();
#ifndef CPU_ONLY
	model.setGPUDevice(0);
#endif
	model.setName("AdEx");

	neuronModel n;
	// variables
	n.varNames.clear();
	n.varTypes.clear();
	n.varNames.push_back(tS("V"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("w"));
	n.varTypes.push_back(tS("scalar"));
	n.varNames.push_back(tS("I0"));
	n.varTypes.push_back(tS("scalar"));
	n.pNames.clear();
	// parameters
	n.pNames.push_back(tS("c"));
	n.pNames.push_back(tS("g_l"));
	n.pNames.push_back(tS("l_v"));
	n.pNames.push_back(tS("s_t"));
    n.pNames.push_back(tS("s_f"));
	n.pNames.push_back(tS("t_w")); 
	n.pNames.push_back(tS("a"));
	n.pNames.push_back(tS("b"));
	n.pNames.push_back(tS("v_r"));
	//n.dpNames.clear();
	
	n.simCode = tS("\n\
	double curr_v = $(V);\n\
    if ($(V) > 0.0){\n\
		$(V)=$(v_r);\n\
		$(w)+=$(b);\n\
		} \n\
	else {\n\
		$(V)+=(($(I0) - ($(g_l)*($(V) - $(l_v))) + ($(g_l)*$(s_f)*exp(($(V) - $(s_t)) / $(s_f))) - $(w))/$(c))*DT;\n\
		$(w)+= ((($(a)*(curr_v - $(l_v))) - $(w)) / $(t_w))*DT; \n\
				if ($(V) > 30.0){\n\
					$(V)=30.0;\n\
					} \n\
		}\n\
		");
	n.thresholdConditionCode = tS("($(V) > 29.99)");
	//n.dps = NULL;
	unsigned int MYADEX = nModels.size();
	nModels.push_back(n);
	//unsigned int ADEX  = nModels.size() - 1;
	model.addNeuronPopulation("AdEx1", _NC1, MYADEX, AdEx_p, AdExvar_ini);
	model.addNeuronPopulation("AdEx2", _NC1, MYADEX, AdEx_p, AdExvar_ini);
	model.addSynapsePopulation("AdEx1AdEx2", NSYNAPSE, ALLTOALL, INDIVIDUALG, NO_DELAY, IZHIKEVICH_PS, "AdEx1", "AdEx2", mySyn_ini, mySyn_p, postSynV, postExp);

	model.setPrecision(GENN_FLOAT);
	model.finalize();
}
