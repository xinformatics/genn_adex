
  Locust olfactory system (Nowotny et al. 2005) with user-defined synapses
  ========================================================================

This examples recapitulates the exact same model as MBody1_project,
but with user-defined model types for neurons and synapses. Also
sparse connectivity is used instead of dense. The way user-defined
types are used should be very instructive to advanced users wishing
to do the same with their models. This example project contains a
helper executable called "generate_run", which also prepares
additional synapse connectivity and input pattern data, before
compiling and executing the model.

To compile it, navigate to genn/userproject/MBody_userdef_project and type:
  nmake /f WINmakefile
for Windows users, or:
  make
for Linux, Mac and other UNIX users. 


  USAGE
  -----

  ./generate_run <0(CPU)/1(GPU)/n(GPU n-2)> <nAL> <nKC> <nLH> <nDN> <gScale> <DIR> <MODEL> 

Mandatory parameters:
CPU/GPU: Choose whether to run the simulation on CPU (`0`), auto GPU (`1`), or GPU (n-2) (`n`).
nAL: Number of neurons in the antennal lobe (AL), the input neurons to this model
nKC: Number of Kenyon cells (KC) in the "hidden layer"
nLH: Number of lateral horn interneurons, implementing gain control
nDN: Number of decision neurons (DN) in the output layer
gScale: A general rescaling factor for snaptic strength
outname: The base name of the output location and output files
model: The name of the model to execute, as provided this would be `MBody1`

Optional arguments:
DEBUG=0 or DEBUG=1 (default 0): Whether to run in a debugger
FTYPE=DOUBLE of FTYPE=FLOAT (default FLOAT): What floating point type to use
REUSE=0 or REUSE=1 (default 0): Whether to reuse generated connectivity from an earlier run
CPU_ONLY=0 or CPU_ONLY=1 (default 0): Whether to compile in (CUDA independent) "CPU only" mode.

An example invocation of generate_run is:

  ./generate_run 1 100 1000 20 100 0.0025 outname MBody_userdef REUSE=1
generate_run 1 100 1000 20 100 0.0025 compare MBody_userdef REUSE=1

Such a command would generate a locust olfaction model with 100
antennal lobe neurons, 1000 mushroom body Kenyon cells, 20 lateral
horn interneurons and 100 mushroom body output neurons, and launch
a simulation of it on a CUDA-enabled GPU using single precision
floating point numbers. All output files will be prefixed with
"outname" and will be created under the "outname" directory.

In more details, what generate_run program does is: 
a) use some other tools to generate the appropriate connectivity
   matrices and store them in files.

b) build the source code for the model by writing neuron numbers into
   ./model/sizes.h, and executing "genn-buildmodel.sh ./model/MBody_userdef.cc".  

c) compile the generated code by invoking "make clean && make" 
   running the code, e.g. "./classol_sim r1 1".

Another example of an invocation would be: 
  ./generate_run 0 100 1000 20 100 0.0025 outname MBody_userdef FTYPE=DOUBLE CPU_ONLY=1


  MODEL INFORMATION
  -----------------

For information regarding the locust olfaction model implemented in this example project, see:

T. Nowotny, R. Huerta, H. D. I. Abarbanel, and M. I. Rabinovich Self-organization in the
olfactory system: One shot odor recognition in insects, Biol Cyber, 93 (6): 436-446 (2005),
doi:10.1007/s00422-005-0019-7 
