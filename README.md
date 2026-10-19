1. Introduction
This is a simple tool designed to simulate one or multiple dPCR experiments. I built it in order to deepen my own understanding of the dPCR technique, and to 
model how variables like partition number, initial sample concentration, and PCR error parameters affect the accuracy of the technique. It has two main modes; 
single-experiment, and multiple parameter sweep. The former simulates a single experiment with parameters entered at the command line; the latter simulates multiple 
experiments with parameters fed in through a csv file.

2. Features
This simulator models one or multiple arrays of partitions using either a Poisson or multinomial occupancy model. It then "digitalizes" the signal from each part-
ition into a binary presence/absence call, also incorporating realistic error rates, in the form of a per-molecule independent probability of a "false-negative" 
and a universal "false-positive" rate. It then recovers the "lambda" or estimated concentration from the array, as well as providing summary statistics such as the 
error rate, mean recovered lambda, and standard deviation of recovered lambdas.

3. Installation
To install, Python3.12 or above is required. 
git clone dpcr_simulator
cd dpcr_simulator
pip install -r requirements.txt

4. Example of a single-run usage 

The parameters are as follows:
	-p is the number of partitions in the simulated array
	-n is the number of molecules of the analyte in the initial solution
	-m is the mode: options are poisson or multinomial 
	-fp is the false positive rate (between 0 and 1)
	-fn is the false negative rate (per molecule probability, between 0 and 1)
	-s is the seed for the rng
  
python3.12 dpcr_simulator.py \
	-p 20000\
	-n 1400 \
	-m poisson
	-fp 0.001
	-fn 0.05
	-s 7035 

5. Example of a bash sweep usage

The parameters are as follows
	-c parameter file, in csv format, WITH NO HEADER, consisting of columns p,n,mode,fp,fn,seed, corresponding to the options for the single experiment usage
	-r repeats per parameter set
	-o your name for the csv file to which the output will be written

6.Assumptions

Some assumptions of the model are listed below:
	•Poisson mode treats partition occupancies as independent Poisson draws with mean (lambda=N/P)
	•Multinomial mode fixes the molecule count exactly and randomly assigns molecules across partitions 
	•False positive probability is the probability across the whole partition array of flipping a genuinely empty partition
		•for this purpose, all genuinely empty partitions are treated identicaly, which may not be realistic
	•False negative probability is per molecule, so the chance that a partition with k molecules fails is p_fn^k

7. Output

The simulator outputs a "lambda_hat" as λ^= -ln ( N_negative/P).
It then evaluats this estimated quantity against the known input, λ_true, and, in sweep mode, calculates also the standard deviation and average error
for each experimental set.

8. Limitations 

There are a number of limitations to this simple simulation. These include, but are not limited to:
 
	- no explicit fluorescence-amplitude model
	- no rain / probabilistic classification
	- simplified false-positive and false-negative mechanisms
	- no explicit extraction or reverse-transcription efficiency
	- no partition-volume heterogeneity
	- no inhibition model yet

9. Background

For more information on dPCR, check out the primer that is part of this repository.  


