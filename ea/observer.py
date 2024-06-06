from typing import Dict, List
import pandas as pd
import numpy as np
from pymoo.indicators.hv import Hypervolume

def ea_observer(population, num_generations, num_evaluations, args):
    # current best individual
    best = max(population)

    # population size
    population_size = len(population)

    print(f"OBSERVER\n[num generations:{num_generations}]\n[num evaluations:{num_evaluations}]\n[current best individual:{best}]\n[population size:{population_size}]\n")

def time_observer(population, num_generations, num_evaluations, args):
	"""
	Save Time (Activation Attempts) at the end of the evolutionary process.
	"""

	df = pd.DataFrame(args["time"])
	df.to_csv(args["activation_attempts_file_path"], index=False, header=None)
	return

def hypervolume_observer(population, num_generations, num_evaluations, args):
    # current best individual
    best = max(population)

    # population size
    population_size = len(population)

    # updating the Hypervolume list troughout the evolutionaty process
    
    # switch all the obj. functions' value to -(minus) in order to have a
    # minimization problem and compute the Hypervolume correctly respect to the
    # pymoo implementation taken by DEAP
    arch = [list(x.fitness) for x in args["_ec"].archive] 
    for i in range(len(arch)):
        for j in range(len(arch[i])):
            if float(arch[i][j])>=0:
                arch[i][j] = -float(arch[i][j])
    F =  np.array(arch)

    metric = Hypervolume(ref_point=np.array([0,0]),norm_ref_point=False,zero_to_one=False)
    hv = metric.do(F)
    args["hypervolume"].append(hv)

    print(f"OBSERVER\n[num generations:{num_generations}]\n[num evaluations:{num_evaluations}]\n[current best individual:{best}]\n[population size:{population_size}]\n[hypervolume:{hv}]\n")         