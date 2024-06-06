from ea.observer import time_observer
import pandas as pd

def generation_termination(population, num_generations, num_evaluations, args):
    """
    Return true when reached the maximum number of generations.
    """
    if num_generations == args["generations_budget"]:
        # store hypervolumes
        x = [x for x in range(1,len(args["hypervolume"])+1)]
        df = pd.DataFrame()
        df["generation"] = x
        df["hv"] = args["hypervolume"]
        df.to_csv(args["hypervolume_file_path"], sep=",",index=False)

        # store activation attempts
        time_observer(population, num_generations, num_evaluations, args)
    return num_generations == args["generations_budget"]