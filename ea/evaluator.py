import inspyred
from tqdm import tqdm
from joblib import Parallel, delayed

def ea_evaluator(candidates, args):
    hypergraph = args["hypergraph"]
    degree_dict = args["degree_dict"]
    hyperdegree_dict = args["hyperdegree_dict"]
    neighbor_dict = args["neighbor_dict"]
    incident_hyperedge_dict = args["incident_hyperedge_dict"]
    p_min = args["p_min"]
    p_max = args["p_max"]
    threshold = args["threshold"]
    model = args["propagation_model"]
    no_simulations = args["no_simulations"]
    max_hop = args["max_hop"]
    random_generator = args["random_generator"]
    fitness_function = args["fitness_function"]
    max_seed_nodes = args["max_seed_nodes"]
    n_threads = args["n_threads"]

    fitness = [None]*len(candidates)
    time_gen = [None]*len(candidates) # calculate Time (Activation Attempts) for every individual in the population 

    if n_threads == 1:
        for index, a in tqdm(enumerate(candidates), total=len(candidates), desc=f"Processing"):
            a_set = set(a)
            influence_mean, influence_std, time = fitness_function(
                hypergraph=hypergraph,
                degree_dict=degree_dict,
                neighbor_dict=neighbor_dict,
                incident_hyperedge_dict=incident_hyperedge_dict,
                a=a_set,
                t=threshold,
                p_min=p_min,
                p_max=p_max,
                no_simulations=no_simulations,
                max_hop=max_hop,
                model=model,
                random_generator=random_generator
            )
            fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / len(hypergraph.get_nodes())), ((max_seed_nodes+1-len(a_set))/max_seed_nodes)])
            time_gen[index] = time
    else:
        # populate the following list with the initial seed set of each
        # candidate in the population
        candidate_seed_sets = [set(a) for a in candidates]
        
        # process the candidates in parallel
        outputs = Parallel(n_threads)(
            delayed(fitness_function)
            (
            hypergraph=hypergraph,
            degree_dict=degree_dict,
            neighbor_dict=neighbor_dict,
            incident_hyperedge_dict=incident_hyperedge_dict,
            a=a_set,
            t=threshold,
            p_min=p_min,
            p_max=p_max,
            no_simulations=no_simulations,
            max_hop=max_hop,
            model=model,
            random_generator=random_generator)
            for a_set in tqdm(candidate_seed_sets, desc=f"Processing threads")
        )

        # read multi-process outputs
        for index, a in tqdm(enumerate(candidates), total=len(candidates), desc=f"Processing thread solutions"):
            a_set = set(a)
            influence_mean, influence_std, time = outputs[index]
            fitness[index] = inspyred.ec.emo.Pareto([(influence_mean / len(hypergraph.get_nodes())), ((max_seed_nodes+1-len(a_set))/max_seed_nodes)])
            time_gen[index] = time

    args["time"].append(time_gen)
    return fitness