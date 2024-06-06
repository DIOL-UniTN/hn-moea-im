from typing import Dict, List, Set, Tuple
import os
import argparse
import random
import hypergraphx as hgx
import json
import time
from datetime import datetime

from hypergraphx.representations.projections import clique_projection
from loaders import load_hypergraph
from smart_initialization import create_initial_population
from moea import moea_influence_maximization

import collections
collections.Mapping = collections.abc.Mapping
collections.Sequence = collections.abc.Sequence
collections.Iterable = collections.abc.Iterable

def read_arguments():
    parser = argparse.ArgumentParser(description="Influence Maximization on Hypergraph Networks")
    
    parser.add_argument("--min_seed_nodes", type=int, default=1, help="Minimum number of nodes in a seed set.")
    #parser.add_argument("--max_seed_nodes", type=float, default=0.1, help="Maximum number of nodes in a seed set as percentage of the whole network.")

    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument('--no_runs', type=int, default=1, help='EA number of runs.')
    parser.add_argument('--n_threads', type=int, default=1, help="Number of threads to handle parallel computation.")

    parser.add_argument("--hypergraph_path", type=str, default="dataset/small/small.json", help="File path of the JSON file encoding the input hypergraph network. (IF summary_input is True THEN this is the file path of the JSON file encoding the input summary)")
    parser.add_argument('--output_file_name', type=str, default="moea.json", help='JSON file name where to store the individuals of the final pareto front at the end of the execution.')
    parser.add_argument('--output_execution_time_file_name', type=str, default="moea_exec_time.txt", help='File name of the txt file where to store the execution time.')
    parser.add_argument('--output_activation_attempts_file_name', type=str, default="moea_activation_attempts.csv", help='File name of the csv file where to store the number of activation attempts.')
    parser.add_argument('--output_hypervolume_file_name', type=str, default="moea_hypervolume.csv", help='File name of the csv file where to store hypervolume for each generation.')
    parser.add_argument('--out_dir', default='output/', type=str, help='Location of the output directory.')

    parser.add_argument('--population_size', type=int, default=100, help='EA population size.')
    parser.add_argument('--offspring_size', type=int, default=100, help='EA offspring size.')
    parser.add_argument('--max_generations', type=int, default=100, help='Generational budget.')
    parser.add_argument('--tournament_size', type=int, default=5, help='EA tournament size.')
    parser.add_argument('--mutation_rate', type=float, default=0.1, help='EA mutation rate.')
    parser.add_argument('--crossover_rate', type=float, default=1.0, help='EA crossover rate.')
    parser.add_argument('--num_elites', type=int, default=2, help='EA number of elite individuals.')

    parser.add_argument('--threshold', type=float, default=0.8, help='Threshold for LT propagation model.')
    parser.add_argument('--p_min', type=float, default=0.005, help='Probability MIN for SICP propagation model.')
    parser.add_argument('--p_max', type=float, default=0.02, help='Probability MAX for SICP propagation model.')
    parser.add_argument('--max_hop', type=int, default=5, help='Number of max hops for the Monte Carlo max hop function.')
    parser.add_argument('--model', default="WC", choices=['WC', 'LT', 'SICP'], help='Influence propagation model.')
    parser.add_argument('--no_simulations', type=int, default=100, help='Number of simulations for spread calculation.')

    parser.add_argument('--custom_mutation', type=bool, default=False, help='Flag to decide to apply custom mutation operators or not.')

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])

    # create directory for saving results
    current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_folder_path = f"{args['out_dir']}/{current_datetime}"
    os.makedirs(output_folder_path)

    # load hypergraph
    inputHypergraph = load_hypergraph(args["hypergraph_path"])

    print(inputHypergraph)
    
    # calculate k based on network size
    #init_seed_set_size = int(len(inputHypergraph.get_nodes())*args["max_seed_nodes"])
    init_seed_set_size = 100
    print(f"init_seed_set_size: {init_seed_set_size}")
    
    # degree, hyperdegree, neighbor list, incident hyperedge list pre-computation
    # note: in order to significantly reduce the execution time, we store the
    # degree of the nodes, the hyperdegree of the nodes and the list of neighbors
    # of each node in dictionary data structures
    degree_dict:Dict[int,int] = dict()
    hyperdegree_dict:Dict[int,int] = dict()
    neighbor_dict:Dict[int,List[int]] = dict()
    incident_hyperedge_dict:Dict[int,List[Tuple[int]]] = dict()
    for n in inputHypergraph.get_nodes():
        degree_dict[n] = len(inputHypergraph.get_neighbors(n))
        hyperdegree_dict[n] = inputHypergraph.degree(n)
        neighbor_dict[n] = inputHypergraph.get_neighbors(n)
        incident_hyperedge_dict[n] = inputHypergraph.get_incident_edges(n)

    for r in range(args["no_runs"]):
        # create directory for saving results of the run
        output_folder_run_path = output_folder_path+"/"+str(r+1)
        os.makedirs(output_folder_run_path)

        start_time = time.time()

        # smart initialization
        degree_function = lambda inputHypergraph, n: len(inputHypergraph.get_neighbors(n))

        initial_population = create_initial_population(hypergraph=inputHypergraph,
                                                       min_k=args["min_seed_nodes"],
                                                       max_k=init_seed_set_size,
                                                       n=args["population_size"],
                                                       degree_function=degree_function,
                                                       prng=rng)
        #print(f"initial_population: {initial_population}")
        print(f"len(initial_population): {len(initial_population)}")

        # run multi-objective evolutionary algorithm optimization
        pareto_front, final_pop = moea_influence_maximization(
                                            hypergraph=inputHypergraph,
                                            degree_dict=degree_dict,
                                            hyperdegree_dict=hyperdegree_dict,
                                            neighbor_dict=neighbor_dict,
                                            incident_hyperedge_dict=incident_hyperedge_dict,
                                            random_gen=rng,
                                            min_seed_nodes=args["min_seed_nodes"],
                                            #max_seed_nodes=args["max_seed_nodes"],
                                            max_seed_nodes=100/inputHypergraph.num_nodes(),
                                            population_size=args["population_size"],
                                            offspring_size=args["offspring_size"],
                                            initial_population=initial_population,
                                            max_generations=args["max_generations"],
                                            tournament_size=args["tournament_size"],
                                            mutation_rate=args["mutation_rate"],
                                            crossover_rate=args["crossover_rate"],
                                            num_elites=args["num_elites"],
                                            p_min=args["p_min"],
                                            p_max=args["p_max"],
                                            threshold=args["threshold"],
                                            max_hop=args["max_hop"],
                                            model=args["model"],
                                            no_simulations=args["no_simulations"],
                                            n_threads=args["n_threads"],
                                            custom_mutation=args["custom_mutation"],
                                            output_activation_attempts_file_path=f"{output_folder_run_path}/{args['output_activation_attempts_file_name']}",
                                            output_hypervolume_file_path=f"{output_folder_run_path}/{args['output_hypervolume_file_name']}")
        execution_time = (time.time() - start_time)
        print(f"\noutput seed set: {pareto_front}")
        print(f"\noutput seed set len: {len(pareto_front)}")
        print(f"\noutput final_population: {final_pop}")
        print(f"\noutput final_population len: {len(final_pop)}")

        # save hypergraph pareto front
        json_object = json.dumps([index[0] for index in pareto_front], indent=1)
        output_file = open(f"{output_folder_run_path}/{args['output_file_name']}", "w")
        output_file.write(json_object)
        output_file.close()
        
        # save execution time
        execution_time_file = open(f"{output_folder_run_path}/{args['output_execution_time_file_name']}", 'w')
        execution_time_file.write(str(execution_time))
        execution_time_file.close()

        print(f"\n---run {r+1}/{args['no_runs']} execution_time={str(execution_time)}\n")
