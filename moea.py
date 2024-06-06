from typing import Dict, Set, Tuple, List
import hypergraphx as hgx
import inspyred
import random

from monte_carlo_max_hop import monte_carlo_max_hop_simulation

from ea.observer import ea_observer, time_observer, hypervolume_observer
from ea.terminator import generation_termination
from ea.generator import ea_generator
from ea.evaluator import ea_evaluator
from ea.crossover import ea_crossover
from ea.mutation import ea_mutation, ea_global_random_mutation
from ea.archiver import ea_archiver

def moea_influence_maximization(hypergraph: hgx.Hypergraph,
                                degree_dict:Dict[int,int],
                                hyperdegree_dict:Dict[int,int],
                                neighbor_dict:Dict[int,List[int]],
                                incident_hyperedge_dict:Dict[int,List[Tuple[int]]],
                                random_gen: random.Random,
                                min_seed_nodes: int,
                                max_seed_nodes: float,
                                population_size: int,
                                offspring_size: int,
                                initial_population: List[List[int]],
                                max_generations: int,
                                tournament_size: int,
                                mutation_rate : float,
                                crossover_rate : float,
                                num_elites : int,
                                p_min : float,
                                p_max : float,
                                threshold: float,
                                max_hop: int,
                                model : str,
                                no_simulations : int,
                                n_threads : int,
                                custom_mutation : bool,
                                output_activation_attempts_file_path : str,
                                output_hypervolume_file_path : str):
    """
    
    Multi-objective evolutionary influence maximization.

    """
    # initialize multi-objective evolutionary algorithm NSGA-II
    max_seed_set_size = int(max_seed_nodes * len(hypergraph.get_nodes()))
    print(f"max_seed_set_size: {max_seed_set_size}")

    fitness_function = monte_carlo_max_hop_simulation                           # the influence is propagated up to a maximum number of hops

    ea = inspyred.ec.emo.NSGA2(random_gen)
    ea.archiver = ea_archiver                                                   # archiver with Pareto preference (Pareto archive)
    if custom_mutation:
        ea.variator = [ea_crossover, ea_mutation]                               # the list of variation operators
    else:
        ea.variator = [ea_crossover, ea_global_random_mutation]                 # the list of variation operators
    ea.observer = [hypervolume_observer]                                        # the (possibly list of) observer(s)
    ea.terminator = generation_termination                                      # the (possibly list of) terminator(s)

    # start the evolutionary process
    final_pop = ea.evolve(
        generator = ea_generator,                                               # the function to be used to generate candidate solutions # TODO riflettere su initial population, vedi anche argument seeds sotto
        evaluator = ea_evaluator,                                               # the function to be used to evaluate candidate solutions
        bounder = inspyred.ec.DiscreteBounder(hypergraph.get_nodes()),          # a function used to bound candidate solutions
        maximize = True,                                                        # boolean value stating use of maximization
        seeds = initial_population,                                             # individuals (seed sets) to be added to the initial population (the rest will be randomly generated) # TODO riflettere su initial population, vedi anche argument generator sopra
        pop_size = population_size,                                             # the number of Individuals in the population 
        num_selected = offspring_size,                                          # offspring of the EA
        generations_budget = max_generations,                                   # maximum generations
        tournament_size = tournament_size,                                      # EA tournament size
        mutation_rate = mutation_rate,                                          # the rate at which mutation is performed
        crossover_rate = crossover_rate,                                        # the rate at which crossover is performed
        num_elites = num_elites,                                                # number of elites to consider
        hypergraph = hypergraph,                                                # input hypergraph network
        degree_dict = degree_dict,                                              # degree_dict[i] = degree node i
        hyperdegree_dict = hyperdegree_dict,                                    # hyperdegree_dict[i] = hyperdegree node i
        neighbor_dict = neighbor_dict,                                          # neighbor_dict[i] = list of neighbors of node i
        incident_hyperedge_dict = incident_hyperedge_dict,                      # incident_hyperedge_dict[i] = list of incident hyperedges of node i
        p_min = p_min,                                                          # probability MIN for SICP propagation model
        p_max = p_max,                                                          # probability MAX for SICP propagation model
        threshold = threshold,                                                  # threshold for LT propagation model
        max_hop = max_hop,                                                      # maximum number of influence propagation time steps for SICP propagation model
        propagation_model = model,                                              # type of influence propagation model
        no_simulations = no_simulations,                                        # number of simulations for spread calculation
        nodes = hypergraph.get_nodes(),                                         # hypergraph nodes
        min_seed_nodes = min_seed_nodes,                                        # minimum number of nodes in a seed set
        max_seed_nodes = max_seed_set_size,                                     # maximum number of nodes in a seed set
        fitness_function = fitness_function,                                    # fitness_function
        random_generator = random_gen,                                          # already initialized pseudo-random number generation
        time = [],                                                              # keep track of Time (Activation Attempts) trend throughout the generations
        hypervolume = [],                                                       # keep track of HV trend throughout the generations
        n_threads = n_threads,                                                  # number of threads to handle parallel computation
        activation_attempts_file_path = output_activation_attempts_file_path,   # file path where to store the number of activation attempts
        hypervolume_file_path = output_hypervolume_file_path                    # file path where to store the hypervolume of the final population
    )

    # extract seed sets from the final Pareto front
    print(f"final_pop: {len(final_pop)}")
    print(f"ea_archive: {len(ea.archive)}")

    pareto_front = [[individual.candidate, individual.fitness[0]*100, ((len(individual.candidate)  / len(hypergraph.get_nodes())) * 100)] for individual in ea.archive] 
    final_pop = [[individual.candidate, individual.fitness[0]*100, ((len(individual.candidate)  / len(hypergraph.get_nodes())) * 100)] for individual in final_pop] 

    return pareto_front, final_pop
