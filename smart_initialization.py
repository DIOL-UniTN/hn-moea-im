from typing import Dict, Set, Tuple, List, Callable
import hypergraphx as hgx
import random

def create_initial_population(hypergraph: hgx.Hypergraph,
                              min_k: int,
                              max_k: int,
                              n: int,
                              degree_function: Callable[[hgx.Hypergraph, int], int],
                              prng: random.Random)->List[List[int]]:
    """
    Apply smart initialization of the initial population.
    - Apply node filtering, which considers only the nodes with the highest
    metric as defined by parameter degree_function.
    - Each of these nodes is added to a candidate solution with a probability
    proportional to its degree.

    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        Hypergraphx Hypergraph object which encodes the input network.
    
    min_k : int
        Minimum size of the seed set of the individuals belonging to the initial population.

    max_k : int
        Maximum size of the seed set of the individuals belonging to the initial population.
    
    n : int
        Number of individuals of the initial population.
    
    degree_function : Calleble[[hgx.Hypergraph, int], int]
        Degree or hyperdegree of input node n
    
    prng : random.Random
        Pseudo-random generator.

    Returns
    -------
        Initial population.
    """

    individuals = []

    # half of the initial population comprises seed sets of nodes chosen uniformly
    # at random from the entire node set V
    for _ in range(int(n//2)):
        # extract random number in 1,max_seed_nodes and initialize individual genome
        individual_size = random.randint(min_k, max_k)
        individuals.append(prng.sample(hypergraph.get_nodes(), individual_size))

    # select a subset of nodes characterized by high degree centrality
    nodes_filtered = filter_nodes(hypergraph, degree_function)
    all_nodes = hypergraph.get_nodes().copy()

    if len(nodes_filtered)<max_k:
        # if might very well happen that the number of filtered nodes is smaller
        # than the maximum seed set size. If this is the case we consider
        # the N nodes with the highest degree. N in this case is k+r such that
        # r is a random number between 0 and (len(hypergraph.get_nodes()-len(nodes_filtered))/2)
        all_nodes_degree = [degree_function(hypergraph, node) for node in all_nodes]
        sorted_all_nodes = [node for _, node in sorted(zip(all_nodes_degree, all_nodes), key=lambda x: x[0], reverse=True)]
        all_nodes_degree_sorted = sorted(all_nodes_degree, reverse=True)

        num_nodes_filtered = max_k+prng.randint(0, (len(hypergraph.get_nodes())-max_k)//2)
        
        nodes_filtered = sorted_all_nodes[:num_nodes_filtered]
        
        for i in range(num_nodes_filtered):
            # with probability 0.5, we replace each node currently in
            # nodes_filtered with a random node sampled from the hypergraph randomly
            if prng.random()<0.5:
                random_node = prng.choices(sorted_all_nodes[num_nodes_filtered:], weights=all_nodes_degree_sorted[num_nodes_filtered:], k=1)[0]
                nodes_filtered[i] = random_node
                all_nodes_degree_sorted.pop(sorted_all_nodes.index(random_node))
                sorted_all_nodes.remove(random_node)

    # choose n/2 individuals containing k nodes, chosen from the input hypergraph
    # with probabilities proportional to their degrees.
    sorted_nodes = sorted(nodes_filtered)
    nodes_degree = [degree_function(hypergraph, node) for node in sorted_nodes]
    for _ in range(n//2):
        nodes_ = sorted_nodes.copy()
        probs_ = nodes_degree.copy()

        new_individual = []
        new_individual_size = prng.randint(min_k, max_k)

        for _ in range(new_individual_size):
            new_node = prng.choices(nodes_, probs_)[0]
            probs_.pop(nodes_.index(new_node))
            nodes_.remove(new_node)
            new_individual.append(new_node)
        individuals.append(new_individual)

    print(f"len(individuals): {len(individuals)}")

    return individuals

def filter_nodes(hypergraph: hgx.Hypergraph, degree_function: Callable[[hgx.Hypergraph, int], int], percentage:int=30):
    # calculate the degree for each node
    node_degrees = {node: degree_function(hypergraph, node) for node in hypergraph.get_nodes()}

    # sort nodes by degree in descending order
    sorted_nodes = sorted(node_degrees.keys(), key=lambda x: node_degrees[x], reverse=True)

    # calculate the number of nodes to select (default 30% of total nodes)
    num_nodes_to_select = int((percentage/100)*len(sorted_nodes))

    # get the top nodes with the highest degrees
    output_nodes = sorted([node for node in sorted_nodes[:num_nodes_to_select]])

    return output_nodes