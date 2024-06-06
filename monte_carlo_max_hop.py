from typing import Dict, Set, Tuple, List
import random
import numpy as np
import hypergraphx as hgx

def monte_carlo_max_hop_simulation(hypergraph: hgx.Hypergraph,
                                   degree_dict:Dict[int,int],
                                   neighbor_dict:Dict[int,List[int]],
                                   incident_hyperedge_dict:Dict[int,List[Tuple[int]]],
                                   a: Set[int],
                                   t: float,
                                   p_min:float,
                                   p_max:float,
                                   no_simulations: int,
                                   max_hop:int,
                                   model: str,
                                   random_generator: random.Random):
    results = []
    times = []

    if model=="WC":
        for i in range(no_simulations):
            res, time = wc_max_hop_model(hypergraph, degree_dict, neighbor_dict, a, max_hop, random_generator)
            results.append(res)
            times.append(time)
    elif model=="LT":
        res, time = lt_max_hop_model(hypergraph, incident_hyperedge_dict, a, t, max_hop)
        results.append(res)
        times.append(time)
    elif model=="SICP":
        for i in range(no_simulations):
            res, time = sicp_max_hop_model(hypergraph, incident_hyperedge_dict, a, p_min, p_max, max_hop, random_generator)
            results.append(res)
            times.append(time)
    else:
        print(f"Invalid propagation model.")
        exit(-1)
        
    return (np.mean(results), np.std(results), sum(times))

def sicp_max_hop_model(hypergraph: hgx.Hypergraph,
                       incident_hyperedge_dict:Dict[int,List[Tuple[int]]],
                       a: Set[int],
                       p_min: float,
                       p_max: float,
                       max_hop:int,
                       random_generator: random.Random):
    """
    Susceptible-Infected (SI) model with Contact Process (CP) dynamics on
    hypergraphs as proposed in: https://arxiv.org/abs/2206.01394

    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        input hypergraph data structure on which the algorithm is executed.
    a : set[int]
        the set of initial active nodes
    p_min, p_max : float
        the system-wide min-max probability of influence on an edge, in [0,1]
    max_hop : int
        number of hops of the propagation model
    random_generator : random.Random
        already initialized pseudo-random number generator

    Returns
    -------
        tuple[int, int]
        tuple[0] is the length of the activated set of nodes at the end of the
        monte carlo exploration
    """
    I = set(a)  # set of infected nodes
    time = 0    # keep track of how much time it takes the propagation to converge to the optimal solution
    converged = False

    while (not converged) and (max_hop > 0):
        nextI = set()
        for n in I:
            # for each I-state node find all the hyperedges it belongs to
            n_incident_hyperedges = incident_hyperedge_dict[n]
            
            # then a hyperedge e is chosen uniformly at random
            e = random_generator.choice(n_incident_hyperedges)

            for m in e:
                # for each of the S-state nodes in e, it will be infected with probability p
                if m not in I:
                    prob = random_generator.random()
                    time = time+1

                    p = random_generator.uniform(p_min, p_max)
                    if prob <= p:
                        nextI.add(m)
        
        if not nextI:
            converged = True
        I = I.union(nextI)
        max_hop -= 1
    
    return len(I), time

def wc_max_hop_model(hypergraph: hgx.Hypergraph,
                     degree_dict:Dict[int,int],
                     neighbor_dict:Dict[int,List[int]],
                     a: Set[int],
                     max_hop:int,
                     random_generator: random.Random):
    """
    Weighted Cascade propagation model.

    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        input hypergraph data structure on which the algorithm is executed.
    a : set[int]
        the set of initial active nodes
    max_hop : int
        number of hops of the propagation model
    random_generator : random.Random
        already initialized pseudo-random number generator

    Returns
    -------
        tuple[int, int]
        tuple[0] is the length of the activated set of nodes at the end of the
        monte carlo exploration
    """
    A = set(a)  # set of active nodes after the propagation ended
    B = set(a)  # set of nodes activated in the last time slot
    converged = False
    time = 0    # keep track of how much time it takes the propagation to converge to the optimal solution

    while (not converged) and (max_hop > 0):
        nextB = set()
        for n in B:
            for m in neighbor_dict[n]-A:
                prob = random_generator.random()
                
                # in this propagation model, the probabilities of activation on
                # links leading to a destination node m are not uniform, but
                # inversely proportional to the degree of m
                p = 1/degree_dict[m]

                time = time+1
                if prob <= p:
                    nextB.add(m)
        B = set(nextB)
        if not B:
            converged = True
        A = A.union(B)
        max_hop -= 1
    
    return len(A), time

def lt_max_hop_model(hypergraph: hgx.Hypergraph,
                     incident_hyperedge_dict:Dict[int,List[Tuple[int]]],
                     a: Set[int],
                     t: float,
                     max_hop:int):
    """
    Linear threshold propagation model as described in https://doi.org/10.1063/5.0178329.

    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        input hypergraph data structure on which the algorithm is executed.
    a : set[int]
        the set of initial active nodes
    t : float
        threshold value in (0,1)
    max_hop : int
        number of hops of the propagation model 

    Returns
    -------
        tuple[int, int]
        tuple[0] is the length of the activated set of nodes at the end of the
        propagation process
    """
    A = set(a)  # set of active nodes after the propagation ended
    B = set(a)  # set of nodes activated in the last time slot
    C = set()   # set of active hyperedges
    converged = False
    time = 0    # keep track of how much time it takes the propagation to converge to the optimal solution

    while (not converged) and (max_hop > 0):
        nextB = set()

        for n in B:
            # iterate through all hyperedges related to the node with ID n
            for h in incident_hyperedge_dict[n]:
                # if the hyperedge is not active
                if h not in C:
                    # compute fraction of activated node in h
                    h_active_nodes = len(set(h).intersection(A))

                    time += 1

                    # if the number of active nodes in the hyperedge reaches its threshold
                    if (h_active_nodes/len(h))>=t:
                        # activate the hyperedge
                        C.add(h)

                        # iterate through all the nodes in the newly activated hyperedge
                        for j in h:
                            # if a node in the hyperedge is not active
                            if j not in A:
                                # activate the node
                                nextB.add(j)

        B = set(nextB)
        if not B:
            converged = True
        A = A.union(B)
        max_hop -= 1
    
    return len(A), time