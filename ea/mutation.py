import inspyred
import numpy as np
import random

@inspyred.ec.variators.mutator
def ea_mutation(rng, candidate, args):
    """
    Custom mutation operator
    """
    # if candidate length is equal to one then perform insertion mutation
    if len(candidate) == 1:
        return ea_insertion_mutation(rng, [candidate], args)[0]

    # stochastic or hypergraph-aware
    randomChoice = rng.choices([i for i in range(4)], k=1)[0]

    if randomChoice==0:
        # global random mutation
        return ea_global_random_mutation(rng, [candidate], args)[0]
    if randomChoice==1:
        # random insertion mutation
        return ea_insertion_mutation(rng, [candidate], args)[0]
    if randomChoice==2:
        # random removal mutation
        return ea_removal_mutation(rng, [candidate], args)[0]
    elif randomChoice==3:
        # hypergraph-aware mutation operator
        gene_selection_strategy = 5
        node_selection_strategy = rng.choices([2,9], k=1)[0]
        return ea_hypergraph_aware_mutation(rng, candidate, gene_selection=gene_selection_strategy, node_selection=node_selection_strategy, args=args)
    else:
        print("\n\nERROR MUTATION CHOICE\n\n")
        return -1

# === STOCHASTIC OPERATORS =====================================================
@inspyred.ec.variators.mutator
def ea_global_random_mutation(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one random node of the hypergraph.
    """
    if len(candidate)>1:
        # remove nodes in the candidate solution from the pool of nodes the mutation algorithm samples from
        nodes = args["nodes"].copy()
        for c in candidate:
            if c in nodes: nodes.remove(c)

        mutated_candidate = candidate.copy()

        # choose the gene to mutate
        new_gene = nodes[rng.randint(0, len(nodes) - 1)]

        # mutate
        mutation_idx = rng.randint(0, len(mutated_candidate) - 1)
        mutated_candidate[mutation_idx] = new_gene
        
        return mutated_candidate
    else:
        return candidate
@inspyred.ec.variators.mutator
def ea_insertion_mutation(rng, candidate, args):
    """
    Randomly add a node to the individual.
    """
    max_seed_nodes = args["max_seed_nodes"]
    mutated_candidate = candidate.copy()

    # if the length of the candidate is already the maximum then perform global mutation
    if len(mutated_candidate) >= max_seed_nodes:
        return ea_global_random_mutation(rng, [candidate], args)[0]
    
    # remove nodes which are in the candidate solution from the pool of nodes
    # the mutation algorithm samples from
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    mutated_node = nodes[rng.randint(0, len(nodes)-1)]
    mutated_candidate.append(mutated_node)

    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_removal_mutation(rng, candidate, args):
    """
    Randomly remove a node of the individual.
    """
    min_seed_nodes = args["min_seed_nodes"]
    mutated_candidate = candidate.copy()

    # if the length of the candidate is smaller then or equal to the minimum
    # seed set size then perform global mutation
    if len(mutated_candidate) <= min_seed_nodes:
        return ea_global_random_mutation(rng, [candidate], args)[0]
    
    gene = rng.randint(0, len(mutated_candidate)-1)
    mutated_candidate.pop(gene)

    return mutated_candidate
# === HYPERGRAPH_AWARE OPERATORS ===============================================
def ea_hypergraph_aware_mutation(rng: random.Random,
                                 candidate: list[int],
                                 gene_selection: int,
                                 node_selection: int,
                                 args):
    """
    Hypergraph-aware mutation.

    Parameters
    ----------
        rng: random.Random
        random number generator

        candidate: list[int]
        individual to be mutated

        gene_selection: int
        with this parameter the algorithm decides how to choose the gene to be
        mutated according to the following mapping
            0: gene is selected uniformly at random among the nodes which populate the individual's seed set
            1: gene is selected among the nodes which populate the individual's seed set with probability proportional to its hyperdegree
            2: gene is selected among the nodes which populate the individual's seed set with probability proportional to its number of neighbors
            3: gene is selected among the nodes which populate the individual's seed set with probability proportional to the average order of the hyperedges it belongs to
            4: gene is selected among the nodes which populate the individual's seed set with probability inversely proportional to its hyperdegree
            5: gene is selected among the nodes which populate the individual's seed set with probability inversely proportional to its number of neighbors
            6: gene is selected among the nodes which populate the individual's seed set with probability inversely proportional to the average order of the hyperedges it belongs to

        node_selection: int
        with this parameter the algorithm decides how to choose the node which
        is going to replace the mutated gene, according to the following mapping
            0:  the new node is selected randomly among the neighbors of the gene to be mutated
            1:  the new node is selected among the neighbors of the gene to be mutated with probability proportional to the hyperdegree of the neighbor
            2:  the new node is selected among the neighbors of the gene to be mutated with probability proportional to the number of neighbors of the neighbor
            3:  the new node is selected among the neighbors of the gene to be mutated with probability proportional to the average order of the hyperedges in which the neighbor partecipate
            4:  the new node is selected among the neighbors of the gene to be mutated with probability inversely proportional to the hyperdegree of the neighbor
            5:  the new node is selected among the neighbors of the gene to be mutated with probability inversely proportional to the number of neighbors of the neighbor
            6:  the new node is selected among the neighbors of the gene to be mutated with probability inversely proportional to the average order of the hyperedges in which the neighbor partecipate
            7:  the new node is selected randomly among the nodes of the hypergraph which are not part of the candidate seed set
            8:  the new node is selected among the hypergraph nodeset with probability proportional to its hyperdegree
            9:  the new node is selected among the hypergraph nodeset with probability proportional to its number of neighbors
            10: the new node is selected among the hypergraph nodeset with probability proportional to average order of its incident hyperedges
            11: the new node is selected among the hypergraph nodeset with probability inversely proportional to its hyperdegree
            12: the new node is selected among the hypergraph nodeset with probability inversely proportional to its number of neighbors
            13: the new node is selected among the hypergraph nodeset with probability inversely proportional to average order of its incident hyperedges

        args
        evolutionary algorithm parameters

    Returns
    -------
        mutated candidate : list[int]
    """
    mutated_candidate = candidate.copy()

    degree_dict = args["degree_dict"]
    hyperdegree_dict = args["hyperdegree_dict"]
    neighbor_dict = args["neighbor_dict"]
    incident_hyperedge_dict = args["incident_hyperedge_dict"]

    # choose the gene to mutate
    if gene_selection==0:
        # gene is selected uniformly at random among the nodes which populate the individual's seed set
        mutation_idx = rng.randint(0, len(mutated_candidate)-1)
    elif gene_selection==1:
        # gene is selected among the nodes which populate the individual's seed set with probability proportional to its hyperdegree
        probs = []
        for node in mutated_candidate:
            probs.append(hyperdegree_dict[node])
        probs = np.array(probs)/max(probs)
        mutation_idx = rng.choices(range(len(mutated_candidate)), probs)[0]
    elif gene_selection==2:
        # gene is selected among the nodes which populate the individual's seed set with probability proportional to its number of neighbors
        probs = []
        for node in mutated_candidate:
            probs.append(degree_dict[node])
        probs = np.array(probs)/max(probs)
        mutation_idx = rng.choices(range(len(mutated_candidate)), probs)[0]
    elif gene_selection==3:
        # gene is selected among the nodes which populate the individual's seed set with probability proportional to the average order of the hyperedges it belongs to
        probs = []
        for node in mutated_candidate:
            node_incident_edges = incident_hyperedge_dict[node]
            probs.append(sum([len(e) for e in node_incident_edges])/len(node_incident_edges))
        probs = np.array(probs)/max(probs)
        mutation_idx = rng.choices(range(len(mutated_candidate)), probs)[0]
    elif gene_selection==4:
        # gene is selected among the nodes which populate the individual's seed set with probability inversely proportional to its hyperdegree
        probs = []
        for node in mutated_candidate:
            probs.append(1/hyperdegree_dict[node])
        probs = np.array(probs)/max(probs)
        mutation_idx = rng.choices(range(len(mutated_candidate)), probs)[0]
    elif gene_selection==5:
        # gene is selected among the nodes which populate the individual's seed set with probability inversely proportional to its number of neighbors
        probs = []
        for node in mutated_candidate:
            probs.append(1/degree_dict[node])
        probs = np.array(probs)/max(probs)
        mutation_idx = rng.choices(range(len(mutated_candidate)), probs)[0]
    elif gene_selection==6:
        # gene is selected among the nodes which populate the individual's seed set with probability inversely proportional to the average order of the hyperedges it belongs to
        probs = []
        for node in mutated_candidate:
            node_incident_edges = incident_hyperedge_dict[node]
            probs.append(1/(sum([len(e) for e in node_incident_edges])/len(node_incident_edges)))
        probs = np.array(probs)/max(probs)
        mutation_idx = rng.choices(range(len(mutated_candidate)), probs)[0]

    # update hypergraph node set removing the nodes which populate the focal candidate solution
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    # update set of neighbors of the selected gene removing the nodes which populate the focal candidate solution
    neighbors = list(neighbor_dict[mutated_candidate[mutation_idx]])
    for c in candidate:
        if c in neighbors: neighbors.remove(c)
    
    # choose the node which is going to replace the mutated gene
    if node_selection==0:
        # the new node is selected randomly among the neighbors of the gene to be mutated
        if len(neighbors) > 0:
            mutated_node = neighbors[rng.randint(0, len(neighbors)-1)]
            mutated_candidate[mutation_idx] = mutated_node
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==1:
        # the new node is selected among the neighbors of the gene to be mutated with probability proportional to the hyperdegree of the neighbor
        if len(neighbors)>0:
            # calculate the hyperdegree of the neighbors of the node represented by the gene to mutate
            probs = []
            for node in neighbors:
                probs.append(hyperdegree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(neighbors)), probs)[0]
            mutated_candidate[mutation_idx] = neighbors[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==2:
        # the new node is selected among the neighbors of the gene to be mutated with probability proportional to the number of neighbors of the neighbor
        if len(neighbors)>0:
            # calculate the number of neighbors of the neighbors of the node represented by the gene to mutate
            probs = []
            for node in neighbors:
                probs.append(degree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(neighbors)), probs)[0]
            mutated_candidate[mutation_idx] = neighbors[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==3:
        # the new node is selected among the neighbors of the gene to be mutated with probability proportional to the average order of the hyperedges in which the neighbor partecipate
        if len(neighbors)>0:
            # calculate the average order of the hyperedges where the neighbor nodes partecipate
            probs = []
            for node in neighbors:
                node_incident_edges = incident_hyperedge_dict[node]
                probs.append(sum([len(e) for e in node_incident_edges])/len(node_incident_edges))
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(neighbors)), probs)[0]
            mutated_candidate[mutation_idx] = neighbors[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==4:
        # the new node is selected among the neighbors of the gene to be mutated with probability inversely proportional to the hyperdegree of the neighbor
        if len(neighbors)>0:
            # calculate the hyperdegree of the neighbors of the node represented by the gene to mutate
            probs = []
            for node in neighbors:
                probs.append(1/degree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(neighbors)), probs)[0]
            mutated_candidate[mutation_idx] = neighbors[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==5:
        # the new node is selected among the neighbors of the gene to be mutated with probability inversely proportional to the number of neighbors of the neighbor
        if len(neighbors)>0:
            # calculate the number of neighbors of the neighbors of the node represented by the gene to mutate
            probs = []
            for node in neighbors:
                probs.append(1/degree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(neighbors)), probs)[0]
            mutated_candidate[mutation_idx] = neighbors[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==6:
        # the new node is selected among the neighbors of the gene to be mutated with probability inversely proportional to the average order of the hyperedges in which the neighbor partecipate
        if len(neighbors)>0:
            # calculate the average order of the hyperedges where the neighbor nodes partecipate
            probs = []
            for node in neighbors:
                node_incident_edges = incident_hyperedge_dict[node]
                probs.append(1/(sum([len(e) for e in node_incident_edges])/len(node_incident_edges)))
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(neighbors)), probs)[0]
            mutated_candidate[mutation_idx] = neighbors[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==7:
        # the new node is selected among the hypegraph nodeset uniformly at random
        if len(nodes) > 0:
            mutated_node = nodes[rng.randint(0, len(nodes)-1)]
            mutated_candidate[mutation_idx] = mutated_node
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==8:
        # the new node is selected among the hypergraph nodeset with probability proportional to its hyperdegree
        if len(nodes)>0:
            # calculate the hyperdegree of the nodes
            probs = []
            for node in nodes:
                probs.append(hyperdegree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(nodes)), probs)[0]
            mutated_candidate[mutation_idx] = nodes[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==9:
        # the new node is selected among the hypergraph nodeset with probability proportional to its number of neighbors
        if len(nodes)>0:
            # calculate the number of neighbors of the nodes
            probs = []
            for node in nodes:
                probs.append(degree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(nodes)), probs)[0]
            mutated_candidate[mutation_idx] = nodes[idx]
        else:
            # if we don't have nodes to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==10:
        # the new node is selected among the hypergraph nodeset with probability proportional to average order of its incident hyperedges
        if len(nodes)>0:
            # calculate the average order of the hyperedges where the node partecipate
            probs = []
            for node in nodes:
                node_incident_edges = incident_hyperedge_dict[node]
                probs.append(sum([len(e) for e in node_incident_edges])/len(node_incident_edges))
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(nodes)), probs)[0]
            mutated_candidate[mutation_idx] = nodes[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==11:
        # the new node is selected among the hypergraph nodeset with probability inversely proportional to its hyperdegree
        if len(nodes)>0:
            # calculate the hyperdegree of the nodes
            probs = []
            for node in nodes:
                probs.append(1/hyperdegree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(nodes)), probs)[0]
            mutated_candidate[mutation_idx] = nodes[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==12:
        # the new node is selected among the hypergraph nodeset with probability inversely proportional to its number of neighbors
        if len(nodes)>0:
            # calculate the number of neighbors of the nodes
            probs = []
            for node in nodes:
                probs.append(1/degree_dict[node])
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(nodes)), probs)[0]
            mutated_candidate[mutation_idx] = nodes[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    elif node_selection==13:
        # the new node is selected among the hypergraph nodeset with probability inversely proportional to average order of its incident hyperedges
        if len(nodes)>0:
            # calculate the average order of the hyperedges where the node partecipate
            probs = []
            for node in nodes:
                node_incident_edges = incident_hyperedge_dict[node]
                probs.append(1/(sum([len(e) for e in node_incident_edges])/len(node_incident_edges)))
            probs = np.array(probs)/max(probs)
            idx = rng.choices(range(len(nodes)), probs)[0]
            mutated_candidate[mutation_idx] = nodes[idx]
        else:
            # if we don't have neighbors to choose from, global mutation
            mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    return mutated_candidate

@inspyred.ec.variators.mutator
def ea_local_neighbors_hyper_degree_mutation(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of it's neighbors,
    but with probability directly proportional to its degree.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the neighbors of the selected node
    nodes = list(args["hypergraph"].get_neighbors(mutated_candidate[mutation_idx]))
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the degree of the neighbors of the node represented
        # by the gene to mutate
        neighbor_degrees = []
        for node in nodes:
            neighbor_degrees.append(args["hypergraph"].degree(node))
        probs = np.array(neighbor_degrees)/max(neighbor_degrees)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have neighbors to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_local_neighbors_degree_mutation(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of it's neighbors,
    but with probability directly proportional its number of neighbors.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the neighbors of the selected node
    nodes = list(args["hypergraph"].get_neighbors(mutated_candidate[mutation_idx]))
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the number of neighbors of the neighbors of the node represented
        # by the gene to mutate
        neighbor_degrees = []
        for node in nodes:
            neighbor_degrees.append(len(args["hypergraph"].get_neighbors(node)))
        probs = np.array(neighbor_degrees)/max(neighbor_degrees)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have neighbors to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_local_neighbors_hyperedge_order_mutation(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of it's neighbors,
    but with probability directly proportional to the average order of the
    interactions in which it partecipates.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the neighbors of the selected node
    nodes = list(args["hypergraph"].get_neighbors(mutated_candidate[mutation_idx]))
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the average order of the hyperedges where the neighbor nodes partecipate
        neighbor_avg_hyperedge_order = []
        for node in nodes:
            node_incident_edges = args["hypergraph"].get_incident_edges(node)
            neighbor_avg_hyperedge_order.append(sum([len(e) for e in node_incident_edges])/len(node_incident_edges))
        probs = np.array(neighbor_avg_hyperedge_order)/max(neighbor_avg_hyperedge_order)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have neighbors to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_local_neighbors_hyper_degree_mutation_inverse(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of it's neighbors,
    but with probability inversely proportional to its degree.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the neighbors of the selected node
    nodes = list(args["hypergraph"].get_neighbors(mutated_candidate[mutation_idx]))
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the degree of the neighbors of the node represented
        # by the gene to mutate
        neighbor_degrees = []
        for node in nodes:
            neighbor_degrees.append(1/args["hypergraph"].degree(node))
        probs = np.array(neighbor_degrees)/max(neighbor_degrees)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have neighbors to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_local_neighbors_degree_mutation_inverse(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of it's neighbors,
    but with probability directly proportional its number of neighbors.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the neighbors of the selected node
    nodes = list(args["hypergraph"].get_neighbors(mutated_candidate[mutation_idx]))
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the number of neighbors of the neighbors of the node represented
        # by the gene to mutate
        neighbor_degrees = []
        for node in nodes:
            neighbor_degrees.append(1/len(args["hypergraph"].get_neighbors(node)))
        probs = np.array(neighbor_degrees)/max(neighbor_degrees)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have neighbors to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_local_neighbors_hyperedge_order_mutation_inverse(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of it's neighbors,
    but with probability inversely proportional to the average order of the
    interactions in which it partecipates.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the neighbors of the selected node
    nodes = list(args["hypergraph"].get_neighbors(mutated_candidate[mutation_idx]))
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the average order of the hyperedges where the neighbor nodes partecipate
        neighbor_avg_hyperedge_order = []
        for node in nodes:
            node_incident_edges = args["hypergraph"].get_incident_edges(node)
            neighbor_avg_hyperedge_order.append(1/(sum([len(e) for e in node_incident_edges])/len(node_incident_edges)))
        probs = np.array(neighbor_avg_hyperedge_order)/max(neighbor_avg_hyperedge_order)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have neighbors to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_global_hyper_degree_mutation(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one node of the hypergraph,
    but with probability directly proportional to its degree.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the nodes of the hypergraph which are not in the
    # candidate solution of the selected node
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the degree of the hypergraph node
        node_degrees = []
        for node in nodes:
            node_degrees.append(args["hypergraph"].degree(node))
        probs = np.array(node_degrees)/max(node_degrees)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have nodes to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_global_degree_mutation(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one node of the hypergraph,
    but with probability directly proportional to its number of neighbors.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the nodes of the hypergraph which are not in the
    # candidate solution of the selected node
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the number of neighbors of the hypergraph nodes
        node_num_neighbors = []
        for node in nodes:
            node_num_neighbors.append(len(args["hypergraph"].get_neighbors(node)))
        probs = np.array(node_num_neighbors)/max(node_num_neighbors)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have nodes to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_global_hyperedge_order_mutation(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of the hypergraph nodes,
    but with probability directly proportional to the average order of the
    interactions in which it partecipates.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the hypergraph nodes which are not in the candidate solution
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the average order of the hyperedges where the nodes partecipate
        node_avg_hyperedge_order = []
        for node in nodes:
            node_incident_edges = args["hypergraph"].get_incident_edges(node)
            node_avg_hyperedge_order.append(sum([len(e) for e in node_incident_edges])/len(node_incident_edges))
        probs = np.array(node_avg_hyperedge_order)/max(node_avg_hyperedge_order)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have nodes to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_global_hyper_degree_mutation_inverse(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one node of the hypergraph,
    but with probability inversely proportional to its degree.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the nodes of the hypergraph which are not in the
    # candidate solution of the selected node
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the degree of the hypergraph node
        node_degrees = []
        for node in nodes:
            node_degrees.append(1/args["hypergraph"].degree(node))
        probs = np.array(node_degrees)/max(node_degrees)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have nodes to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_global_degree_mutation_inverse(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one node of the hypergraph,
    but with probability inversely proportional its number of neighbors.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the nodes of the hypergraph which are not in the
    # candidate solution of the selected node
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the number of neighbors of the hypergraph nodes
        node_num_neighbors = []
        for node in nodes:
            node_num_neighbors.append(1/len(args["hypergraph"].get_neighbors(node)))
        probs = np.array(node_num_neighbors)/max(node_num_neighbors)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have nodes to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate
@inspyred.ec.variators.mutator
def ea_global_hyperedge_order_mutation_inverse(rng, candidate, args):
    """
    Randomly mutates one gene of the individual with one of the hypergraph nodes,
    but with probability inversely proportional to the average order of the
    interactions in which it partecipates.
    """
    mutated_candidate = candidate.copy()

    # choose the gene to mutate
    mutation_idx = rng.randint(0, len(mutated_candidate)-1)

    # choose among the hypergraph nodes which are not in the candidate solution
    nodes = args["nodes"].copy()
    for c in candidate:
        if c in nodes: nodes.remove(c)
    
    if len(nodes)>0:
        # calculate the average order of the hyperedges where the nodes partecipate
        node_avg_hyperedge_order = []
        for node in nodes:
            node_incident_edges = args["hypergraph"].get_incident_edges(node)
            node_avg_hyperedge_order.append(1/(sum([len(e) for e in node_incident_edges])/len(node_incident_edges)))
        probs = np.array(node_avg_hyperedge_order)/max(node_avg_hyperedge_order)
        idx = rng.choices(range(0, len(nodes)), probs)[0]
        mutated_candidate[mutation_idx] = nodes[idx]
    else:
        # if we don't have nodes to choose from, global mutation
        mutated_candidate = ea_global_random_mutation(rng, [candidate], args)[0]
    
    return mutated_candidate