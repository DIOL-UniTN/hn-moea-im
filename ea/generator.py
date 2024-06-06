import inspyred
@inspyred.ec.generators.diversify # decorator that makes it impossible to generate copies
def ea_generator(random, args):
    min_seed_nodes = args["min_seed_nodes"] # min seed set size
    max_seed_nodes = args["max_seed_nodes"] # max seed set size
    nodes = args["nodes"]                   # node set

    # extract random number in 1,max_seed_nodes and initialize individual genome
    individual_size = random.randint(min_seed_nodes, max_seed_nodes)
    individual = [0] * individual_size
    
    for i in range(0, individual_size):
        individual[i] = nodes[random.randint(0,len(nodes)-1)]

    return list(set(individual))