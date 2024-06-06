import random
import argparse
import time
import json
import hypergraphx as hgx
from loaders import load_hypergraph

def high_degree_discount(hypergraph: hgx.Hypergraph, k: int):
    """
    Execute High Degree Discount algoritm as proposed in:
    https://arxiv.org/abs/2206.01394

    Parameters
    ----------
    hypergraph : hgx.Hypergraph
        Hypergraphx Hypergraph object which encodes the input network.
    
    k : int
        Cardinality of the seed set.

    Returns
    -------
        Seed set of k nodes selected by HDD optimization algorithm.
    """
    seeds = set()

    # compute deg_0
    degree = {} # dict[int, int]
                # key: node id
                # value: degree as the number of neighbors
    for n in hypergraph.get_nodes():
        degree[n] = len(hypergraph.get_neighbors(n))

    for i in range(k):
        # sort nodes according to their adaptive degree
        sorted_nodes = sorted(degree.keys(), key=lambda x: degree[x], reverse=True)

        # select the node with the largest adaptive degree which is not in the
        # seed set yet and add it to the seed set
        for node in sorted_nodes:
            if node not in seeds:
                chosenNode = node
                break
        seeds.add(chosenNode)

        # update adaptive degree
        for v_q in hypergraph.get_neighbors(chosenNode):
            z = 0
            for v_q_neighbor in hypergraph.get_neighbors(v_q):
                if v_q_neighbor in seeds:
                    z += 1
            degree[v_q] = degree[v_q] - z

    return list(seeds)


def read_arguments():
    parser = argparse.ArgumentParser(description="Influence Maximization on Hypergraph Networks")
    
    parser.add_argument("--min_seed_nodes", type=int, default=1, help="Minimum number of nodes in a seed set.")
    parser.add_argument("--max_seed_nodes", type=float, default=100, help="Maximum number of nodes in a seed set.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--hypergraph_path", type=str, default="dataset/small/small.json", help="File path of the JSON file encoding the input hypergraph network. (IF summary_input is True THEN this is the file path of the JSON file encoding the input summary)")
    parser.add_argument('--k_step', type=int, default=1, help='The algorithm executes the algorithm for all several values of k (seed set size) within the interval [min-k, max-k]. With this parameter we specify how divide this interval. For example, if k=2 and min-k=1 and max_k=5 then we are going to execute the algorithm for k=1, k=3, k=5.')

    parser.add_argument("--output_file_path", type=str, default="output/hdd_solution/hdd.json", help="File path of the output JSON file.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])

    # load hypergraph
    inputHypergraph = load_hypergraph(args["hypergraph_path"])
    
    # calculate max seed set size based on network size
    max_seed_set_size = int(args["max_seed_nodes"])
    print(f"max_seed_set_size: {max_seed_set_size}")

    start_time = time.time()

    output_seed_sets = list()
    for k in range(args["min_seed_nodes"], max_seed_set_size+1, args["k_step"]):
        print(f"\nEXECUTION HDD with k={k}")

        seed_set = high_degree_discount(inputHypergraph, k)
        output_seed_sets.append(seed_set)
    
    execution_time = (time.time() - start_time)

    print("\nComplete output")
    print(output_seed_sets)

    # save output to JSON file
    json_object = json.dumps(output_seed_sets, indent=1)
    output_file = open(args["output_file_path"], "w")
    output_file.write(json_object)
    output_file.close()
