import argparse
import random
import hypergraphx as hgx
import json
from loaders import load_hypergraph
import time

def read_arguments():
    parser = argparse.ArgumentParser(description="Influence Maximization on Hypergraph Networks")
    
    parser.add_argument("--min_seed_nodes", type=int, default=1, help="Minimum number of nodes in a seed set.")
    parser.add_argument("--max_seed_nodes", type=float, default=100, help="Maximum number of nodes in a seed set.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--hypergraph_path", type=str, default="dataset/small/small.json", help="File path of the JSON file encoding the input hypergraph network.")
    parser.add_argument('--degree', default="hyperdegree", choices=["degree", "hyperdegree"], help='Degree OR Hyperdegree.')
    parser.add_argument('--k_step', type=int, default=1, help='The algorithm executes High-degree algorithm for all several values of k (seed set size) within the interval [min-k, max-k]. With this parameter we specify how divide this interval. For example, if k=2 and min-k=1 and max_k=5 then we are going to execute the algorithm for k=1, k=3, k=5.')

    parser.add_argument("--output_file_path", type=str, default="output/high_degree_solution/high_degree.json", help="File path of the output JSON file.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    rng = random.Random(args["random_seed"])

    # load hypergraph
    inputHypergraph = load_hypergraph(args["hypergraph_path"])
    print(inputHypergraph)
    #print(inputHypergraph.get_edges())

    start_time = time.time()

    # get node degrees or hyperdegrees
    node_degree = dict()    # key: node id ; value: node degree
    for node in inputHypergraph.get_nodes():
        node_degree[node] = len(inputHypergraph.get_neighbors(node)) if args["degree"]=="degree" else inputHypergraph.degree(node)

    # sort nodes according to their degree
    node_sorted = [n[0] for n in sorted(node_degree.items(), key=lambda x: x[1], reverse=True)]
    print(len(node_sorted))

    # calculate max seed set size based on network size
    max_seed_set_size = int(args["max_seed_nodes"])
    print(f"max_seed_set_size: {max_seed_set_size}")

    output_seed_sets = list()
    for k in range(args["min_seed_nodes"], max_seed_set_size+1, args["k_step"]):
        print(f"\nEXECUTION high_degree with k={k}")

        # add nodes n to seed_set in order of decreasing out-degrees
        seed_set = node_sorted[:k]
        output_seed_sets.append(seed_set)
    
    execution_time = (time.time() - start_time)

    print("\nComplete output")
    print(output_seed_sets)

    # save output to JSON file
    json_object = json.dumps(output_seed_sets, indent=1)
    output_file = open(args["output_file_path"], "w")
    output_file.write(json_object)
    output_file.close()