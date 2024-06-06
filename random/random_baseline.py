import os
import argparse
import random
import hypergraphx as hgx
import json
from datetime import datetime
from loaders import load_hypergraph

def read_arguments():
    parser = argparse.ArgumentParser(description="Influence Maximization on Hypergraph Networks")

    parser.add_argument("--no_runs", type=int, default=5, help="Number of runs to be executed.")
    parser.add_argument("--min_seed_nodes", type=int, default=1, help="Minimum number of nodes in a seed set.")
    parser.add_argument("--max_seed_nodes", type=float, default=100, help="Maximum number of nodes in a seed set.")
    parser.add_argument("--random_seed", type=int, default=42, help="Seed to initialize the pseudo-random number generation.")
    parser.add_argument("--hypergraph_path", type=str, default="dataset/small/small.json", help="File path of the JSON file encoding the input hypergraph network.")
    parser.add_argument('--no_simulations', type=int, default=1, help='Number of simulations for spread calculation.')
    parser.add_argument('--k_step', type=int, default=1, help='The algorithm executes High-degree algorithm for all several values of k (seed set size) within the interval [min-k, max-k]. With this parameter we specify how divide this interval. For example, if k=2 and min-k=1 and max_k=5 then we are going to execute the algorithm for k=1, k=3, k=5.')

    parser.add_argument('--output_file_name', type=str, default="random.json", help='JSON file name where to store the individuals of the solution at the end of the execution.')
    parser.add_argument('--out_dir', default='output/', type=str, help='Location of the output directory.')

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
    #print(inputHypergraph.get_edges())

    # calculate max seed set size based on network size
    max_seed_set_size = int(args["max_seed_nodes"])
    print(f"max_seed_set_size: {max_seed_set_size}")

    for r in range(args["no_runs"]):
        # create directory for saving results of the run
        output_folder_run_path = output_folder_path+"/"+str(r+1)
        os.makedirs(output_folder_run_path)

        # add max seed set size nodes to seed_set sampling uniformly at random  
        seed_set = rng.sample(inputHypergraph.get_nodes(), max_seed_set_size)

        output_seed_sets = list()
        for k in range(args["min_seed_nodes"], max_seed_set_size+1, args["k_step"]):
            print(f"\nEXECUTION random baseline with k={k}")
            s = seed_set[:k]
            output_seed_sets.append(s)

        print("\nComplete output")
        print(output_seed_sets)

        # save output to JSON file
        json_object = json.dumps(output_seed_sets, indent=1)
        output_file = open(f"{output_folder_run_path}/{args['output_file_name']}", "w")
        output_file.write(json_object)
        output_file.close()
