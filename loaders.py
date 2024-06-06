from typing import Dict, Set, Tuple
import hypergraphx as hgx
import json
import argparse
import matplotlib.pyplot as plt

def load_hypergraph(file_path:str)->hgx.Hypergraph:
    """
    Load a hypergraph from a JSON file.

    Parameters
    ----------
    file_path : str
        File path of the JSON file where the hypergraph is stored.

    Returns
    -------
        Hypergraph object.
    """
    print("\nloading hypergraph from file, this might take a while...")
    json_file = open(file_path)
    json_object = json.load(json_file)
    json_hypergraph = hgx.Hypergraph(json_object)
    print("hypergraph loaded.")
    return json_hypergraph

def save_hypergraph(hypergraph, file_path : str) -> None:
    """
    Save the hypergraph in a JSON file.
    Format: {0: {1,2,3}, 1: {3,4,5} ... E: {n_1, n_2, ..., n_d}}
    Where 0, 1, ..., E are hyperedges numerical identifiers.
    n_1, n_2, ..., n_d are the nodes which populare a hyperedge.

    Parameters
    ----------
    hypergraph : Hypergraph
        The hypergraph of interest.
    file_path : str
        File path of a json file where to store the hypergraph.

    Returns
    -------
    None
    """
    print("\nsaving hypergraph to file, this might take a while...")
    json_object = json.dumps(hypergraph.get_edges(), indent=1)
    
    output_file = open(file_path, "w")
    output_file.write(json_object)
    output_file.close()
    print("hypergraph saved.")

def read_arguments():
    parser = argparse.ArgumentParser(description="Hypergraph dataset loader.")
    
    parser.add_argument("--input_file_path", type=str, default="dataset/small/small.json", help="File path of the JSON where the hypergraph is encoded.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    # === load hypergraph ==============================================================================================
    hypergraph = load_hypergraph(args["input_file_path"])
    print(hypergraph)
    # ==================================================================================================================

    # === write hyperdegree ============================================================================================
    node_hyperdegrees = [hypergraph.degree(n) for n in hypergraph.get_nodes()]
    node_hyperdegrees_avg = sum(node_hyperdegrees)/len(node_hyperdegrees)
    node_hyperdegrees_max = max(node_hyperdegrees)
    node_hyperdegrees_std = (sum((x - node_hyperdegrees_avg) ** 2 for x in node_hyperdegrees) / len(node_hyperdegrees))** 0.5
    print(f"\n===\nNode's hyperdegree\nAvg:{node_hyperdegrees_avg}\tStd:{node_hyperdegrees_std}\tMax:{node_hyperdegrees_max}\n===\n")
    # ==================================================================================================================

    # === write degree =================================================================================================
    node_degrees = [len(hypergraph.get_neighbors(n)) for n in hypergraph.get_nodes()]
    node_degrees_avg = sum(node_degrees)/len(node_degrees)
    node_degrees_max = max(node_degrees)
    node_degrees_std = (sum((x - node_degrees_avg) ** 2 for x in node_degrees) / len(node_degrees))** 0.5
    print(f"\n===\nNode's degree\nAvg:{node_degrees_avg}\tStd:{node_degrees_std}\tMax:{node_degrees_max}\n===\n")
    # ==================================================================================================================

    # === write hyperedge size =========================================================================================
    hyperedge_sizes = [len(e) for e in hypergraph.get_edges()]
    hyperedge_sizes_avg = sum(hyperedge_sizes)/len(hyperedge_sizes)
    hyperedge_sizes_max = max(hyperedge_sizes)
    hyperedge_sizes_std = (sum((x - hyperedge_sizes_avg) ** 2 for x in hyperedge_sizes) / len(hyperedge_sizes))**0.5
    print(f"\n===\nHyperedge size\nAvg:{hyperedge_sizes_avg}\tStd:{hyperedge_sizes_std}\tMax:{hyperedge_sizes_max}\n===\n")
    # ==================================================================================================================


    # === distribution of sizes of the hyperedges in the hypergraph ====================================================
    hypergraph_size_distribution = hypergraph.distribution_sizes()  # dict[int, int]
                                                                    # hyperedge order
                                                                    # how many hyperedges with that order
    print(f"hypergraph_size_distribution: {hypergraph_size_distribution}")

    # flat the distribution into a list
    sorted_orders = sorted(hypergraph_size_distribution.keys())
    hypergraph_size_distribution_list = [hypergraph_size_distribution[i] for i in sorted_orders]

    # plot histogram
    plt.bar(sorted_orders, hypergraph_size_distribution_list)
    plt.xlabel('Hyperedge Order')
    plt.ylabel('Count')
    plt.title('Hypergraph Size Distribution')
    plt.yscale('log')
    plt.grid()
    plt.savefig(f"output/size_histogram.png", dpi=300)
    plt.show()
    # ==================================================================================================================
    # plot node degree distribution
    node_degrees = sorted(node_degrees)
    plt.hist(node_degrees, bins=range(max(node_degrees) + 2), alpha=0.5)
    plt.axvline(node_degrees_avg, color='blue', linestyle='--')
    plt.axvline(node_degrees_avg + node_degrees_std, color='blue', linestyle=':')
    plt.axvline(node_degrees_avg - node_degrees_std, color='blue', linestyle=':')
    plt.xlabel('degree')
    plt.ylabel('# nodes')
    plt.title('degree distribution')
    plt.tight_layout()
    plt.savefig(f"output/degree_distribution.png", dpi=300)
    plt.show()
    # ==================================================================================================================
    # plot node hyperdegree distribution
    node_hyperdegrees = sorted(node_hyperdegrees)
    plt.hist(node_hyperdegrees, bins=range(max(node_hyperdegrees) + 2), alpha=0.5)
    plt.axvline(node_hyperdegrees_avg, color='blue', linestyle='--')
    plt.axvline(node_hyperdegrees_avg + node_hyperdegrees_std, color='blue', linestyle=':')
    plt.axvline(node_hyperdegrees_avg - node_hyperdegrees_std, color='blue', linestyle=':')
    plt.xlabel('hyperdegree')
    plt.ylabel('# nodes')
    plt.title('hyperdegree distribution')
    plt.tight_layout()
    plt.savefig(f"output/hyperdegree_distribution.png", dpi=300)
    plt.show()