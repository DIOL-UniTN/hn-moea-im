import json
import hypergraphx as hgx

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