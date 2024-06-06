import hypergraphx as hgx
import argparse
import json

def save_hci_solution(input_file_path: str, output_file_path: str):
    '''
    Read the TXT containing the solution
    proposed by HCI algorithm
    (https://github.com/QDragon18/Influence-Maximization-based-on-Threshold-Model-in-Hypergraphs.git).

    Format: one line with a list of MAX_SEED_SET_SIZE nodes
    n_1, n_2, ..., n_100

    Store the proposed solution in a JSON file with one solution for each value
    of k.
    [
    [n_1],
    [n_1, n_2],
    ...
    [n_1, ..., n_100]
    ]

    Parameters
    ----------
    input_file_path : str
        File path of the TXT file with the solution proposed by HCI influence maximization algorithm.
    output_file_path : str
        File path of the JSON file with the solution proposed by HCI with the encoding commonly adopted in our code.

    Returns
    -------
    None
    '''
    # read solution from file
    input_file = open(input_file_path, 'r')
    hci_solution = [int(x) for x in input_file.readline().strip().split()]
    input_file.close()

    # remove zero entries at the end of the solution proposal
    while hci_solution and hci_solution[-1] == 0:
        hci_solution.pop()

    # generate subsets
    subsets = []
    for i in range(1, len(hci_solution) + 1):
        subsets.append(hci_solution[:i])

    # write solution to JSON file
    output_file = open(output_file_path, 'w')
    json.dump(subsets, output_file, indent=1)
    output_file.close()

def read_arguments():
    parser = argparse.ArgumentParser(description="JSON Hypergraph to QDragon dataset converter.")
    
    parser.add_argument("--input_file_path", type=str, default="output/hci.txt", help="File path of the TXT file with the solution proposed by HCI influence maximization algorithm.")
    parser.add_argument("--output_file_path", type=str, default="output/hci.json", help="File path of the JSON file with the solution proposed by HCI with the encoding commonly adopted in our code.")

    args = parser.parse_args()
    args = vars(args)

    return args

if __name__ == '__main__':
    args = read_arguments()
    save_hci_solution(args["input_file_path"], args["output_file_path"])