# Influence Maximization in Hypergraph Networks using Multi-Objective Evolutionary Algorithm
*Conference [PPSN 2024](https://ppsn2024.fh-ooe.at/)* </br>
The goal of influence maximization (IM) is to reach the maximum number of entities in a network, starting from a small set of seed nodes, and assuming a model for information propagation. While this task has been widely studied in ordinary graph networks, IM in hypergraphs (where hyperedges represent interactions among more than two nodes) has not been adequately explored yet. The purpose of this study is to propose a multi-objective evolutionary algorithm to tackle the IM problem in hypergraph networks.<br>

[![arXiv](https://img.shields.io/badge/arXiv-2404.05621-b31b1b.svg)](https://arxiv.org/pdf/2405.10187)
 

```bibtex
@article{genetti2024influence,
  title={Influence Maximization in Hypergraphs using Multi-Objective Evolutionary Algorithms},
  author={Genetti, Stefano and Ribaga, Eros and Cunegatti, Elia and Lotito, Quintino Francesco and Iacca, Giovanni},
  journal={arXiv preprint arXiv:2405.10187},
  year={2024}
}
```

<p align="center">
<img src="hypergraph-im-visualization.png" width="500">
</p>

## Requirements
Before getting started, make sure you have installed all the requirements.
```
pip install -r requirements.txt
```

## Structure
The repository is structured as follows:
```
    .
    ├── data                            # Hypergraphs dataset
    ├── ea                              # Files implementing the inspyred functions (evaluator, mutator, ...)
    ├── greedy                          # Implementation of the high-degree baseline
    ├── random                          # Implementation of the random baseline
    ├── hdd                             # Implementation of the HDD baseline
    ├── smart_initialization.py         # Code for generating the initial population as described in the paper
    ├── moea.py                         # Source code HN-MOEA
    ├── main.py                         # Code main file
    └── monte_carlo_max_hop.py          # Propagation models
```

## External libraries and codes
### HyperGraphX Python Library
In this implementation in order to represent and handle hypergraphs we use the library **HGX**.
- GitHub: [https://github.com/HGX-Team/hypergraphx.git](https://github.com/HGX-Team/hypergraphx.git)
- Paper:  [https://github.com/HGX-Team/hypergraphx.git](https://github.com/HGX-Team/hypergraphx.git)
### HCI-TM-algorithm
The code of HCI-1 and HCI-2 baseline algorithms analyzed in the paper have been taken from the GitHub repository made available by the original authors.
- GitHub: [https://github.com/QDragon18/Influence-Maximization-based-on-Threshold-Model-in-Hypergraphs.git](https://github.com/QDragon18/Influence-Maximization-based-on-Threshold-Model-in-Hypergraphs.git)
- Paper:  [https://doi.org/10.1063/5.0178329](https://doi.org/10.1063/5.0178329)

## Contribution
Authors:
- Stefano Genetti, MSc Student University of Trento (Italy), stefano.genetti@studenti.unitn.it
- Eros Ribaga, MSc Student University of Trento (Italy), eros.ribaga@studenti.unitn.it
- Elia Cunegatti, Ph.D Student University of Trento (Italy), elia.cunegatti@unitn.it
- Francesco Quintino Lotito, Ph.D Student University of Trento (Italy), quintino.lotito@unitn.it
- Giovanni Iacca, Associate Professor University of Trento (Italy), giovanni.iacca@unitn.it

For every type of doubts/questions about the repository please do not hesitate to contact us.
