# Node Classification

When it comes to graph data, the most common task is node classification. In this task, we are given a graph where each node has a label, and we are interested in predicting the label of the nodes for which the label is unknown. Representing our data as a graph allows us to leverage the relationships between nodes to make better predictions.

## Dataset

### CORA

The Cora dataset consists of `2708` scientific publications classified into one of `7` classes:

- `Case_Based`
- `Genetic_Algorithms`
- `Neural_Networks`
- `Probabilistic_Methods`
- `Reinforcement_Learning`
- `Rule_Learning`
- `Theory`

Each publication in the dataset is described by a `0/1`-valued word vector indicating the absence/presence of the corresponding word from the dictionary. The dictionary consists of `1433` unique words. The citation network consists of `5429` links.

You can download the dataset from [here](https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz). However, the dataset is already available in the `data` directory.

## Problem Statement

The objective is to develop a machine learning approach to predict the subjects of scientific papers.
