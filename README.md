# CS224W Colab 5: Heterogeneous Graph Neural Networks

## Overview

This project is based on **CS224W Colab 5** and focuses on building **Heterogeneous Graph Neural Networks** for node classification.

Unlike previous graph neural network models that operate on homogeneous graphs, this notebook introduces **heterogeneous graphs**, where nodes and edges can have different types. The project uses **DeepSNAP**, **PyTorch Geometric**, and **NetworkX** to represent, process, and train models on heterogeneous graph data.

The main objective is to implement a heterogeneous GNN model and apply it to a node property prediction task on the **ACM heterogeneous graph dataset**.

## Project Goals

The goals of this project are to:

1. Understand the difference between homogeneous and heterogeneous graphs.
2. Learn how to represent heterogeneous graphs using NetworkX and DeepSNAP.
3. Assign node types, edge types, node features, and labels.
4. Analyze heterogeneous message types.
5. Split heterogeneous graph data into train, validation, and test sets.
6. Implement a custom heterogeneous graph convolution layer.
7. Implement message-type aggregation using mean aggregation and attention.
8. Train and evaluate a heterogeneous GNN for node classification.

## Main Concepts

### Heterogeneous Graphs

A heterogeneous graph is a graph with multiple types of nodes and/or edges.

For example, in an academic graph, nodes may represent papers, authors, and subjects, while edges may represent relationships such as:

- author writes paper
- paper belongs to subject
- paper cites paper

This additional type information allows the model to learn richer representations than a standard homogeneous graph.

### Node Types and Edge Types

In this project, nodes and edges are assigned type labels.

For the introductory NetworkX example, nodes are divided into two types based on club membership:

- `n0`: nodes in the “Mr. Hi” club
- `n1`: nodes in the “Officer” club

Edges are divided into three types:

- `e0`: edges within the “Mr. Hi” club
- `e1`: edges within the “Officer” club
- `e2`: edges between the two clubs

### Message Types

In heterogeneous GNNs, message passing depends on the source node type, edge type, and destination node type.

A message type is represented as:

```text
(source node type, edge type, destination node type)
