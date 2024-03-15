# Cluster-Based Genetic Algorithm for TSP

This repository contains a Python implementation of a cluster-based genetic algorithm to solve the Traveling Salesman Problem (TSP). The code consists of several key components including data loading, distance calculations, K-means clustering, and genetic algorithm operations such as selection, crossover, and mutation.

## Overview

The Traveling Salesman Problem (TSP) is a classic algorithmic problem in the field of computer science and operations research. It focuses on finding the shortest possible route that visits a set of points exactly once and returns to the origin point. This implementation uses a cluster-based genetic algorithm approach to break down the problem into more manageable sub-problems and solve them more efficiently.

## Functions

- Load customer location data from a CSV file.
- Calculate the distance between two points and the total distance of a route.
- Perform K-means clustering to group nearby points together.
- Implement genetic algorithm functions for evolving a population towards better solutions.
- Solve the TSP for clustered data to find an optimal or near-optimal solution.


## Dependencies

- Python 3.x
- Numpy
- Matplotlib
- pandas
- numpy
- scikit-learn


## Usage

To use this code, follow these steps:

1. Prepare your data in a CSV format where each row represents a point with its coordinates.
2. Load the data using the `load_data` function.
3. Cluster the data points using the K-means algorithm with the `cluster_points` function.
4. Run the genetic algorithm on the clustered data with the `genetic_algorithm` function.
5. The main function `run_cluster_genetic_algorithm` combines all the steps and solves the TSP for the given data.

```bash
python Large_TSP.py
```

## Customization

You can customize the genetic algorithm by adjusting the following parameters:

- `num_clusters`: The number of clusters to divide the data points into.
- `pop_size`: The size of the population in each generation.
- `num_generations`: The number of generations to run the genetic algorithm.
- `mutation_rate`: The rate at which mutations occur in the population.
- `crossover_rate`: The rate at which crossover occurs between parents to produce offspring.


## Results

To see the results of the genetic algorithm with different parameters, you can check the `tsp_results1.csv` file which is created after running the main program block. This CSV file will contain columns for `pop_size`, `num_generations`, `mutation_rate`, `crossover_rate`, and `best_fitness`.
