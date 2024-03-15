# Genetic Algorithm for the Traveling Salesman Problem

This repository contains a Python implementation of a genetic algorithm (GA) for solving the Traveling Salesman Problem (TSP), where the objective is to find the shortest possible route that visits a set of cities and returns to the origin city.

## Description

The TSP is a well-known optimization problem which has applications in logistics, planning, and the manufacture of microchips. This particular implementation uses a genetic algorithm, a search heuristic that mimics the process of natural selection, to find an approximate solution to the TSP.

## Functions

- Calculation of the Euclidean distance between two points.
- Loading of city coordinates and profits from a CSV file.
- Creation of an initial population of random routes.
- Calculation of the fitness of a route based on the total distance and profits.
- Selection of parent routes for breeding based on their fitness.
- Crossover and mutation functions to produce offspring from the parent routes.
- A function to fix any duplicated cities in a route after crossover.
- A function to run the genetic algorithm which includes the selection, crossover, mutation, and the replacement of the population with new offspring.
- Plotting of the best route distance by generation and saving the plot as a PNG file.
- Saving the results of the genetic algorithm runs with different parameters to a CSV file.


## Dependencies
- numpy
- random
- csv
- matplotlib
- pandas


## Usage

To use this genetic algorithm, you must have a CSV file (`TSP.csv`) with the coordinates of the cities and the associated profits. The CSV file should have a header with fields `city`, `x`, `y`, `profit`.

You can run the algorithm with custom parameters as follows:

```bash
python Multi_objective_TSP.py
```

After the algorithm has finished running, it will print out the best route found and the distance of that route. It will also save a plot of the best distance by generation to a PNG file.


## Customization

You can customize the genetic algorithm by adjusting the following parameters:

- `pop_size`: The size of the population in each generation.
- `num_generations`: The number of generations to run the genetic algorithm.
- `mutation_rate`: The rate at which mutations occur in the population.
- `crossover_rate`: The rate at which crossover occurs between parents to produce offspring.



## Results

To see the results of the genetic algorithm with different parameters, you can check the `tsp_results.csv` file which is created after running the main program block. This CSV file will contain columns for `pop_size`, `num_generations`, `mutation_rate`, `crossover_rate`, and `best_fitness`.
