# Genetic Algorithm for the Traveling Salesman Problem (TSP)

## Overview

This repository contains a Python implementation of a genetic algorithm to solve the Traveling Salesman Problem (TSP). The TSP is a well-known optimization problem, which aims to find the shortest possible route that visits a list of cities and returns to the origin city.

## Functions

- Calculation of the distance between two points
- Calculation of the total distance of a route
- Creation of an initial population for the genetic algorithm
- Selection process for choosing the fittest individuals as parents
- Crossover function to combine two parents to create offspring
- Mutation function to introduce variability into the offspring
- Functions to load data points from a CSV file
- Implementation of the genetic algorithm to find the best route and distance

## Dependencies

To run this script, you will need the following Python libraries:

- numpy
- random
- itertools
- csv
- matplotlib.pyplot
- pandas



## Usage

The entry point of the program is the `main` function, which defines the filename for the CSV containing the TSP data and sets up a range of parameters for testing the genetic algorithm.

To run the genetic algorithm with the default settings, simply execute the script:

```bash
python TSP.py
```

You can modify the parameters `pop_sizes`, `num_generationss`, `mutation_rates`, and `crossover_rates` in the main block to test different configurations and observe their impact on the algorithm's performance.

## Result

The script will output the best route found and its corresponding total distance. It will also save a plot of the best distances per generation as a `.png` file and save the test results as a `tsp_results.csv` file.
