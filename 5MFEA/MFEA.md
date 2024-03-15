# Multi-Task Evolutionary Algorithm for TSP

This project implements a Multi-Task Evolutionary Algorithm (MFEA) to solve multiple instances of the Traveling Salesman Problem (TSP). The code is written in Python and makes use of several libraries including numpy, pandas, and matplotlib to manage data and perform calculations efficiently.

## Functions

- **TSP Route Distance Calculation**: Computes the total distance of a TSP route using the Euclidean distance.
- **Population Initialization**: Generates a random initial population for the genetic algorithm with a specified size.
- **Skill Factor Assignment**: Ensures that each task has a minimum proportion of individuals with the required skill factor.
- **Fitness Evaluation**: Calculates the fitness of an individual based on the path length for its assigned task.
- **Roulette Wheel Selection**: Selects individuals for reproduction based on their fitness.
- **Ordered Crossover**: Performs ordered crossover between two parent individuals to produce offspring.
- **Swap Mutation**: Mutates individuals by swapping two cities in the route.
- **Data Loading**: Loads TSP data from CSV files containing city coordinates.
- **MFEA Main Function**: Orchestrates the evolutionary process for multiple tasks, evolving a population to solve each TSP instance.


## Dependencies
- numpy
- random
- csv
- matplotlib
- pandas


## Usage

To use the MFEA algorithm, you'll need to specify the paths to the TSP data files and set the algorithm's parameters, including population size, number of generations, mutation rate, crossover rate, and the minimum task proportion.

Example usage:

```sh
python MFEA.py
```

## Parameter Testing

The main script defines a range of parameters to test the algorithm's performance. It runs the MFEA for different combinations of population size, number of generations, mutation rate, crossover rate, and minimum task proportions.

## Results

The results of the parameter testing are collected in a list, converted to a pandas DataFrame, and then saved to a CSV file named `tsp_results.csv`.



