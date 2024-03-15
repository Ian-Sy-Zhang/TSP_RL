# Genetic Algorithm for TSP Optimization

This Python program implements a genetic algorithm to optimize routes for the Travelling Salesman Problem (TSP) while considering profits and time windows for each point.

## Functions

- **Distance Calculation**: Compute the Euclidean distance between two points.
- **Time Window Violations**: Calculate penalty for early or late arrival compared to a specified time window for each customer.
- **Total Profit Calculation**: Sum up the profits for a given route.
- **Total Distance Calculation**: Calculate the sum of distances for the entire route.
- **Population Initialization**: Generate an initial population of possible routes.
- **Parent Selection**: Select parent routes for breeding based on fitness (inverse of route distance).
- **Crossover**: Combine two parent routes to create new offspring routes.
- **Mutation**: Introduce random changes to offspring routes to maintain genetic diversity.
- **Data Loading**: Load customer points and related data from a CSV file.

## Dependencies
- numpy
- random
- csv
- matplotlib
- pandas


## Usage

To run the genetic algorithm with default parameters:

```bash
python WindowTSP.py
```


## Customization

You can customize the genetic algorithm by adjusting the following parameters:

- `pop_size`: The size of the population.
- `num_generations`: The number of generations to evolve the population.
- `mutation_rate`: The probability of mutating a gene in an individual.
- `crossover_rate`: The probability of crossing over parents to create offspring.

## Results

The program outputs:
- The best route found after the final generation.
- A plot of the best fitness scores by generation saved as a PNG file.



