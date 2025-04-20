# Jump Point Search Pathfinding in 4-connected Grids

## Overview

This project combines a classic Snake game with a demonstration of **JPS4**—a powerful pathfinding algorithm designed for efficient navigation in 4-connected grid environments. In addition to the game, a dedicated component compares the performance and results of the JPS4 algorithm with the A\* algorithm. The benchmarking tool focuses on path length, search space size, and success rate.

## What Is JPS4?

**JPS4** is an adaptation of **Jump Point Search (JPS)** tailored for grids that allow movement only in the four cardinal directions (**up, down, left, right**). The algorithm speeds up the traditional A\* search by "jumping" over unnecessary nodes, significantly reducing the number of expanded nodes while still guaranteeing an optimal path.

### Key Points of JPS4

- **Canonical Ordering**: Enforces a strict order for expanding movements (e.g., **horizontal-first** or **vertical-first**) to prune redundant paths and define a unique optimal route.
- **Pruning Neighbors**: Considers only *natural* and *forced* neighbors, greatly reducing the number of nodes expanded.
- **Jumping Over Empty Space**: Instead of moving step-by-step, jump points are introduced at strategic corners to skip over multiple cells efficiently.

- **Recursive Jump Mechanism**: The `jps_jump()` function explores straight paths until it hits the goal, a forced neighbor, or a direction change.

Despite pruning and skipping nodes, **JPS4 guarantees the same optimal path cost as A\***—but typically with far fewer node expansions.

## Why Use JPS4 Over A\*?

- **Performance Gains**: In complex grid environments with many obstacles, JPS4 can significantly reduce the number of nodes evaluated compared to A\*, leading to faster pathfinding.

- **Guaranteed Optimality**: The algorithm maintains the same optimal path cost as A\* while offering improved efficiency.

## Technical Challenges and Implementation

- **Recursion Depth**: the `jps_jump()` function may hit stack limits in large environments. 
- **Edge Case Handling**: Narrow corridors or obstacle-heavy regions require careful logic to preserve optimality.

- **Grid & Heap Optimization**: Efficient data structures (e.g., priority queues for the open list) can help keep runtime and memory usage in check.

## Expected Outcomes

- **Consistent Optimality**: JPS4 should always return the same optimal path as A\*.

- **Enhanced Performance**: Particularly in environments with many obstacles or open areas, JPS4 is expected to outperform A\* in terms of speed.
- **Insightful Analysis**: Any discrepancies between the two algorithms provide valuable insights for further optimization.

## How It Relates to the Snake Game

- **AI Pathfinding**: The snake's movement is driven by the JPS4 algorithm, allowing it to rapidly and efficiently reach its food targets.

- **Real-Time Efficiency**: The optimized pathfinding ensures that the game remains responsive and engaging, even as paths are recalculated every game tick.

- **Algorithm Comparison for Research**: The inclusion of the A\* comparison module deepens the research aspect of the project, providing a clear analysis of the benefits and limitations of each algorithm.

## Requirements

- Python
- Required packages:
  - matplotlib
  - numpy
  - tkinter
  
Install dependencies:

```bash
pip install matplotlib numpy tkinter
```

## Running the Snake Game

Navigate to the game directory:

```bash
cd '.\Checkpoint 3\src\Snake Game Code'
```

Run:

```bash
python snake_game_jps4.py
```

## Running the Benchmarks

The project includes several benchmark types to evaluate pathfinding performance:

Navigate to the benchmark directory:

```bash
cd '.\Checkpoint 3\src\Benchmarking Code'
```

### Quick Benchmark

Provides a fast comparison between JPS4 and A* with different obstacle densities.

```bash
python main.py --benchmark-type quick
```

This runs 3 trials with obstacle densities of 0.40, 0.60, and 0.80 on a 100x100 grid.

### Research Benchmark

Runs a comprehensive set of tests with varying obstacle densities to provide a thorough comparison.

```bash
python main.py --benchmark-type research
```

This runs trials with obstacle densities from 0.00 to 0.80 (in increments of 0.10), providing detailed metrics on success rates, node expansion, execution time, and path lengths.

### Maze Benchmark

Tests performance specifically in maze-like environments with different grid sizes.

```bash
python main.py --benchmark-type maze
```

This runs 5 trials each on maze grids of sizes 50x50, 100x100, 200x200, and 300x300.

### Large Grid Benchmark

Evaluates how the algorithms scale with increasing grid sizes.

```bash
python main.py --benchmark-type large
```

This benchmark tests performance on grids ranging from 100x100 to 500x500.

## Benchmark Results

Results are saved in the `tests\benchmark_results` directory (can change it through `path_dir` in `src\Algorithms\helper.py`), organized by benchmark type. Currently, results are available in the repository for the following types: Large, Maze, and Research.

- `benchmark_report.txt`: Overall summary of algorithm performance
- Individual benchmark folders (e.g., `maze`, `research`) contain:
  - Detailed reports for each trial
  - Performance plots comparing JPS4 vs A*
  - CSV data files of raw results

## Custom Benchmarks

You can create custom benchmarks by modifying parameters:

```bash
python main.py --benchmark-type custom --grid-size 200 --obstacle-density 0.3 --trials 10
```

## Interpreting Results

- **Success Rate**: Percentage of trials where a path was found (when one exists)
- **Average Time**: Average execution time in milliseconds
- **Nodes Expanded**: Number of nodes processed during search
- **Path Length**: Length of the final path found
- **Ratios**: JPS4-to-A* comparison (values > 1 mean JPS4 used more resources)

## Troubleshooting

If you encounter any issues:

1. Ensure all dependencies are installed
2. For very large benchmarks, increase available memory or reduce grid size

## Team

- **Muhammad Sameer Kamani** - Computer Science, Habib University

- **Muhammad Ibad Nadeem** - Computer Science, Habib University
- **Muhammad Taqi** - Computer Science, Habib University
- **Supervisor**: Dr. Waqar Saleem, Habib University

---