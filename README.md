# Jump Point Search Pathfinding in 4-connected Grids

## Overview

This project combines a classic Snake game with a demonstration of **JPS4**—a powerful pathfinding algorithm designed for efficient navigation in 4-connected grid environments. In addition to the game, a dedicated component compares the performance and results of the JPS4 algorithm with the A\* algorithm.

## What Is JPS4?

**JPS4** is an adaptation of **Jump Point Search (JPS)** tailored for grids that allow movement only in the four cardinal directions (**up, down, left, right**). The algorithm speeds up the traditional A\* search by "jumping" over unnecessary nodes, significantly reducing the number of expanded nodes while still guaranteeing an optimal path.

### Key Points of JPS4

- **Canonical Ordering**: Enforces a strict order for expanding movements (e.g., **horizontal-first** or **vertical-first**) to prune redundant paths and define a unique optimal route.
- **Pruning Neighbors**: Considers only *natural* and *forced* neighbors, greatly reducing the number of nodes expanded.
- **Jumping Over Empty Space**: Instead of moving step-by-step, jump points are introduced at strategic corners to skip over multiple cells efficiently.

- **Recursive Jump Mechanism**: The `jump()` function explores straight paths until it hits the goal, a forced neighbor, or a direction change.

Despite pruning and skipping nodes, **JPS4 guarantees the same optimal path cost as A\***—but typically with far fewer node expansions.

## Why Use JPS4 Over A\*?

- **Performance Gains**: In complex grid environments with many obstacles, JPS4 can significantly reduce the number of nodes evaluated compared to A\*, leading to faster pathfinding.

- **Guaranteed Optimality**: The algorithm maintains the same optimal path cost as A\* while offering improved efficiency.

## Benchmarking

Based on this [Repository](https://github.com/GurkNathe/Pathfinding-Algorithms) by @GurkNathe, we can generate random mazes using [maze.py](https://github.com/OrWestSide/python-scripts/blob/master/maze.py) that simulate complex environments to evaluate A\* vs. JPS4 under various conditions.

## Technical Challenges and Implementation

- **Recursion Depth**: the `jump()` function may hit stack limits in large environments. 
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

---