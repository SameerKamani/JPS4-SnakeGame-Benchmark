# Jump Point Search Pathfinding in 4-connected Grids

## Overview

This project combines a classic Snake game with a demonstration of **JPS4**—a pathfinding algorithm designed for efficient navigation in 4-connected grid environments. In addition to the game, a dedicated component compares the performance and results of the JPS4 algorithm with the A\* algorithm.

## What Is JPS4?

**JPS4** is an adaptation of **Jump Point Search (JPS)** tailored for grids that allow movement only in the four cardinal directions (**up, down, left, right**). The algorithm speeds up the traditional A\* search by "jumping" over unnecessary nodes, significantly reducing the number of expanded nodes while still guaranteeing an optimal path.

### Key Points of JPS4

- **Canonical Ordering**: Enforces a strict order for expanding movements (e.g., **horizontal-first** or **vertical-first**), which eliminates redundant paths.
- **Pruning Neighbors**: Selectively expands only those neighbors that are essential for reaching the goal, maintaining optimality while reducing computation.
- **Jumping Over Empty Space**: Instead of moving step-by-step, JPS4 jumps across open cells until it reaches an obstacle, forced neighbor, or the goal.

- **Maintaining Optimality**: Despite skipping nodes, the algorithm ensures the final path remains optimal, matching the results of A\*.

## Why Use JPS4 Over A\*?

- **Performance Gains**: In complex grid environments with many obstacles, JPS4 can significantly reduce the number of nodes evaluated compared to A\*, leading to faster pathfinding.

- **Guaranteed Optimality**: The algorithm maintains the same optimal path cost as A\* while offering improved efficiency.

## Expected Outcomes

- **Consistent Optimality**: JPS4 should always return the same optimal path as A\*.

- **Enhanced Performance**: Particularly in environments with many obstacles or open areas, JPS4 is expected to outperform A\* in terms of speed.
- **Insightful Analysis**: Any discrepancies between the two algorithms provide valuable insights for further optimization.

## How It Relates to the Snake Game

- **AI Pathfinding**: The snake's movement is driven by the JPS4 algorithm, allowing it to rapidly and efficiently reach its food targets.

- **Real-Time Efficiency**: The optimized pathfinding ensures that the game remains responsive and engaging, even as paths are recalculated every game tick.

- **Algorithm Comparison for Research**: The inclusion of the A\* comparison module deepens the research aspect of the project, providing a clear analysis of the benefits and limitations of each algorithm.

---