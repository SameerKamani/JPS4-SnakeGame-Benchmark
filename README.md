# JPS4 Snake Game

## Overview

This project combines a classic Snake game with a demonstration of **JPS4 (Jump Point Search for 4-connected grids)**, a pathfinding algorithm designed for efficient navigation. In addition to the game itself, the repository now includes a module to **compare the performance and results of the JPS4 algorithm against the standard A\* algorithm**. This comparison aims to verify the research findings or dispute certain aspects by analyzing optimality, execution time, and node expansions.

## What Is JPS4?

**JPS4** is an adaptation of **Jump Point Search (JPS)** designed for **4-connected grid environments** (i.e., where movement is limited to **up, down, left, or right**). Traditional JPS was introduced to speed up A\* on uniform-cost grids by **jumping** over large sections of the map. JPS4 applies the same principle, but strictly for the four cardinal directions.

### Key Points of JPS4

- **Canonical Ordering**: JPS4 imposes a strict order on movement expansion (e.g., **horizontal-first** or **vertical-first**), eliminating many redundant paths present in standard A\* searches.
- **Pruning Neighbors**: The algorithm prunes neighbors that don’t contribute to an optimal path, ensuring that optimality is maintained while reducing unnecessary calculations.
- **Jumping Over Empty Space**: Instead of moving step by step, JPS4 "jumps" over open cells until it reaches the goal, encounters an obstacle, or finds a forced neighbor where a decision is required.
- **Maintaining Optimality**: Even though it skips nodes, JPS4 still guarantees optimal pathfinding by identifying critical jump points.

## Why Use JPS4 Over A\*?

- **Performance Gains**: On maps with many obstacles, JPS4 can reduce the number of expanded nodes dramatically compared to A\*, resulting in faster pathfinding.
- **Guaranteed Optimality**: Despite its speed, JPS4 maintains the same optimal path cost as A\*.

## Comparing JPS4 with A\*

### Motivation for Comparison

The repository now includes a dedicated component to directly compare **JPS4** and **A\***:
- **Research Verification**: To confirm that JPS4 provides a significant performance improvement while maintaining optimality.
- **Result Dispute**: To challenge or validate previous research findings by observing any discrepancies in real-time gameplay or test scenarios.

### Comparison Methodology

- **Path Optimality**: Both algorithms are used to compute paths in identical grid scenarios. Their path costs are compared to ensure that JPS4 is indeed returning optimal paths as per A\*’s results.
- **Performance Metrics**: Metrics such as **execution time** and **number of nodes expanded** during the search are recorded. This data helps to evaluate the efficiency gains of JPS4 over A\*.
- **Visual Debugging**: In some test cases, the computed paths are overlaid on the game grid. This visual representation allows for immediate verification and easier debugging when results differ.

### Expected Outcomes

- **Consistent Optimality**: JPS4 should consistently return the same optimal path as A\*.
- **Enhanced Performance**: In scenarios with many obstacles or large open areas, JPS4 is expected to outperform A\* by reducing the number of computations.
- **Insightful Analysis**: Any deviations or edge cases discovered during comparisons provide valuable insights into potential improvements or limitations in either algorithm.

## How It Relates to the Snake Game

- **JPS4 for AI Pathfinding**: In the game, the snake uses the JPS4 algorithm to quickly locate and move towards the food within a 4-connected grid environment.
- **Real-Time Efficiency**: The efficient pathfinding provided by JPS4 ensures that the game remains responsive, even when the snake must calculate paths every tick.
- **Algorithm Comparison for Research**: The additional comparison module allows developers and researchers to directly observe the benefits and any potential drawbacks of JPS4 in real-world scenarios, enhancing the credibility and depth of the research.

## 🏆 Credits

- **JPS4 Concept Credits**: The project builds on research by **Daniel Harabor** and **Alban Grastien**, who initially developed Jump Point Search (JPS). The adaptation for 4-connected grids has been further refined in this project.
- **Research Comparison**: Special thanks to the community and researchers who have contributed to understanding the nuances of pathfinding algorithms through comparative analysis.
- For more information on Jump Point Search and its variants, refer to the original JPS paper (2011) and subsequent related works.

---
🚀 **Enjoy the game, explore JPS4-based AI, and dive into algorithm comparisons to further your understanding of pathfinding!** 🐍
