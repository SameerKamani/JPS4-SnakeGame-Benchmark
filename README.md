# JPS4 Snake Game

## Overview

This Snake game project uses **JPS4 (Jump Point Search for 4-connected grids)** under the hood for pathfinding. Below is a brief explanation of what JPS4 is and why it matters, without diving into specific code details.

## What Is JPS4?

**JPS4** is an adaptation of **Jump Point Search (JPS)** designed for **4-connected grid environments** (i.e., where you can only move **up, down, left, or right**). Traditional JPS was introduced to speed up A* on uniform-cost grids by **jumping** over large sections of the map. JPS4 applies the same principle but only for the four cardinal directions.

## Key Points of JPS4

### 📌 Canonical Ordering
JPS4 imposes a strict order on how movements are expanded (e.g., **horizontal-first** or **vertical-first**). This approach eliminates many redundant paths that would otherwise appear in standard **A*** searches on a grid.

### ✂️ Pruning Neighbors
Instead of expanding every possible neighbor, JPS4 **prunes neighbors** that don’t contribute to an optimal path. This pruning is carefully done so that **optimality is still guaranteed**.

### 🚀 Jumping Over Empty Space
Rather than moving step by step, **JPS4 will jump over open cells** until it either:
- **Reaches the goal**,
- **Encounters an obstacle**, or
- **Arrives at a forced neighbor** (where a turn or decision is needed).

### ✅ Maintaining Optimality
Despite skipping nodes, JPS4 still finds **optimal solutions**. It does so by carefully identifying **jump points**—the critical cells where decisions or turns must be made.

## Why Use JPS4 Over A*?

- **⚡ Performance Gains**: On maps with many obstacles, **JPS4 can significantly reduce** the number of expanded nodes compared to **A***, speeding up pathfinding.
- **🔍 Still Optimal**: JPS4 ensures the final path cost is **the same as A*** but achieves this with **fewer expansions**.

## How It Relates to the Snake Game

In this project, the **Snake’s AI** uses **JPS4-based pathfinding** to quickly find a path to the food in a **4-connected grid**:

- **🧠 Efficiency**: Jumping through open corridors and pruning useless moves makes the AI much more **responsive**.
- **⏳ Real-Time**: The algorithm is **fast enough** to run at every game tick without slowing down gameplay.

## 🏆 Credits

The code is for **demonstration purposes**, combining a **classic Snake game** with **JPS4 pathfinding concepts**.

- **JPS4 Concept Credits**: Based on research by **Daniel Harabor** and **Alban Grastien**, adapting **Jump Point Search** to **4-connected grids**.
- For more information on Jump Point Search and its variants, see the original paper on **JPS (2011)** and subsequent works adapting it to different grid configurations.

---
🚀 **Enjoy the game and explore JPS4-based AI!** 🐍
