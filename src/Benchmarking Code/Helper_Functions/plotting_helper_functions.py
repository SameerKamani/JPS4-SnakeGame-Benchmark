import numpy as np
import os
import matplotlib.pyplot as plt

from Algorithms.helper import path_dir

def visualize_grid(grid_data, start, goal, filename):
    """
    Create a visualization of a grid and save it to a file
    """
    height = len(grid_data)
    width = len(grid_data[0]) if height > 0 else 0
    
    # Create figure
    plt.figure(figsize=(10, 10))
    
    # Create grid visualization
    grid_vis = np.zeros((height, width, 3))
    for y in range(height):
        for x in range(width):
            if grid_data[y][x]:
                grid_vis[y, x] = [0, 0, 0]  # Obstacle (black)
            else:
                grid_vis[y, x] = [1, 1, 1]  # Open (white)
    
    # Mark start and goal
    if 0 <= start.y < height and 0 <= start.x < width:
        grid_vis[start.y, start.x] = [0, 1, 0]  # Start (green)
    if 0 <= goal.y < height and 0 <= goal.x < width:
        grid_vis[goal.y, goal.x] = [1, 0, 0]  # Goal (red)
    
    plt.imshow(grid_vis)
    plt.title(f"Grid Example - Start: ({start.x}, {start.y}), Goal: ({goal.x}, {goal.y})")
    plt.savefig(filename, dpi=150)
    plt.close()

def plot_results(results, obstacle_densities):
    """
    Generate enhanced plots from benchmark results
    """
    # Create a results directory for overall plots
    os.makedirs(f"{path_dir}benchmark_results/overall", exist_ok=True)
    
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('JPS4 vs A* Performance Comparison Across Obstacle Densities', fontsize=18)
    
    # Calculate average values
    avg_times = {alg: [] for alg in results.keys()}
    avg_nodes = {alg: [] for alg in results.keys()}
    avg_path_lengths = {alg: [] for alg in results.keys()}
    success_rates = {alg: [] for alg in results.keys()}
    
    for alg in results.keys():
        for density in obstacle_densities:
            # Calculate success rate first as we'll use it for other metrics
            successes = results[alg][density]["success"]
            success_rate = sum(successes) / len(successes) * 100
            success_rates[alg].append(success_rate)
            
            # Calculate average execution time (only for successful paths)
            times = [t for t, s in zip(results[alg][density]["time"], successes) if s]
            avg_times[alg].append(sum(times) / len(times) if times else 0)
            
            # Calculate average nodes expanded (only for successful paths)
            nodes = [n for n, s in zip(results[alg][density]["nodes"], successes) if s]
            avg_nodes[alg].append(sum(nodes) / len(nodes) if nodes else 0)
            
            # Calculate average path length (only for successful paths)
            path_lengths = [l for l, s in zip(results[alg][density]["path_length"], successes) if s and l != float('inf')]
            avg_path_lengths[alg].append(sum(path_lengths) / len(path_lengths) if path_lengths else 0)
    
    # Set up colors and markers
    colors = {'JPS4': 'blue', 'AStar': 'red'}
    markers = {'JPS4': 'o', 'AStar': 's'}
    
    # Plot execution time
    for alg in results.keys():
        axs[0, 0].plot(obstacle_densities, avg_times[alg], marker=markers[alg], label=alg, 
                      color=colors[alg], linewidth=2, markersize=8)
    axs[0, 0].set_xlabel('Obstacle Density', fontsize=12)
    axs[0, 0].set_ylabel('Average Time (s)', fontsize=12)
    axs[0, 0].set_title('Execution Time vs. Obstacle Density', fontsize=14)
    axs[0, 0].legend(fontsize=12)
    axs[0, 0].grid(True, alpha=0.3)
    
    # Plot nodes expanded (search space size)
    for alg in results.keys():
        axs[0, 1].plot(obstacle_densities, avg_nodes[alg], marker=markers[alg], label=alg, 
                      color=colors[alg], linewidth=2, markersize=8)
    axs[0, 1].set_xlabel('Obstacle Density', fontsize=12)
    axs[0, 1].set_ylabel('Nodes Expanded', fontsize=12)
    axs[0, 1].set_title('Search Space Size vs. Obstacle Density', fontsize=14)
    axs[0, 1].legend(fontsize=12)
    axs[0, 1].grid(True, alpha=0.3)
    
    # Plot path length
    for alg in results.keys():
        axs[1, 0].plot(obstacle_densities, avg_path_lengths[alg], marker=markers[alg], label=alg, 
                      color=colors[alg], linewidth=2, markersize=8)
    axs[1, 0].set_xlabel('Obstacle Density', fontsize=12)
    axs[1, 0].set_ylabel('Path Length', fontsize=12)
    axs[1, 0].set_title('Path Length vs. Obstacle Density', fontsize=14)
    axs[1, 0].legend(fontsize=12)
    axs[1, 0].grid(True, alpha=0.3)
    
    # Plot success rate
    for alg in results.keys():
        axs[1, 1].plot(obstacle_densities, success_rates[alg], marker=markers[alg], label=alg, 
                      color=colors[alg], linewidth=2, markersize=8)
    axs[1, 1].set_xlabel('Obstacle Density', fontsize=12)
    axs[1, 1].set_ylabel('Success Rate (%)', fontsize=12)
    axs[1, 1].set_title('Success Rate vs. Obstacle Density', fontsize=14)
    axs[1, 1].legend(fontsize=12)
    axs[1, 1].grid(True, alpha=0.3)
    axs[1, 1].set_ylim(0, 105)
    
    plt.tight_layout()
    plt.savefig(f'{path_dir}benchmark_results/overall/performance_metrics.png', dpi=300)
    print("Overall performance metrics plotted and saved")
    
    # Create a figure to analyze JPS4 vs A* ratio across densities
    plt.figure(figsize=(16, 10))
    
    # Plot the ratio of JPS4/A* for key metrics
    ratios = {}
    ratios["time"] = [j/a if a > 0 and j > 0 else 0 for j, a in zip(avg_times["JPS4"], avg_times["AStar"])]
    ratios["nodes"] = [j/a if a > 0 and j > 0 else 0 for j, a in zip(avg_nodes["JPS4"], avg_nodes["AStar"])]
    ratios["path"] = [j/a if a > 0 and j > 0 else 0 for j, a in zip(avg_path_lengths["JPS4"], avg_path_lengths["AStar"])]
    
    plt.subplot(1, 1, 1)
    plt.plot(obstacle_densities, ratios["time"], marker='o', label='Time Ratio (JPS4/A*)', color='green', linewidth=2)
    plt.plot(obstacle_densities, ratios["nodes"], marker='s', label='Nodes Ratio (JPS4/A*)', color='purple', linewidth=2)
    plt.plot(obstacle_densities, ratios["path"], marker='^', label='Path Length Ratio (JPS4/A*)', color='orange', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.7)
    plt.xlabel('Obstacle Density', fontsize=14)
    plt.ylabel('JPS4/A* Ratio', fontsize=14)
    plt.title('JPS4 Performance Relative to A*', fontsize=18)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{path_dir}benchmark_results/overall/JPS4_performance_ratio.png', dpi=300)
    print("JPS4 performance ratio analysis saved")
    
    # Create category-specific comparisons (low, medium, high density)
    # Group densities into categories
    low_densities = [d for d in obstacle_densities if d < 0.3]
    medium_densities = [d for d in obstacle_densities if 0.3 <= d < 0.6]
    high_densities = [d for d in obstacle_densities if d >= 0.6]
    
    density_categories = {
        "low": low_densities,
        "medium": medium_densities,
        "high": high_densities
    }
    
    for category, densities in density_categories.items():
        if not densities:
            continue
            
        # Create figure for this category
        plt.figure(figsize=(16, 8))
        
        # Extract data for this category
        cat_times = {alg: [] for alg in results.keys()}
        cat_nodes = {alg: [] for alg in results.keys()}
        
        for alg in results.keys():
            for density in densities:
                idx = obstacle_densities.index(density)
                cat_times[alg].append(avg_times[alg][idx])
                cat_nodes[alg].append(avg_nodes[alg][idx])
        
        # Plot time and nodes side by side
        plt.subplot(1, 2, 1)
        for alg in results.keys():
            plt.plot(densities, cat_times[alg], marker=markers[alg], label=alg, 
                    color=colors[alg], linewidth=2, markersize=8)
        plt.xlabel('Obstacle Density', fontsize=12)
        plt.ylabel('Average Time (s)', fontsize=12)
        plt.title(f'Execution Time - {category.capitalize()} Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.subplot(1, 2, 2)
        for alg in results.keys():
            plt.plot(densities, cat_nodes[alg], marker=markers[alg], label=alg, 
                    color=colors[alg], linewidth=2, markersize=8)
        plt.xlabel('Obstacle Density', fontsize=12)
        plt.ylabel('Nodes Expanded', fontsize=12)
        plt.title(f'Search Space Size - {category.capitalize()} Density', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f'{path_dir}benchmark_results/overall/{category}_density_comparison.png', dpi=300)
        print(f"{category.capitalize()} density comparison saved")

def plot_density_results(results, density, filename):
    """
    Generate plots for a specific obstacle density
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Performance Metrics for Obstacle Density {density:.2f}', fontsize=18)
    
    # Extract data for this density
    JPS4_success = sum(results["JPS4"][density]["success"]) / len(results["JPS4"][density]["success"]) * 100
    astar_success = sum(results["AStar"][density]["success"]) / len(results["AStar"][density]["success"]) * 100
    
    # Only include successful trials
    JPS4_times = [t for t, s in zip(results["JPS4"][density]["time"], results["JPS4"][density]["success"]) if s]
    astar_times = [t for t, s in zip(results["AStar"][density]["time"], results["AStar"][density]["success"]) if s]
    
    JPS4_nodes = [n for n, s in zip(results["JPS4"][density]["nodes"], results["JPS4"][density]["success"]) if s]
    astar_nodes = [n for n, s in zip(results["AStar"][density]["nodes"], results["JPS4"][density]["success"]) if s]
    
    JPS4_paths = [p for p, s in zip(results["JPS4"][density]["path_length"], results["JPS4"][density]["success"]) if s and p != float('inf')]
    astar_paths = [p for p, s in zip(results["AStar"][density]["path_length"], results["AStar"][density]["success"]) if s and p != float('inf')]
    
    # Success rate bar chart
    axs[0, 0].bar(['JPS4', 'A*'], [JPS4_success, astar_success], color=['blue', 'red'])
    axs[0, 0].set_ylabel('Success Rate (%)', fontsize=12)
    axs[0, 0].set_title('Algorithm Success Rate', fontsize=14)
    axs[0, 0].set_ylim(0, 105)
    for i, v in enumerate([JPS4_success, astar_success]):
        axs[0, 0].text(i, v + 3, f"{v:.1f}%", ha='center', fontsize=10)
    
    # Execution time box plot
    if JPS4_times and astar_times:
        axs[0, 1].boxplot([JPS4_times, astar_times], labels=['JPS4', 'A*'], 
                         patch_artist=True, boxprops=dict(facecolor='lightblue'))
        axs[0, 1].set_ylabel('Time (seconds)', fontsize=12)
        axs[0, 1].set_title('Execution Time Comparison', fontsize=14)
    else:
        axs[0, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=14)
        axs[0, 1].set_title('Execution Time Comparison', fontsize=14)
    
    # Nodes expanded box plot
    if JPS4_nodes and astar_nodes:
        axs[1, 0].boxplot([JPS4_nodes, astar_nodes], labels=['JPS4', 'A*'],
                        patch_artist=True, boxprops=dict(facecolor='lightgreen'))
        axs[1, 0].set_ylabel('Nodes Expanded', fontsize=12)
        axs[1, 0].set_title('Search Space Size Comparison', fontsize=14)
    else:
        axs[1, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=14)
        axs[1, 0].set_title('Search Space Size Comparison', fontsize=14)
    
    # Path length box plot
    if JPS4_paths and astar_paths:
        axs[1, 1].boxplot([JPS4_paths, astar_paths], labels=['JPS4', 'A*'],
                        patch_artist=True, boxprops=dict(facecolor='lightyellow'))
        axs[1, 1].set_ylabel('Path Length', fontsize=12)
        axs[1, 1].set_title('Path Length Comparison', fontsize=14)
    else:
        axs[1, 1].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', fontsize=14)
        axs[1, 1].set_title('Path Length Comparison', fontsize=14)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    plt.close()
    print(f"Density-specific plots generated: {filename}")

def plot_scaling_results(results, grid_sizes):
    """
    Generate plots showing how JPS4 and A* scale with grid size
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(16, 14))
    fig.suptitle('JPS4 vs A* Scaling Properties', fontsize=18)
    
    # Prepare data
    JPS4_times = []
    astar_times = []
    JPS4_nodes = []
    astar_nodes = []
    JPS4_success = []
    astar_success = []
    
    for size in grid_sizes:
        # Calculate success rates
        JPS4_success_rate = sum(results["JPS4"][size]["success"]) / len(results["JPS4"][size]["success"]) * 100
        astar_success_rate = sum(results["AStar"][size]["success"]) / len(results["AStar"][size]["success"]) * 100
        
        JPS4_success.append(JPS4_success_rate)
        astar_success.append(astar_success_rate)
        
        # Calculate averages for successful trials
        JPS4_time_avg = sum([t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s]) / sum(results["JPS4"][size]["success"]) if sum(results["JPS4"][size]["success"]) > 0 else 0
        astar_time_avg = sum([t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s]) / sum(results["AStar"][size]["success"]) if sum(results["AStar"][size]["success"]) > 0 else 0
        
        JPS4_nodes_avg = sum([n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s]) / sum(results["JPS4"][size]["success"]) if sum(results["JPS4"][size]["success"]) > 0 else 0
        astar_nodes_avg = sum([n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s]) / sum(results["AStar"][size]["success"]) if sum(results["AStar"][size]["success"]) > 0 else 0
        
        JPS4_times.append(JPS4_time_avg)
        astar_times.append(astar_time_avg)
        JPS4_nodes.append(JPS4_nodes_avg)
        astar_nodes.append(astar_nodes_avg)
    
    # Plot 1: Execution Time vs Grid Size
    axs[0, 0].plot(grid_sizes, JPS4_times, 'o-', label='JPS4')
    axs[0, 0].plot(grid_sizes, astar_times, 's-', label='A*')
    axs[0, 0].set_xlabel('Grid Size')
    axs[0, 0].set_ylabel('Execution Time (s)')
    axs[0, 0].set_title('Execution Time vs Grid Size')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot 2: Nodes Expanded vs Grid Size
    axs[0, 1].plot(grid_sizes, JPS4_nodes, 'o-', label='JPS4')
    axs[0, 1].plot(grid_sizes, astar_nodes, 's-', label='A*')
    axs[0, 1].set_xlabel('Grid Size')
    axs[0, 1].set_ylabel('Nodes Expanded')
    axs[0, 1].set_title('Nodes Expanded vs Grid Size')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot 3: Success Rate vs Grid Size
    axs[1, 0].plot(grid_sizes, JPS4_success, 'o-', label='JPS4')
    axs[1, 0].plot(grid_sizes, astar_success, 's-', label='A*')
    axs[1, 0].set_xlabel('Grid Size')
    axs[1, 0].set_ylabel('Success Rate (%)')
    axs[1, 0].set_title('Success Rate vs Grid Size')
    axs[1, 0].legend()
    axs[1, 0].grid(True)
    
    # Plot 4: Performance Ratio (JPS4/A*) vs Grid Size
    time_ratios = [JPS4_time/astar_time if astar_time > 0 else 0 for JPS4_time, astar_time in zip(JPS4_times, astar_times)]
    node_ratios = [JPS4_node/astar_node if astar_node > 0 else 0 for JPS4_node, astar_node in zip(JPS4_nodes, astar_nodes)]
    
    axs[1, 1].plot(grid_sizes, time_ratios, 'o-', label='Time Ratio')
    axs[1, 1].plot(grid_sizes, node_ratios, 's-', label='Node Expansion Ratio')
    axs[1, 1].axhline(y=1.0, color='r', linestyle='--', label='Equal Performance')
    axs[1, 1].set_xlabel('Grid Size')
    axs[1, 1].set_ylabel('JPS4/A* Ratio')
    axs[1, 1].set_title('Performance Ratio (JPS4/A*) vs Grid Size')
    axs[1, 1].legend()
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(f"{path_dir}benchmark_results/large_grid/scaling_plots.png", dpi=150)
    plt.close()
    
    print(f"Scaling plots generated: {path_dir}benchmark_results/large_grid/scaling_plots.png")

def plot_maze_results(results, grid_sizes):
    """
    Generate plots for maze benchmark results
    """
    # Create figure with subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle('JPS4 vs A* Performance in Maze-like Environments', fontsize=16)
    
    # Prepare data
    JPS4_times = []
    astar_times = []
    JPS4_nodes = []
    astar_nodes = []
    time_ratios = []
    node_ratios = []
    
    for size in grid_sizes:
        # Calculate averages for successful trials
        JPS4_time_avg = sum([t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s]) / sum(results["JPS4"][size]["success"]) if sum(results["JPS4"][size]["success"]) > 0 else 0
        astar_time_avg = sum([t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s]) / sum(results["AStar"][size]["success"]) if sum(results["AStar"][size]["success"]) > 0 else 0
        
        JPS4_nodes_avg = sum([n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s]) / sum(results["JPS4"][size]["success"]) if sum(results["JPS4"][size]["success"]) > 0 else 0
        astar_nodes_avg = sum([n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s]) / sum(results["AStar"][size]["success"]) if sum(results["AStar"][size]["success"]) > 0 else 0
        
        JPS4_times.append(JPS4_time_avg)
        astar_times.append(astar_time_avg)
        JPS4_nodes.append(JPS4_nodes_avg)
        astar_nodes.append(astar_nodes_avg)
        
        # Calculate ratios
        time_ratio = JPS4_time_avg / astar_time_avg if astar_time_avg > 0 else 0
        node_ratio = JPS4_nodes_avg / astar_nodes_avg if astar_nodes_avg > 0 else 0
        
        time_ratios.append(time_ratio)
        node_ratios.append(node_ratio)
    
    # Plot execution time
    axs[0, 0].plot(grid_sizes, JPS4_times, 'o-', label='JPS4')
    axs[0, 0].plot(grid_sizes, astar_times, 's-', label='A*')
    axs[0, 0].set_xlabel('Grid Size')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].set_title('Execution Time in Maze Environments')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot nodes expanded
    axs[0, 1].plot(grid_sizes, JPS4_nodes, 'o-', label='JPS4')
    axs[0, 1].plot(grid_sizes, astar_nodes, 's-', label='A*')
    axs[0, 1].set_xlabel('Grid Size')
    axs[0, 1].set_ylabel('Nodes Expanded')
    axs[0, 1].set_title('Search Space Size in Maze Environments')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot time ratio
    axs[1, 0].plot(grid_sizes, time_ratios, 'o-', color='purple')
    axs[1, 0].axhline(y=1.0, color='gray', linestyle='--')
    axs[1, 0].set_xlabel('Grid Size')
    axs[1, 0].set_ylabel('JPS4/A* Time Ratio')
    axs[1, 0].set_title('Time Efficiency Ratio in Maze Environments')
    axs[1, 0].grid(True)
    
    # Plot node ratio
    axs[1, 1].plot(grid_sizes, node_ratios, 'o-', color='green')
    axs[1, 1].axhline(y=1.0, color='gray', linestyle='--')
    axs[1, 1].set_xlabel('Grid Size')
    axs[1, 1].set_ylabel('JPS4/A* Node Expansion Ratio')
    axs[1, 1].set_title('Search Efficiency Ratio in Maze Environments')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{path_dir}benchmark_results/maze/maze_performance.png', dpi=300)
    plt.close()
    
    print(f"Maze benchmark plots generated: {path_dir}benchmark_results/maze/maze_performance.png")
