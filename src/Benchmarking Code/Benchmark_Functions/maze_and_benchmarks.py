from Algorithms.a_star import Point, AStar
from Algorithms.jump_point_search import JPS4
from Algorithms.helper import path_dir

from Benchmark_Functions.grid_and_benchmarks import validate_grid_connectivity, run_benchmark, generate_grid

from Helper_Functions.report_helper_functions import generate_maze_report
from Helper_Functions.plotting_helper_functions import visualize_grid, plot_maze_results

import time
import random
import os
import matplotlib.pyplot as plt


def generate_maze_grid(width: int, height: int, seed: int = None):
    """
    Generate a maze-like grid with long corridors that highlight JPS4's advantages
    """
    if seed is not None:
        random.seed(seed)
    
    # Start with all cells blocked
    grid = [[True for _ in range(width)] for _ in range(height)]
    
    # Define margins for start and goal
    margin = 2
    
    # Place start and goal points far apart
    start = Point(random.randint(margin, width//4), random.randint(margin, height//4))
    goal = Point(random.randint(width*3//4, width-margin), random.randint(height*3//4, height-margin))
    
    # Clear start and goal cells
    grid[start.y][start.x] = False
    grid[goal.y][goal.x] = False
    
    # Maze generation using a simple recursive division method
    def carve_passages(x1, y1, x2, y2, pattern="horizontal"):
        # Base case: if the region is too small, stop
        if x2 - x1 < 3 or y2 - y1 < 3:
            return
        
        # Choose where to divide
        if pattern == "horizontal":
            # Divide horizontally
            divide_y = random.randint(y1 + 1, y2 - 1)
            passage_x = random.randint(x1, x2)
            
            # Carve horizontal passage
            for x in range(x1, x2 + 1):
                grid[divide_y][x] = True  # Set wall
            
            # Carve passage through the wall
            grid[divide_y][passage_x] = False
            
            # Recursively process sub-regions
            carve_passages(x1, y1, x2, divide_y - 1, "vertical")
            carve_passages(x1, divide_y + 1, x2, y2, "vertical")
            
        else:  # pattern == "vertical"
            # Divide vertically
            divide_x = random.randint(x1 + 1, x2 - 1)
            passage_y = random.randint(y1, y2)
            
            # Carve vertical passage
            for y in range(y1, y2 + 1):
                grid[y][divide_x] = True  # Set wall
            
            # Carve passage through the wall
            grid[passage_y][divide_x] = False
            
            # Recursively process sub-regions
            carve_passages(x1, y1, divide_x - 1, y2, "horizontal")
            carve_passages(divide_x + 1, y1, x2, y2, "horizontal")
    
    # Initialize maze with passages
    # First clear a border around the edge
    for y in range(height):
        for x in range(width):
            # Make the edges walls
            if x == 0 or y == 0 or x == width - 1 or y == height - 1:
                grid[y][x] = True
            else:
                # Start with all inner cells as passages
                grid[y][x] = False
    
    # Then carve maze passages starting with a horizontal division
    carve_passages(1, 1, width - 2, height - 2, "horizontal")
    
    # Ensure start and goal are connected
    # Create a direct path from start to goal
    path_points = []
    steps = max(abs(goal.x - start.x), abs(goal.y - start.y)) * 2
    for i in range(steps):
        t = i / (steps - 1)
        x = int(start.x + t * (goal.x - start.x))
        y = int(start.y + t * (goal.y - start.y))
        path_points.append((x, y))
    
    # Randomly drop some points to make the path less direct (more maze-like)
    path_points = [(start.x, start.y)] + random.sample(path_points[1:-1], len(path_points) // 3) + [(goal.x, goal.y)]
    
    # Clear and ensure path
    for i in range(len(path_points) - 1):
        x1, y1 = path_points[i]
        x2, y2 = path_points[i + 1]
        
        # Create L-shaped paths between consecutive points
        # First horizontal
        for x in range(min(x1, x2), max(x1, x2) + 1):
            grid[y1][x] = False  # Clear cell
        
        # Then vertical
        for y in range(min(y1, y2), max(y1, y2) + 1):
            grid[y][x2] = False  # Clear cell
    
    # Make sure start and goal are clear
    grid[start.y][start.x] = False
    grid[goal.y][goal.x] = False
    
    return grid, start, goal

def run_maze_benchmark():
    """
    Run benchmark on maze-like environments to highlight JPS4's advantages with long corridors
    """
    # Define grid sizes to test
    grid_sizes = [50, 100, 200, 300]
    
    # Multiple trials for statistical significance
    trials_per_size = 5
    
    # Create directory for results
    os.makedirs(f"{path_dir}benchmark_results/maze", exist_ok=True)
    
    # Store results
    results = {
        "JPS4": {size: {"time": [], "nodes": [], "success": [], "path_length": []} for size in grid_sizes},
        "AStar": {size: {"time": [], "nodes": [], "success": [], "path_length": []} for size in grid_sizes}
    }
    
    for size in grid_sizes:
        print(f"\nRunning maze benchmark with size {size}x{size}")
        
        for trial in range(trials_per_size):
            print(f"  Trial {trial+1}/{trials_per_size}")
            
            # Generate maze grid
            grid_data, start, goal = generate_maze_grid(size, size, seed=trial*10)
            
            # Validate grid connectivity
            has_path = validate_grid_connectivity(grid_data, start, goal, size, size)
            if not has_path:
                print(f"    WARNING: Generated maze does not have a valid path - regenerating")
                grid_data, start, goal = generate_maze_grid(size, size, seed=trial*10+100)
                has_path = validate_grid_connectivity(grid_data, start, goal, size, size)
                if not has_path:
                    print(f"    ERROR: Failed to generate a maze with a valid path")
                    continue
            
            # Save grid visualization for the first trial
            if trial == 0:
                filepath = f"{path_dir}benchmark_results/maze/grid_size_{size}.png"
                visualize_grid(grid_data, start, goal, filepath)
            
            # Test JPS4
            JPS4_grid = JPS4(size, size)
            for y in range(size):
                for x in range(size):
                    if grid_data[y][x]:
                        JPS4_grid.set(x, y, True)
            
            nodes_expanded = [0]
            def count_nodes_JPS4(node):
                nodes_expanded[0] += 1
            
            try:
                # Set timeout for larger grids
                timeout = min(60, size / 50)
                start_time = time.time()
                path_JPS4 = JPS4_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_JPS4)
                elapsed = time.time() - start_time
                
                success = path_JPS4 is not None
                if not success and has_path:
                    print(f"    WARNING: JPS4 failed to find a path even though one exists")
                
                results["JPS4"][size]["time"].append(elapsed)
                results["JPS4"][size]["nodes"].append(nodes_expanded[0])
                results["JPS4"][size]["success"].append(success)
                results["JPS4"][size]["path_length"].append(len(path_JPS4) if path_JPS4 else 0)
                
                if elapsed > timeout:
                    print(f"    NOTE: JPS4 took {elapsed:.2f}s for {size}x{size} grid")
            except Exception as e:
                print(f"    ERROR: JPS4 failed with exception: {str(e)}")
                results["JPS4"][size]["time"].append(float('inf'))
                results["JPS4"][size]["nodes"].append(0)
                results["JPS4"][size]["success"].append(False)
                results["JPS4"][size]["path_length"].append(0)
            
            # Test A*
            astar_grid = AStar(size, size)
            for y in range(size):
                for x in range(size):
                    if grid_data[y][x]:
                        astar_grid.set(x, y, True)
            
            nodes_expanded_astar = [0]
            def count_nodes_astar(node):
                nodes_expanded_astar[0] += 1
            
            try:
                # Set timeout for larger grids
                timeout = min(60, size / 50)
                start_time = time.time()
                path_astar = astar_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_astar)
                elapsed = time.time() - start_time
                
                success = path_astar is not None
                if not success and has_path:
                    print(f"    WARNING: A* failed to find a path even though one exists")
                
                results["AStar"][size]["time"].append(elapsed)
                results["AStar"][size]["nodes"].append(nodes_expanded_astar[0])
                results["AStar"][size]["success"].append(success)
                results["AStar"][size]["path_length"].append(len(path_astar) if path_astar else 0)
                
                if elapsed > timeout:
                    print(f"    NOTE: A* took {elapsed:.2f}s for {size}x{size} grid")
            except Exception as e:
                print(f"    ERROR: A* failed with exception: {str(e)}")
                results["AStar"][size]["time"].append(float('inf'))
                results["AStar"][size]["nodes"].append(0)
                results["AStar"][size]["success"].append(False)
                results["AStar"][size]["path_length"].append(0)
    
    # Generate maze benchmark report
    generate_maze_report(results, grid_sizes)
    plot_maze_results(results, grid_sizes)
    
    return results

def run_tiny_benchmark():
    """
    Run a minimal benchmark for very quick testing
    """
    width = 25
    height = 25
    obstacle_densities = [0.2, 0.5]
    trials_per_density = 2
    
    print(f"Running tiny benchmark with {width}x{height} grid...")
    run_benchmark(width, height, obstacle_densities, trials_per_density, save_grids=True, group_by_density=True)
    print("Tiny benchmark complete!")

def run_medium_large_benchmark():
    """
    Run benchmarks with moderately large grid sizes to assess scaling properties faster
    """
    # Define grid sizes to test
    grid_sizes = [50, 100, 150, 200]
    
    # Use a moderate obstacle density
    obstacle_density = 0.4
    
    # Multiple trials for statistical significance
    trials_per_size = 3
    
    # Create directory for results
    os.makedirs(f"{path_dir}benchmark_results/medium_large_grid", exist_ok=True)
    
    # Store results
    results = {
        "JPS4": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes},
        "AStar": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes}
    }
    
    for size in grid_sizes:
        print(f"\nRunning medium-large grid benchmark with size {size}x{size}")
        
        for trial in range(trials_per_size):
            print(f"  Trial {trial+1}/{trials_per_size}")
            
            # Generate grid with a valid path
            max_attempts = 5
            for attempt in range(max_attempts):
                grid_data, start, goal = generate_grid(size, size, obstacle_density, seed=trial*10 + attempt)
                
                # Validate grid connectivity
                has_path = validate_grid_connectivity(grid_data, start, goal, size, size)
                if has_path:
                    if attempt > 0:
                        print(f"    Found valid path after {attempt+1} attempts")
                    break
                elif attempt == max_attempts - 1:
                    print(f"    WARNING: Failed to generate a grid with a valid path after {max_attempts} attempts")
            
            # Save grid visualization for the first trial
            if trial == 0:
                filepath = f"{path_dir}benchmark_results/medium_large_grid/grid_size_{size}.png"
                visualize_grid(grid_data, start, goal, filepath)
            
            # Test JPS4
            JPS4_grid = JPS4(size, size)
            for y in range(size):
                for x in range(size):
                    if grid_data[y][x]:
                        JPS4_grid.set(x, y, True)
            
            nodes_expanded = [0]
            def count_nodes_JPS4(node):
                nodes_expanded[0] += 1
            
            try:
                # Set timeout for very large grids
                timeout = min(30, size / 50)  # Scale timeout with grid size
                start_time = time.time()
                path = JPS4_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_JPS4)
                elapsed = time.time() - start_time
                
                success = path is not None
                if not success and has_path:
                    print(f"    WARNING: JPS4 failed to find a path even though one exists")
                    
                results["JPS4"][size]["time"].append(elapsed)
                results["JPS4"][size]["nodes"].append(nodes_expanded[0])
                results["JPS4"][size]["success"].append(success)
                
                if elapsed > timeout:
                    print(f"    NOTE: JPS4 took {elapsed:.2f}s for {size}x{size} grid")
            except Exception as e:
                print(f"    ERROR: JPS4 failed with exception: {str(e)}")
                results["JPS4"][size]["time"].append(float('inf'))
                results["JPS4"][size]["nodes"].append(0)
                results["JPS4"][size]["success"].append(False)
            
            # Test A*
            astar_grid = AStar(size, size)
            for y in range(size):
                for x in range(size):
                    if grid_data[y][x]:
                        astar_grid.set(x, y, True)
            
            nodes_expanded_astar = [0]
            def count_nodes_astar(node):
                nodes_expanded_astar[0] += 1
            
            try:
                # Set timeout for very large grids
                timeout = min(30, size / 50)  # Scale timeout with grid size
                start_time = time.time()
                path = astar_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_astar)
                elapsed = time.time() - start_time
                
                success = path is not None
                if not success and has_path:
                    print(f"    WARNING: A* failed to find a path even though one exists")
                    
                results["AStar"][size]["time"].append(elapsed)
                results["AStar"][size]["nodes"].append(nodes_expanded_astar[0])
                results["AStar"][size]["success"].append(success)
                
                if elapsed > timeout:
                    print(f"    NOTE: A* took {elapsed:.2f}s for {size}x{size} grid")
            except Exception as e:
                print(f"    ERROR: A* failed with exception: {str(e)}")
                results["AStar"][size]["time"].append(float('inf'))
                results["AStar"][size]["nodes"].append(0)
                results["AStar"][size]["success"].append(False)
    
    # Generate scaling report and plots
    with open(f"{path_dir}benchmark_results/medium_large_grid/scaling_report.txt", "w") as f:
        f.write("JPS4 vs A* Scaling Performance Report\n")
        f.write("====================================\n\n")
        
        f.write("Grid Size Performance Metrics\n")
        f.write("------------------------------\n\n")
        
        for size in grid_sizes:
            f.write(f"Grid Size: {size}x{size}\n")
            f.write("-" * 20 + "\n")
            
            # Calculate success rates
            JPS4_success = sum(results["JPS4"][size]["success"]) / len(results["JPS4"][size]["success"]) * 100
            astar_success = sum(results["AStar"][size]["success"]) / len(results["AStar"][size]["success"]) * 100
            
            f.write(f"Success Rate: JPS4 {JPS4_success:.1f}%, A* {astar_success:.1f}%\n")
            
            # Average times for successful trials
            JPS4_times = [t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s]
            astar_times = [t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s]
            
            # Average nodes for successful trials
            JPS4_nodes = [n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s]
            astar_nodes = [n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s]
            
            if JPS4_times and astar_times:
                avg_JPS4_time = sum(JPS4_times) / len(JPS4_times)
                avg_astar_time = sum(astar_times) / len(astar_times)
                time_ratio = avg_JPS4_time / avg_astar_time if avg_astar_time > 0 else 0
                
                f.write(f"Avg Time (JPS4): {avg_JPS4_time:.6f}s\n")
                f.write(f"Avg Time (A*): {avg_astar_time:.6f}s\n")
                f.write(f"Time Ratio (JPS4/A*): {time_ratio:.2f}x\n")
            
            if JPS4_nodes and astar_nodes:
                avg_JPS4_nodes = sum(JPS4_nodes) / len(JPS4_nodes)
                avg_astar_nodes = sum(astar_nodes) / len(astar_nodes)
                node_ratio = avg_JPS4_nodes / avg_astar_nodes if avg_astar_nodes > 0 else 0
                
                f.write(f"Avg Nodes Expanded (JPS4): {avg_JPS4_nodes:.1f}\n")
                f.write(f"Avg Nodes Expanded (A*): {avg_astar_nodes:.1f}\n")
                f.write(f"Node Expansion Ratio (JPS4/A*): {node_ratio:.2f}x\n\n")
    
    print(f"Scaling report generated: {path_dir}benchmark_results/medium_large_grid/scaling_report.txt")
    
    # Create plots
    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('JPS4 vs A* Scaling Performance', fontsize=16)
    
    # Extract data for plotting
    sizes = grid_sizes
    
    JPS4_times_avg = []
    astar_times_avg = []
    JPS4_nodes_avg = []
    astar_nodes_avg = []
    
    for size in sizes:
        # Times
        JPS4_time = sum([t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s]) / sum(results["JPS4"][size]["success"]) if sum(results["JPS4"][size]["success"]) > 0 else 0
        astar_time = sum([t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s]) / sum(results["AStar"][size]["success"]) if sum(results["AStar"][size]["success"]) > 0 else 0
        
        # Nodes
        JPS4_nodes = sum([n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s]) / sum(results["JPS4"][size]["success"]) if sum(results["JPS4"][size]["success"]) > 0 else 0
        astar_nodes = sum([n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s]) / sum(results["AStar"][size]["success"]) if sum(results["AStar"][size]["success"]) > 0 else 0
        
        JPS4_times_avg.append(JPS4_time)
        astar_times_avg.append(astar_time)
        JPS4_nodes_avg.append(JPS4_nodes)
        astar_nodes_avg.append(astar_nodes)
    
    # Plot execution time
    axs[0, 0].plot(sizes, JPS4_times_avg, 'o-', label='JPS4')
    axs[0, 0].plot(sizes, astar_times_avg, 's-', label='A*')
    axs[0, 0].set_xlabel('Grid Size (NxN)')
    axs[0, 0].set_ylabel('Time (seconds)')
    axs[0, 0].set_title('Execution Time')
    axs[0, 0].legend()
    axs[0, 0].grid(True)
    
    # Plot nodes expanded
    axs[0, 1].plot(sizes, JPS4_nodes_avg, 'o-', label='JPS4')
    axs[0, 1].plot(sizes, astar_nodes_avg, 's-', label='A*')
    axs[0, 1].set_xlabel('Grid Size (NxN)')
    axs[0, 1].set_ylabel('Nodes Expanded')
    axs[0, 1].set_title('Search Space Size')
    axs[0, 1].legend()
    axs[0, 1].grid(True)
    
    # Plot time ratio
    time_ratios = []
    for j, a in zip(JPS4_times_avg, astar_times_avg):
        ratio = j / a if a > 0 else 0
        time_ratios.append(ratio)
    
    axs[1, 0].plot(sizes, time_ratios, 'o-', color='purple')
    axs[1, 0].axhline(y=1.0, color='gray', linestyle='--')
    axs[1, 0].set_xlabel('Grid Size (NxN)')
    axs[1, 0].set_ylabel('JPS4/A* Time Ratio')
    axs[1, 0].set_title('Time Efficiency Ratio')
    axs[1, 0].grid(True)
    
    # Plot node ratio
    node_ratios = []
    for j, a in zip(JPS4_nodes_avg, astar_nodes_avg):
        ratio = j / a if a > 0 else 0
        node_ratios.append(ratio)
    
    axs[1, 1].plot(sizes, node_ratios, 'o-', color='green')
    axs[1, 1].axhline(y=1.0, color='gray', linestyle='--')
    axs[1, 1].set_xlabel('Grid Size (NxN)')
    axs[1, 1].set_ylabel('JPS4/A* Node Expansion Ratio')
    axs[1, 1].set_title('Search Efficiency Ratio')
    axs[1, 1].grid(True)
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(f'{path_dir}benchmark_results/medium_large_grid/scaling_plots.png')
    plt.close()
    
    print(f"Scaling plots generated: {path_dir}benchmark_results/medium_large_grid/scaling_plots.png")
    
    return results

def run_tiny_scaling_benchmark():
    """
    Run a minimal scaling benchmark for very quick testing
    """
    # Define grid sizes to test
    grid_sizes = [20, 40, 60]
    
    # Use a moderate obstacle density
    obstacle_density = 0.4
    
    # Multiple trials for statistical significance
    trials_per_size = 2
    
    # Create directory for results
    os.makedirs(f"{path_dir}benchmark_results/tiny_scaling", exist_ok=True)
    
    # Store results
    results = {
        "JPS4": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes},
        "AStar": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes}
    }
    
    for size in grid_sizes:
        print(f"\nRunning tiny scaling benchmark with size {size}x{size}")
        
        for trial in range(trials_per_size):
            print(f"  Trial {trial+1}/{trials_per_size}")
            
            # Generate grid with a valid path
            max_attempts = 5
            for attempt in range(max_attempts):
                grid_data, start, goal = generate_grid(size, size, obstacle_density, seed=trial*10 + attempt)
                
                # Validate grid connectivity
                has_path = validate_grid_connectivity(grid_data, start, goal, size, size)
                if has_path:
                    if attempt > 0:
                        print(f"    Found valid path after {attempt+1} attempts")
                    break
                elif attempt == max_attempts - 1:
                    print(f"    WARNING: Failed to generate a grid with a valid path after {max_attempts} attempts")
            
            # Test JPS4
            JPS4_grid = JPS4(size, size)
            for y in range(size):
                for x in range(size):
                    if grid_data[y][x]:
                        JPS4_grid.set(x, y, True)
            
            nodes_expanded = [0]
            def count_nodes_JPS4(node):
                nodes_expanded[0] += 1
            
            try:
                start_time = time.time()
                path = JPS4_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_JPS4)
                elapsed = time.time() - start_time
                
                success = path is not None
                if not success and has_path:
                    print(f"    WARNING: JPS4 failed to find a path even though one exists")
                    
                results["JPS4"][size]["time"].append(elapsed)
                results["JPS4"][size]["nodes"].append(nodes_expanded[0])
                results["JPS4"][size]["success"].append(success)
            except Exception as e:
                print(f"    ERROR: JPS4 failed with exception: {str(e)}")
                results["JPS4"][size]["time"].append(float('inf'))
                results["JPS4"][size]["nodes"].append(0)
                results["JPS4"][size]["success"].append(False)
            
            # Test A*
            astar_grid = AStar(size, size)
            for y in range(size):
                for x in range(size):
                    if grid_data[y][x]:
                        astar_grid.set(x, y, True)
            
            nodes_expanded_astar = [0]
            def count_nodes_astar(node):
                nodes_expanded_astar[0] += 1
            
            try:
                start_time = time.time()
                path = astar_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_astar)
                elapsed = time.time() - start_time
                
                success = path is not None
                if not success and has_path:
                    print(f"    WARNING: A* failed to find a path even though one exists")
                    
                results["AStar"][size]["time"].append(elapsed)
                results["AStar"][size]["nodes"].append(nodes_expanded_astar[0])
                results["AStar"][size]["success"].append(success)
            except Exception as e:
                print(f"    ERROR: A* failed with exception: {str(e)}")
                results["AStar"][size]["time"].append(float('inf'))
                results["AStar"][size]["nodes"].append(0)
                results["AStar"][size]["success"].append(False)
    
    # Generate scaling report
    with open(f"{path_dir}benchmark_results/tiny_scaling/scaling_report.txt", "w") as f:
        f.write("JPS4 vs A* Tiny Scaling Report\n")
        f.write("============================\n\n")
        
        for size in grid_sizes:
            f.write(f"Grid Size: {size}x{size}\n")
            f.write("-" * 20 + "\n")
            
            # Calculate success rates
            JPS4_success = sum(results["JPS4"][size]["success"]) / len(results["JPS4"][size]["success"]) * 100
            astar_success = sum(results["AStar"][size]["success"]) / len(results["AStar"][size]["success"]) * 100
            
            f.write(f"Success Rate: JPS4 {JPS4_success:.1f}%, A* {astar_success:.1f}%\n")
            
            # Average times for successful trials
            JPS4_times = [t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s]
            astar_times = [t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s]
            
            # Average nodes for successful trials
            JPS4_nodes = [n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s]
            astar_nodes = [n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s]
            
            if JPS4_times and astar_times:
                avg_JPS4_time = sum(JPS4_times) / len(JPS4_times)
                avg_astar_time = sum(astar_times) / len(astar_times)
                time_ratio = avg_JPS4_time / avg_astar_time if avg_astar_time > 0 else 0
                
                f.write(f"Avg Time (JPS4): {avg_JPS4_time:.6f}s\n")
                f.write(f"Avg Time (A*): {avg_astar_time:.6f}s\n")
                f.write(f"Time Ratio (JPS4/A*): {time_ratio:.2f}x\n")
            
            if JPS4_nodes and astar_nodes:
                avg_JPS4_nodes = sum(JPS4_nodes) / len(JPS4_nodes)
                avg_astar_nodes = sum(astar_nodes) / len(astar_nodes)
                node_ratio = avg_JPS4_nodes / avg_astar_nodes if avg_astar_nodes > 0 else 0
                
                f.write(f"Avg Nodes Expanded (JPS4): {avg_JPS4_nodes:.1f}\n")
                f.write(f"Avg Nodes Expanded (A*): {avg_astar_nodes:.1f}\n")
                f.write(f"Node Expansion Ratio (JPS4/A*): {node_ratio:.2f}x\n\n")
    
    print(f"Scaling report generated: {path_dir}benchmark_results/tiny_scaling/scaling_report.txt")
    
    return results
