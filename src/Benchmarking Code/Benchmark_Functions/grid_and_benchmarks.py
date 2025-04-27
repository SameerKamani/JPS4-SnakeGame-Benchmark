from Algorithms.a_star import List, Point, AStar
from Algorithms.jump_point_search import JPS4
from Algorithms.helper import path_dir

from Helper_Functions.report_helper_functions import generate_report, generate_density_report, generate_scaling_report
from Helper_Functions.plotting_helper_functions import visualize_grid, plot_results, plot_density_results, plot_scaling_results

import time
import random
import os

def generate_grid(width: int, height: int, obstacle_density: float, seed: int = None):
    """
    Generate a random grid with obstacles and valid start/goal points
    """
    if seed is not None:
        random.seed(seed)
    
    # Generate grid with random obstacles
    grid = [[False for _ in range(width)] for _ in range(height)]
    for y in range(height):
        for x in range(width):
            if random.random() < obstacle_density:
                grid[y][x] = True
    
    # Ensure start and goal are far apart and not blocked
    margin = 2
    start = Point(random.randint(margin, width//3), random.randint(margin, height//3))
    goal = Point(random.randint(width*2//3, width-margin), random.randint(height*2//3, height-margin))
    
    # Make sure start and goal are not blocked
    if 0 <= start.x < width and 0 <= start.y < height:
        grid[start.y][start.x] = False
    if 0 <= goal.x < width and 0 <= goal.y < height:
        grid[goal.y][goal.x] = False
    
    # For high obstacle densities (>= 0.5), ensure a path exists by carving a corridor
    # For lower densities, we'll rely on the natural connectivity of the grid
    if obstacle_density >= 0.5:
        # First, carve a horizontal corridor from start to goal's x-coordinate
        x1, y1 = start.x, start.y
        x2, y2 = goal.x, start.y
        # Horizontal segment (with some randomness)
        current_y = y1
        for x in range(min(x1, x2), max(x1, x2) + 1):
            # Add some randomness to the path to make it more natural
            if random.random() < 0.2 and 0 <= current_y + 1 < height:
                current_y += 1
            elif random.random() < 0.2 and 0 <= current_y - 1 < height:
                current_y -= 1
                
            # Clear a 3-cell wide corridor for better connectivity
            for dy in [-1, 0, 1]:
                corridor_y = current_y + dy
                if 0 <= corridor_y < height and 0 <= x < width:
                    grid[corridor_y][x] = False
        
        # Then, carve a vertical corridor from the end of horizontal corridor to goal
        x1, y1 = x2, current_y
        x2, y2 = goal.x, goal.y
        # Vertical segment
        current_x = x1
        for y in range(min(y1, y2), max(y1, y2) + 1):
            # Add some randomness to the path
            if random.random() < 0.2 and 0 <= current_x + 1 < width:
                current_x += 1
            elif random.random() < 0.2 and 0 <= current_x - 1 < width:
                current_x -= 1
                
            # Clear a 3-cell wide corridor for better connectivity
            for dx in [-1, 0, 1]:
                corridor_x = current_x + dx
                if 0 <= corridor_x < width and 0 <= y < height:
                    grid[y][corridor_x] = False
    elif obstacle_density == 0:
        # For zero density, clear a direct path
        x1, y1 = start.x, start.y
        x2, y2 = goal.x, goal.y
        for i in range(100):  # Limit iterations
            t = i / 100
            x = int(x1 + t * (x2 - x1))
            y = int(y1 + t * (y2 - y1))
            if 0 <= x < width and 0 <= y < height:
                grid[y][x] = False
    else:
        # For mid-range densities, create a simpler L-shaped path
        # Horizontal segment
        for x in range(min(start.x, goal.x), max(start.x, goal.x) + 1):
            if 0 <= x < width and 0 <= start.y < height:
                grid[start.y][x] = False
        
        # Vertical segment
        for y in range(min(start.y, goal.y), max(start.y, goal.y) + 1):
            if 0 <= goal.x < width and 0 <= y < height:
                grid[y][goal.x] = False
    
    return grid, start, goal

def validate_grid_connectivity(grid_data, start, goal, width, height):
    """
    Validate that a path exists between start and goal using a simple BFS search.
    Returns True if a path exists, False otherwise.
    """
    if not (0 <= start.x < width and 0 <= start.y < height and 
            0 <= goal.x < width and 0 <= goal.y < height):
        return False
        
    # Check if start or goal is blocked
    if grid_data[start.y][start.x] or grid_data[goal.y][goal.x]:
        return False
    
    # Simple BFS to check if goal is reachable from start
    queue = [start]
    visited = set([(start.x, start.y)])
    
    while queue:
        current = queue.pop(0)
        
        if current.x == goal.x and current.y == goal.y:
            return True
            
        # Check all 4 neighbors
        for dx, dy in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            nx, ny = current.x + dx, current.y + dy
            
            if (0 <= nx < width and 0 <= ny < height and 
                not grid_data[ny][nx] and (nx, ny) not in visited):
                neighbor = Point(nx, ny)
                queue.append(neighbor)
                visited.add((nx, ny))
    
    return False

def run_benchmark(width: int, height: int, obstacle_densities: List[float], trials_per_density: int = 5, 
                 save_grids: bool = False, group_by_density: bool = True):
    """
    Run benchmark comparing JPS4 and A* at different obstacle densities
    """
    results = {
        "JPS4": {density: {"time": [], "path_length": [], "nodes": [], "success": []} for density in obstacle_densities},
        "AStar": {density: {"time": [], "path_length": [], "nodes": [], "success": []} for density in obstacle_densities}
    }
    
    # Create directory for benchmark results
    os.makedirs(f"{path_dir}benchmark_results", exist_ok=True)
    
    for density in obstacle_densities:
        print(f"\nRunning benchmark with obstacle density: {density:.2f}")
        
        # Create density-specific subdirectory if grouping by density
        density_dir = f"{path_dir}benchmark_results/density_{density:.2f}"
        if group_by_density and save_grids:
            os.makedirs(density_dir, exist_ok=True)
            
        for trial in range(trials_per_density):
            print(f"  Trial {trial+1}/{trials_per_density}")
            
            # Generate a random grid and ensure it has a valid path
            max_attempts = 5  # Limit retries to avoid infinite loops
            for attempt in range(max_attempts):
                grid_data, start, goal = generate_grid(width, height, density, seed=trial*10 + attempt)
                
                # Validate grid connectivity
                has_path = validate_grid_connectivity(grid_data, start, goal, width, height)
                if has_path:
                    if attempt > 0:
                        print(f"    Found valid path after {attempt+1} attempts")
                    break
                elif attempt == max_attempts - 1:
                    print(f"    WARNING: Failed to generate a grid with a valid path after {max_attempts} attempts")
            
            # Save grid visualization if requested
            if save_grids and trial == 0:  # Save just the first trial for each density
                if group_by_density:
                    filepath = f"{density_dir}/grid_example.png"
                else:
                    filepath = f"{path_dir}benchmark_results/grid_density_{density:.2f}.png"
                visualize_grid(grid_data, start, goal, filepath)
            
            # Test JPS4
            JPS4_grid = JPS4(width, height)
            for y in range(height):
                for x in range(width):
                    if grid_data[y][x]:
                        JPS4_grid.set(x, y, True)
            
            nodes_expanded = [0]
            def count_nodes_JPS4(node):
                nodes_expanded[0] += 1
            
            start_time = time.time()
            path = JPS4_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_JPS4)
            elapsed = time.time() - start_time
            
            success = path is not None
            if not success and has_path:
                print(f"    WARNING: JPS4 failed to find a path even though one exists")
                
            results["JPS4"][density]["time"].append(elapsed)
            results["JPS4"][density]["nodes"].append(nodes_expanded[0])
            results["JPS4"][density]["success"].append(success)
            results["JPS4"][density]["path_length"].append(len(path) if path else float('inf'))
            
            # Test A*
            astar_grid = AStar(width, height)
            for y in range(height):
                for x in range(width):
                    if grid_data[y][x]:
                        astar_grid.set(x, y, True)
            
            nodes_expanded_astar = [0]
            def count_nodes_astar(node):
                nodes_expanded_astar[0] += 1
            
            start_time = time.time()
            path = astar_grid.find_path(start, goal, debug=False, on_node_expansion=count_nodes_astar)
            elapsed = time.time() - start_time
            
            success = path is not None
            if not success and has_path:
                print(f"    WARNING: A* failed to find a path even though one exists")
                
            results["AStar"][density]["time"].append(elapsed)
            results["AStar"][density]["nodes"].append(nodes_expanded_astar[0])
            results["AStar"][density]["success"].append(success)
            results["AStar"][density]["path_length"].append(len(path) if path else float('inf'))
            
            # Save density-specific report for this trial if grouping
            if group_by_density and trial == trials_per_density - 1:  # Last trial for this density
                generate_density_report(results, density, width, height, f"{density_dir}/report.txt")
                plot_density_results(results, density, f"{density_dir}/plots.png")
    
    # Generate overall summary report
    generate_report(results, obstacle_densities)
    
    return results

def run_research_benchmark():
    """
    Run a comprehensive benchmark similar to research paper parameters
    """
    # Define research paper grid sizes (large grids)
    width, height = 200, 200
    
    # Define obstacle densities focused on key ranges
    # Low (0-0.2), Medium (0.3-0.5), High (0.6-0.8)
    densities = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    
    # Number of trials per density (higher for research quality)
    trials = 10
    
    print("Starting RESEARCH-GRADE benchmark of JPS4 vs A* pathfinding...")
    print(f"Grid size: {width}x{height} (LARGE)")
    print(f"Obstacle densities: {densities}")
    print(f"Trials per density: {trials}")
    print("\nThis benchmark may take a long time to complete...\n")
    
    # Run the benchmark with grid visualizations and grouped results
    results = run_benchmark(width, height, densities, trials, True, True)
    
    # Plot results
    plot_results(results, densities)
    
    print("Research-grade benchmark complete!")

def run_quick_benchmark():
    """
    Run a quick benchmark for testing purposes
    """
    # Smaller grid size for quick results
    width, height = 50, 50
    
    # Key obstacle densities representing different scenarios
    densities = [0.0, 0.2, 0.4, 0.6, 0.8]
    
    # Fewer trials for faster results
    trials = 3
    
    print("Starting QUICK benchmark of JPS4 vs A* pathfinding...")
    print(f"Grid size: {width}x{height}")
    print(f"Obstacle densities: {densities}")
    print(f"Trials per density: {trials}")
    
    # Run the benchmark
    results = run_benchmark(width, height, densities, trials, True, True)
    
    # Plot results
    plot_results(results, densities)
    
    print("Quick benchmark complete!")

def run_specific_density_benchmark(density):
    """
    Run a focused benchmark for a specific density
    """
    # Moderate grid size
    width, height = 100, 100
    
    # Single density with more trials for statistical significance
    densities = [density]
    trials = 20
    
    print(f"Starting focused benchmark at density {density}...")
    print(f"Grid size: {width}x{height}")
    print(f"Trials: {trials}")
    
    # Run the benchmark
    results = run_benchmark(width, height, densities, trials, True, True)
    
    # Plot results
    plot_results(results, densities)
    
    print(f"Density {density} benchmark complete!")

def run_large_grid_benchmark():
    """
    Run benchmarks with progressively larger grid sizes to assess scaling properties
    """
    # Define grid sizes to test
    grid_sizes = [50, 100, 200, 300, 400, 500]
    
    # Use a moderate obstacle density
    obstacle_density = 0.4
    
    # Multiple trials for statistical significance
    trials_per_size = 5
    
    # Create directory for results
    os.makedirs(f"{path_dir}benchmark_results/large_grid", exist_ok=True)
    
    # Store results
    results = {
        "JPS4": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes},
        "AStar": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes}
    }
    
    for size in grid_sizes:
        print(f"\nRunning large grid benchmark with size {size}x{size}")
        
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
                filepath = f"{path_dir}benchmark_results/large_grid/grid_size_{size}.png"
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
                timeout = min(60, size / 50)  # Scale timeout with grid size
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
                timeout = min(60, size / 50)  # Scale timeout with grid size
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
    generate_scaling_report(results, grid_sizes)
    plot_scaling_results(results, grid_sizes)
    
    return results
