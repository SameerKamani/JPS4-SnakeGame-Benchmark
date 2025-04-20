from Benchmark_Functions.grid_and_benchmarks import run_large_grid_benchmark
from Benchmark_Functions.maze_and_benchmarks import run_maze_benchmark, run_benchmark, run_tiny_benchmark, run_tiny_scaling_benchmark, run_medium_large_benchmark
from Helper_Functions.plotting_helper_functions import plot_results

import argparse
import sys

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Benchmark JPS4 vs A*")

    parser.add_argument("--benchmark-type", type=str, default="quick", 
                        choices=["tiny", "tiny-scaling", "quick", "research", "low", "medium", "high", "large", "medium-large", "xlarge", "maze", "custom"],
                        help="Type of benchmark to run")
    
    parser.add_argument("--grid-size", type=int, default=None, 
                        help="Grid size for custom benchmark")

    parser.add_argument("--obstacle-densities", type=str, default=None, 
                       help="Comma-separated list of obstacle densities for custom benchmark")
    
    parser.add_argument("--trials", type=int, default=None, 
                        help="Number of trials for custom benchmark")
    
    args = parser.parse_args()

    # Define benchmark presets
    if args.benchmark_type == "tiny":
        print("Running tiny benchmark...")
        run_tiny_benchmark()
    
    elif args.benchmark_type == "tiny-scaling":
        print("Running tiny scaling benchmark...")
        run_tiny_scaling_benchmark()
    
    elif args.benchmark_type == "quick":
        print("Running quick benchmark...")
        run_benchmark(width=50, height=50, 
                     obstacle_densities=[0.40, 0.60, 0.80], 
                     trials_per_density=3, 
                     save_grids=True,
                     group_by_density=True)
                     
    elif args.benchmark_type == "research":
        print("Running research-grade benchmark...")
        run_benchmark(width=200, height=200, 
                     obstacle_densities=[0.00, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80], 
                     trials_per_density=10, 
                     save_grids=True,
                     group_by_density=True)
                     
    elif args.benchmark_type == "low":
        print("Running low-density benchmark...")
        run_benchmark(width=100, height=100, 
                     obstacle_densities=[0.05, 0.10, 0.15, 0.20, 0.25], 
                     trials_per_density=10, 
                     save_grids=True,
                     group_by_density=True)
                     
    elif args.benchmark_type == "medium":
        print("Running medium-density benchmark...")
        run_benchmark(width=100, height=100, 
                     obstacle_densities=[0.30, 0.35, 0.40, 0.45, 0.50], 
                     trials_per_density=10, 
                     save_grids=True,
                     group_by_density=True)
                     
    elif args.benchmark_type == "high":
        print("Running high-density benchmark...")
        run_benchmark(width=100, height=100, 
                     obstacle_densities=[0.55, 0.60, 0.65, 0.70, 0.75, 0.80], 
                     trials_per_density=10, 
                     save_grids=True,
                     group_by_density=True)

    elif args.benchmark_type == "large":
        print("Running large-grid benchmark...")
        run_large_grid_benchmark()
        
    elif args.benchmark_type == "medium-large":
        print("Running medium-large grid benchmark...")
        run_medium_large_benchmark()
        
    elif args.benchmark_type == "xlarge":
        print("Running extra-large grid benchmark (up to 500x500)...")

        # Define an extra large grid benchmark that uses our optimized implementation
        grid_sizes = [50, 100, 200, 300, 400, 500]
        
        # Use moderate obstacle density for consistency
        obstacle_density = 0.4
        trials_per_size = 3
        
        # Store results
        results = {
            "JPS4": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes},
            "AStar": {size: {"time": [], "nodes": [], "success": []} for size in grid_sizes}
        }
        
        # Run benchmark
        for size in grid_sizes:
            print(f"\nRunning XL benchmark with size {size}x{size}")
            run_benchmark(width=size, height=size, 
                         obstacle_densities=[obstacle_density], 
                         trials_per_density=trials_per_size, 
                         save_grids=(size <= 300),  # Don't save very large grids to avoid memory issues
                         group_by_density=True)
        
        print("Extra-large grid benchmark complete!")
        
    elif args.benchmark_type == "maze":
        print("Running maze-environment benchmark...")
        run_maze_benchmark()
                     
    elif args.benchmark_type == "custom":
        if args.grid_size is None or args.obstacle_densities is None or args.trials is None:
            print("For custom benchmark, must specify --grid-size, --obstacle-densities, and --trials")
            sys.exit(1)
            
        densities = [float(d) for d in args.obstacle_densities.split(",")]
        print(f"Running custom benchmark with grid size {args.grid_size}x{args.grid_size}, "
              f"densities {densities}, and {args.trials} trials per density...")
              
        run_benchmark(width=args.grid_size, height=args.grid_size, 
                     obstacle_densities=densities, 
                     trials_per_density=args.trials, 
                     save_grids=True,
                     group_by_density=True)

    else:
        # Custom benchmark with provided arguments
        print("Starting custom benchmark of JPS4 vs A* pathfinding...")
        print(f"Grid size: {args.width}x{args.height}")
        print(f"Obstacle densities: {args.densities}")
        print(f"Trials per density: {args.trials}")
        
        # Run benchmark
        results = run_benchmark(args.width, args.height, args.densities, args.trials, 
                              args.save_grids, args.group_by_density)
        
        # Plot results
        plot_results(results, args.densities)
        
        print("Benchmark complete!") 
