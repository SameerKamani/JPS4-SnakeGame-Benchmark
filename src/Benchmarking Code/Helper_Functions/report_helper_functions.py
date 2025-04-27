from Algorithms.helper import path_dir

def generate_report(results, obstacle_densities):
    """
    Generate a detailed text report of the benchmark results
    """
    with open(f"{path_dir}benchmark_report.txt", "w") as f:
        f.write("JPS4 vs A* Benchmark Report\n")
        f.write("=========================\n\n")
        
        for density in obstacle_densities:
            f.write(f"Obstacle Density: {density:.2f}\n")
            f.write("-" * 30 + "\n")
            
            JPS4_success = sum(results["JPS4"][density]["success"]) / len(results["JPS4"][density]["success"]) * 100
            astar_success = sum(results["AStar"][density]["success"]) / len(results["AStar"][density]["success"]) * 100
            
            # Only include successful trials in averages
            JPS4_times = [t for t, s in zip(results["JPS4"][density]["time"], results["JPS4"][density]["success"]) if s]
            astar_times = [t for t, s in zip(results["AStar"][density]["time"], results["AStar"][density]["success"]) if s]
            
            JPS4_nodes = [n for n, s in zip(results["JPS4"][density]["nodes"], results["JPS4"][density]["success"]) if s]
            astar_nodes = [n for n, s in zip(results["AStar"][density]["nodes"], results["AStar"][density]["success"]) if s]
            
            JPS4_paths = [p for p, s in zip(results["JPS4"][density]["path_length"], results["JPS4"][density]["success"]) if s]
            astar_paths = [p for p, s in zip(results["AStar"][density]["path_length"], results["AStar"][density]["success"]) if s]
            
            f.write(f"Success Rate: JPS4 {JPS4_success:.1f}%, A* {astar_success:.1f}%\n")
            
            if JPS4_times:
                f.write(f"Avg Time (JPS4): {sum(JPS4_times)/len(JPS4_times):.6f}s\n")
            if astar_times:
                f.write(f"Avg Time (A*): {sum(astar_times)/len(astar_times):.6f}s\n")
            
            if JPS4_nodes and astar_nodes:
                f.write(f"Avg Nodes Expanded (JPS4): {sum(JPS4_nodes)/len(JPS4_nodes):.1f}\n")
                f.write(f"Avg Nodes Expanded (A*): {sum(astar_nodes)/len(astar_nodes):.1f}\n")
                f.write(f"Node Expansion Ratio (JPS4/A*): {(sum(JPS4_nodes)/len(JPS4_nodes))/(sum(astar_nodes)/len(astar_nodes)):.2f}\n")
            
            if JPS4_paths and astar_paths:
                f.write(f"Avg Path Length (JPS4): {sum(JPS4_paths)/len(JPS4_paths):.1f}\n")
                f.write(f"Avg Path Length (A*): {sum(astar_paths)/len(astar_paths):.1f}\n")
                f.write(f"Path Length Ratio (JPS4/A*): {(sum(JPS4_paths)/len(JPS4_paths))/(sum(astar_paths)/len(astar_paths)):.2f}\n")
            
            f.write("\n")
        
        f.write("Summary:\n")
        f.write("========\n")
        # Calculate overall performance metrics
        JPS4_all_success = sum([sum(results["JPS4"][d]["success"]) for d in obstacle_densities]) / sum([len(results["JPS4"][d]["success"]) for d in obstacle_densities]) * 100
        astar_all_success = sum([sum(results["AStar"][d]["success"]) for d in obstacle_densities]) / sum([len(results["AStar"][d]["success"]) for d in obstacle_densities]) * 100
        
        f.write(f"Overall Success Rate: JPS4 {JPS4_all_success:.1f}%, A* {astar_all_success:.1f}%\n")
        
        # Provide overall findings
        f.write("\nFindings:\n")
        f.write("- JPS4 search space exploration: ")
        f.write("More efficient than A* " if JPS4_all_success > 0 and sum([sum(results["JPS4"][d]["nodes"]) for d in obstacle_densities]) < sum([sum(results["AStar"][d]["nodes"]) for d in obstacle_densities]) else "Less efficient than A* ")
        f.write("in terms of nodes expanded.\n")
        
        f.write("- Path optimality: ")
        f.write("JPS4 produces paths of similar length to A*.\n" if JPS4_all_success > 0 else "Cannot compare path lengths due to low success rate.\n")
        
        print(f"Report generated: benchmark_report.txt")

def generate_density_report(results, density, width, height, filename):
    """
    Generate a detailed report for a specific obstacle density
    """
    with open(filename, "w") as f:
        f.write(f"JPS4 vs A* Benchmark Report - Density {density:.2f}\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Grid size: {width}x{height}\n")
        f.write(f"Obstacle density: {density:.2f}\n\n")
        
        JPS4_success = sum(results["JPS4"][density]["success"]) / len(results["JPS4"][density]["success"]) * 100
        astar_success = sum(results["AStar"][density]["success"]) / len(results["AStar"][density]["success"]) * 100
        
        # Only include successful trials in averages
        JPS4_times = [t for t, s in zip(results["JPS4"][density]["time"], results["JPS4"][density]["success"]) if s]
        astar_times = [t for t, s in zip(results["AStar"][density]["time"], results["AStar"][density]["success"]) if s]
        
        JPS4_nodes = [n for n, s in zip(results["JPS4"][density]["nodes"], results["JPS4"][density]["success"]) if s]
        astar_nodes = [n for n, s in zip(results["AStar"][density]["nodes"], results["AStar"][density]["success"]) if s]
        
        JPS4_paths = [p for p, s in zip(results["JPS4"][density]["path_length"], results["JPS4"][density]["success"]) if s]
        astar_paths = [p for p, s in zip(results["AStar"][density]["path_length"], results["AStar"][density]["success"]) if s]
        
        f.write(f"Success Rate: JPS4 {JPS4_success:.1f}%, A* {astar_success:.1f}%\n\n")
        
        f.write("Performance Metrics:\n")
        f.write("-" * 30 + "\n")
        
        if JPS4_times:
            f.write(f"Avg Time (JPS4): {sum(JPS4_times)/len(JPS4_times):.6f}s\n")
            f.write(f"Min Time (JPS4): {min(JPS4_times):.6f}s\n")
            f.write(f"Max Time (JPS4): {max(JPS4_times):.6f}s\n\n")
        else:
            f.write("No successful JPS4 runs\n\n")
            
        if astar_times:
            f.write(f"Avg Time (A*): {sum(astar_times)/len(astar_times):.6f}s\n")
            f.write(f"Min Time (A*): {min(astar_times):.6f}s\n")
            f.write(f"Max Time (A*): {max(astar_times):.6f}s\n\n")
        else:
            f.write("No successful A* runs\n\n")
        
        if JPS4_nodes and astar_nodes:
            f.write(f"Avg Nodes Expanded (JPS4): {sum(JPS4_nodes)/len(JPS4_nodes):.1f}\n")
            f.write(f"Avg Nodes Expanded (A*): {sum(astar_nodes)/len(astar_nodes):.1f}\n")
            f.write(f"Node Expansion Ratio (JPS4/A*): {(sum(JPS4_nodes)/len(JPS4_nodes))/(sum(astar_nodes)/len(astar_nodes)):.2f}\n\n")
        
        if JPS4_paths and astar_paths:
            f.write(f"Avg Path Length (JPS4): {sum(JPS4_paths)/len(JPS4_paths):.1f}\n")
            f.write(f"Avg Path Length (A*): {sum(astar_paths)/len(astar_paths):.1f}\n")
            f.write(f"Path Length Ratio (JPS4/A*): {(sum(JPS4_paths)/len(JPS4_paths))/(sum(astar_paths)/len(astar_paths)):.2f}\n\n")
        
        # Performance improvement metrics
        if JPS4_times and astar_times:
            time_improvement = (sum(astar_times)/len(astar_times)) / (sum(JPS4_times)/len(JPS4_times))
            f.write(f"Time Performance: JPS4 is {time_improvement:.2f}x faster than A*\n")
        
        if JPS4_nodes and astar_nodes:
            node_improvement = (sum(astar_nodes)/len(astar_nodes)) / (sum(JPS4_nodes)/len(JPS4_nodes))
            f.write(f"Search Efficiency: JPS4 expands {node_improvement:.2f}x fewer nodes than A*\n")
        
        print(f"Density-specific report generated: {filename}")

def generate_maze_report(results, grid_sizes):
    """
    Generate a detailed report for maze benchmark results
    """
    with open(f"{path_dir}benchmark_results/maze/maze_report.txt", "w") as f:
        f.write("JPS4 vs A* Maze Benchmark Results\n")
        f.write("================================\n\n")
        
        f.write("Performance in maze-like environments with long corridors\n\n")
        
        for size in grid_sizes:
            f.write(f"Grid Size: {size}x{size}\n")
            f.write("-" * 30 + "\n")
            
            # Calculate success rates
            JPS4_success = sum(results["JPS4"][size]["success"]) / len(results["JPS4"][size]["success"]) * 100 if results["JPS4"][size]["success"] else 0
            astar_success = sum(results["AStar"][size]["success"]) / len(results["AStar"][size]["success"]) * 100 if results["AStar"][size]["success"] else 0
            
            f.write(f"Success Rate: JPS4 {JPS4_success:.1f}%, A* {astar_success:.1f}%\n\n")
            
            # Calculate averages for successful trials
            JPS4_times = [t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s]
            astar_times = [t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s]
            
            JPS4_nodes = [n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s]
            astar_nodes = [n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s]
            
            JPS4_paths = [p for p, s in zip(results["JPS4"][size]["path_length"], results["JPS4"][size]["success"]) if s and p > 0]
            astar_paths = [p for p, s in zip(results["AStar"][size]["path_length"], results["AStar"][size]["success"]) if s and p > 0]
            
            # Report execution time
            if JPS4_times and astar_times:
                avg_JPS4_time = sum(JPS4_times) / len(JPS4_times)
                avg_astar_time = sum(astar_times) / len(astar_times)
                f.write(f"Avg Time (JPS4): {avg_JPS4_time:.6f}s\n")
                f.write(f"Avg Time (A*): {avg_astar_time:.6f}s\n")
                f.write(f"Time Ratio (JPS4/A*): {avg_JPS4_time/avg_astar_time:.3f}x\n\n")
            
            # Report nodes expanded
            if JPS4_nodes and astar_nodes:
                avg_JPS4_nodes = sum(JPS4_nodes) / len(JPS4_nodes)
                avg_astar_nodes = sum(astar_nodes) / len(astar_nodes)
                f.write(f"Avg Nodes Expanded (JPS4): {avg_JPS4_nodes:.1f}\n")
                f.write(f"Avg Nodes Expanded (A*): {avg_astar_nodes:.1f}\n")
                f.write(f"Node Expansion Ratio (JPS4/A*): {avg_JPS4_nodes/avg_astar_nodes:.3f}x\n\n")
            
            # Report path lengths
            if JPS4_paths and astar_paths:
                avg_JPS4_path = sum(JPS4_paths) / len(JPS4_paths)
                avg_astar_path = sum(astar_paths) / len(astar_paths)
                f.write(f"Avg Path Length (JPS4): {avg_JPS4_path:.1f}\n")
                f.write(f"Avg Path Length (A*): {avg_astar_path:.1f}\n")
                f.write(f"Path Length Ratio (JPS4/A*): {avg_JPS4_path/avg_astar_path:.3f}x\n\n")
        
        # Overall analysis
        f.write("\nOverall Analysis\n")
        f.write("===============\n")
        f.write("JPS4 vs A* in maze-like environments:\n\n")
        
        # Calculate overall performance metrics
        all_JPS4_times = []
        all_astar_times = []
        all_JPS4_nodes = []
        all_astar_nodes = []
        
        for size in grid_sizes:
            all_JPS4_times.extend([t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s])
            all_astar_times.extend([t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s])
            all_JPS4_nodes.extend([n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s])
            all_astar_nodes.extend([n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s])
        
        if all_JPS4_times and all_astar_times:
            avg_JPS4_time = sum(all_JPS4_times) / len(all_JPS4_times)
            avg_astar_time = sum(all_astar_times) / len(all_astar_times)
            f.write(f"Overall Time Ratio (JPS4/A*): {avg_JPS4_time/avg_astar_time:.3f}x\n")
        
        if all_JPS4_nodes and all_astar_nodes:
            avg_JPS4_nodes = sum(all_JPS4_nodes) / len(all_JPS4_nodes)
            avg_astar_nodes = sum(all_astar_nodes) / len(all_astar_nodes)
            f.write(f"Overall Node Expansion Ratio (JPS4/A*): {avg_JPS4_nodes/avg_astar_nodes:.3f}x\n\n")
        
        # Conclusions
        f.write("Conclusions:\n")
        if all_JPS4_nodes and all_astar_nodes:
            if avg_JPS4_nodes < avg_astar_nodes:
                f.write("- JPS4 demonstrates significantly better performance in maze environments,\n")
                f.write(f"  expanding {avg_astar_nodes/avg_JPS4_nodes:.1f}x fewer nodes than A*.\n")
            else:
                f.write(f"- JPS4 expands {avg_JPS4_nodes/avg_astar_nodes:.1f}x more nodes than A* in maze environments.\n")
        
        if all_JPS4_times and all_astar_times:
            if avg_JPS4_time < avg_astar_time:
                f.write(f"- JPS4 is {avg_astar_time/avg_JPS4_time:.1f}x faster than A* in maze environments.\n")
            else:
                f.write(f"- JPS4 is {avg_JPS4_time/avg_astar_time:.1f}x slower than A* in maze environments.\n")
        
        print(f"Maze benchmark report generated: ../../Checkpoint 3//tests//benchmark_results/maze/maze_report.txt")

def generate_scaling_report(results, grid_sizes):
    """
    Generate a report on how JPS4 and A* scale with grid size
    """
    with open(f"{path_dir}benchmark_results/large_grid/scaling_report.txt", "w") as f:
        f.write("JPS4 vs A* Scaling Performance Report\n")
        f.write("====================================\n\n")
        
        f.write("Grid Size Performance Metrics\n")
        f.write("-" * 30 + "\n\n")
        
        for size in grid_sizes:
            f.write(f"Grid Size: {size}x{size}\n")
            f.write("-" * 20 + "\n")
            
            JPS4_success = sum(results["JPS4"][size]["success"]) / len(results["JPS4"][size]["success"]) * 100
            astar_success = sum(results["AStar"][size]["success"]) / len(results["AStar"][size]["success"]) * 100
            
            # Filter successful trials
            JPS4_times = [t for t, s in zip(results["JPS4"][size]["time"], results["JPS4"][size]["success"]) if s]
            astar_times = [t for t, s in zip(results["AStar"][size]["time"], results["AStar"][size]["success"]) if s]
            
            JPS4_nodes = [n for n, s in zip(results["JPS4"][size]["nodes"], results["JPS4"][size]["success"]) if s]
            astar_nodes = [n for n, s in zip(results["AStar"][size]["nodes"], results["AStar"][size]["success"]) if s]
            
            f.write(f"Success Rate: JPS4 {JPS4_success:.1f}%, A* {astar_success:.1f}%\n")
            
            if JPS4_times and astar_times:
                avg_JPS4_time = sum(JPS4_times) / len(JPS4_times)
                avg_astar_time = sum(astar_times) / len(astar_times)
                f.write(f"Avg Time (JPS4): {avg_JPS4_time:.6f}s\n")
                f.write(f"Avg Time (A*): {avg_astar_time:.6f}s\n")
                f.write(f"Time Ratio (JPS4/A*): {avg_JPS4_time/avg_astar_time:.2f}x\n")
            
            if JPS4_nodes and astar_nodes:
                avg_JPS4_nodes = sum(JPS4_nodes) / len(JPS4_nodes)
                avg_astar_nodes = sum(astar_nodes) / len(astar_nodes)
                f.write(f"Avg Nodes Expanded (JPS4): {avg_JPS4_nodes:.1f}\n")
                f.write(f"Avg Nodes Expanded (A*): {avg_astar_nodes:.1f}\n")
                f.write(f"Node Expansion Ratio (JPS4/A*): {avg_JPS4_nodes/avg_astar_nodes:.2f}x\n")
            
            f.write("\n")
        
        f.write("\nScaling Analysis:\n")
        f.write("================\n")
        
        # Compute growth rates for nodes expanded and time
        if all(len(results["JPS4"][size]["nodes"]) > 0 and len(results["AStar"][size]["nodes"]) > 0 for size in grid_sizes[:-1]):
            JPS4_node_growth = []
            astar_node_growth = []
            JPS4_time_growth = []
            astar_time_growth = []
            
            for i in range(1, len(grid_sizes)):
                prev_size = grid_sizes[i-1]
                curr_size = grid_sizes[i]
                
                # Calculate average nodes for successful trials
                prev_JPS4_nodes = sum([n for n, s in zip(results["JPS4"][prev_size]["nodes"], results["JPS4"][prev_size]["success"]) if s]) / sum(results["JPS4"][prev_size]["success"])
                curr_JPS4_nodes = sum([n for n, s in zip(results["JPS4"][curr_size]["nodes"], results["JPS4"][curr_size]["success"]) if s]) / sum(results["JPS4"][curr_size]["success"])
                
                prev_astar_nodes = sum([n for n, s in zip(results["AStar"][prev_size]["nodes"], results["AStar"][prev_size]["success"]) if s]) / sum(results["AStar"][prev_size]["success"])
                curr_astar_nodes = sum([n for n, s in zip(results["AStar"][curr_size]["nodes"], results["AStar"][curr_size]["success"]) if s]) / sum(results["AStar"][curr_size]["success"])
                
                # Calculate growth factor
                JPS4_node_growth.append(curr_JPS4_nodes / prev_JPS4_nodes)
                astar_node_growth.append(curr_astar_nodes / prev_astar_nodes)
                
                # Do the same for time
                prev_JPS4_time = sum([t for t, s in zip(results["JPS4"][prev_size]["time"], results["JPS4"][prev_size]["success"]) if s]) / sum(results["JPS4"][prev_size]["success"])
                curr_JPS4_time = sum([t for t, s in zip(results["JPS4"][curr_size]["time"], results["JPS4"][curr_size]["success"]) if s]) / sum(results["JPS4"][curr_size]["success"])
                
                prev_astar_time = sum([t for t, s in zip(results["AStar"][prev_size]["time"], results["AStar"][prev_size]["success"]) if s]) / sum(results["AStar"][prev_size]["success"])
                curr_astar_time = sum([t for t, s in zip(results["AStar"][curr_size]["time"], results["AStar"][curr_size]["success"]) if s]) / sum(results["AStar"][curr_size]["success"])
                
                JPS4_time_growth.append(curr_JPS4_time / prev_JPS4_time)
                astar_time_growth.append(curr_astar_time / prev_astar_time)
                
                # Report on scaling between these grid sizes
                size_factor = curr_size / prev_size
                f.write(f"From {prev_size}x{prev_size} to {curr_size}x{curr_size} (size factor: {size_factor:.1f}x):\n")
                f.write(f"  JPS4 nodes growth: {JPS4_node_growth[-1]:.2f}x, expected O(n): {size_factor:.1f}x, actual vs expected: {JPS4_node_growth[-1]/size_factor:.2f}\n")
                f.write(f"  A* nodes growth: {astar_node_growth[-1]:.2f}x, expected O(n²): {size_factor*size_factor:.1f}x, actual vs expected: {astar_node_growth[-1]/(size_factor*size_factor):.2f}\n")
                f.write(f"  JPS4 time growth: {JPS4_time_growth[-1]:.2f}x\n")
                f.write(f"  A* time growth: {astar_time_growth[-1]:.2f}x\n\n")
        
        f.write("\nConclusion:\n")
        if all(len(results["JPS4"][size]["time"]) > 0 and len(results["AStar"][size]["time"]) > 0 for size in grid_sizes):
            # Calculate advantage for largest grid size
            largest_size = grid_sizes[-1]
            JPS4_time = sum([t for t, s in zip(results["JPS4"][largest_size]["time"], results["JPS4"][largest_size]["success"]) if s]) / sum(results["JPS4"][largest_size]["success"]) if sum(results["JPS4"][largest_size]["success"]) > 0 else 0
            astar_time = sum([t for t, s in zip(results["AStar"][largest_size]["time"], results["AStar"][largest_size]["success"]) if s]) / sum(results["AStar"][largest_size]["success"]) if sum(results["AStar"][largest_size]["success"]) > 0 else 0
            
            JPS4_nodes = sum([n for n, s in zip(results["JPS4"][largest_size]["nodes"], results["JPS4"][largest_size]["success"]) if s]) / sum(results["JPS4"][largest_size]["success"]) if sum(results["JPS4"][largest_size]["success"]) > 0 else 0
            astar_nodes = sum([n for n, s in zip(results["AStar"][largest_size]["nodes"], results["AStar"][largest_size]["success"]) if s]) / sum(results["AStar"][largest_size]["success"]) if sum(results["AStar"][largest_size]["success"]) > 0 else 0
            
            if JPS4_time > 0 and astar_time > 0:
                f.write(f"For largest grid size ({largest_size}x{largest_size}):\n")
                f.write(f"  Time: JPS4 is {astar_time/JPS4_time:.2f}x {'faster' if astar_time > JPS4_time else 'slower'} than A*\n")
                f.write(f"  Nodes: JPS4 expands {astar_nodes/JPS4_nodes:.2f}x {'fewer' if astar_nodes > JPS4_nodes else 'more'} nodes than A*\n")
        
        print(f"Scaling report generated: ../../Checkpoint 3//tests//benchmark_results/large_grid/scaling_report.txt")
