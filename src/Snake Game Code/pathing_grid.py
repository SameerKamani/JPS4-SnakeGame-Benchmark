from typing import List, Tuple, Optional, Callable
from helper import Point, UnionFind, AstarContext

# Uniform movement cost (for cardinal moves)
C = 1

class PathingGrid:
    def __init__(self, width: int, height: int, default_value: bool):
        self.width = width
        self.height = height
        self.grid: List[List[bool]] = [[default_value for _ in range(width)] for _ in range(height)]
        self.components = UnionFind(width * height)
        self.components_dirty = False
        self.heuristic_factor = 1.0
        self.context = AstarContext()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> bool:
        return self.grid[y][x]

    def set(self, x: int, y: int, blocked: bool):
        if not self.in_bounds(x, y):
            return
        old_val = self.grid[y][x]
        self.grid[y][x] = blocked
        if old_val != blocked:
            self.components_dirty = True

    def neighborhood_points(self, point: Point) -> List[Point]:
        return point.neumann_neighborhood()

    def neighborhood_points_and_cost(self, pos: Point) -> List[Tuple[Point, int]]:
        result = []
        for p in self.neighborhood_points(pos):
            if self.can_move_to(p):
                result.append((p, C))
        return result

    def heuristic(self, p1: Point, p2: Point) -> int:
        return p1.manhattan_distance(p2) * C

    def can_move_to(self, pos: Point) -> bool:
        return self.in_bounds(pos.x, pos.y) and not self.grid[pos.y][pos.x]

    # --- JPS4-specific methods ---
    def jps_prune_neighbors(self, parent: Optional[Point], node: Point) -> List[Point]:
        """
        Prune neighbors according to JPS4 (Horizontal-First Jump Point Search) rules.
        Returns a list of neighbors that should be considered for jumping.
        """
        # Without a parent, return all 4-connected neighbors that are not blocked
        if parent is None:
            return [n for n in node.neumann_neighborhood() if self.can_move_to(n)]
            
        # Get the direction from parent to current node
        direction = parent.direction_to(node)
        
        # Horizontal movement only allows horizontal continuation and forced neighbors
        # Vertical movement only allows vertical continuation and forced neighbors
        pruned = []
        
        # JPS4 always prioritizes horizontal movement first if possible 
        if direction.x != 0:  # We're moving horizontally
            # Continue in same horizontal direction
            next_horizontal = Point(node.x + direction.x, node.y)
            if self.can_move_to(next_horizontal):
                pruned.append(next_horizontal)
                
            # Check for forced neighbors in vertical directions
            for dy in [-1, 1]:
                # A neighbor is forced if the cell adjacent to parent is blocked
                # but the neighbor is not blocked
                adjacent_to_parent = Point(parent.x, parent.y + dy)
                candidate = Point(node.x, node.y + dy)
                if not self.can_move_to(adjacent_to_parent) and self.can_move_to(candidate):
                    pruned.append(candidate)
                    
        else:  # We're moving vertically
            # Continue in same vertical direction
            next_vertical = Point(node.x, node.y + direction.y)
            if self.can_move_to(next_vertical):
                pruned.append(next_vertical)
                
            # Check for forced neighbors in horizontal directions
            for dx in [-1, 1]:
                # A neighbor is forced if the cell adjacent to parent is blocked
                # but the neighbor is not blocked
                adjacent_to_parent = Point(parent.x + dx, parent.y)
                candidate = Point(node.x + dx, node.y)
                if not self.can_move_to(adjacent_to_parent) and self.can_move_to(candidate):
                    pruned.append(candidate)
                    
        return pruned

    def jps_jump(self, current: Point, direction: Point, goal: Point) -> Optional[Point]:
        """
        Jump from current node in direction until finding:
        1. The goal node
        2. A forced neighbor
        3. A blocked cell
        
        For JPS4, we always return horizontal steps immediately and continue vertical jumps.
        """
        # Calculate the next point in the direction
        next_point = Point(current.x + direction.x, current.y + direction.y)
        
        # If out of bounds or blocked, stop
        if not self.can_move_to(next_point):
            print(f"DEBUG: Jump stopped at {next_point} - blocked or out of bounds")
            return None
            
        # If this is the goal, return it
        if isinstance(goal, Point) and next_point == goal:
            print(f"DEBUG: Jump found goal at {next_point}")
            return next_point
        elif callable(goal) and goal(next_point):
            print(f"DEBUG: Jump found goal at {next_point}")
            return next_point
            
        # For horizontal movement, return immediately (key aspect of JPS4)
        if direction.x != 0:
            # Check for forced neighbors before returning
            has_forced = False
            for dy in [-1, 1]:
                check_prev = Point(current.x, current.y + dy)
                check_next = Point(next_point.x, next_point.y + dy)
                if not self.can_move_to(check_prev) and self.can_move_to(check_next):
                    has_forced = True
                    break
                    
            if has_forced:
                print(f"DEBUG: Jump found forced neighbor at {next_point} (horizontal)")
            else:
                print(f"DEBUG: Jump stopped at {next_point} (horizontal - no forced)")
                
            return next_point  # Always return for horizontal movement in JPS4
                    
        # For vertical movement, check for forced neighbors
        if direction.y != 0:
            for dx in [-1, 1]:
                check_prev = Point(current.x + dx, current.y)
                check_next = Point(next_point.x + dx, next_point.y)
                if not self.can_move_to(check_prev) and self.can_move_to(check_next):
                    print(f"DEBUG: Jump found forced neighbor at {next_point} (vertical)")
                    return next_point  # Found a forced neighbor
                    
            # No forced neighbors, recursively continue in same direction
            result = self.jps_jump(next_point, direction, goal)
            return result
                
        # Should never reach here
        return None

    def jps_successors(self, parent: Optional[Point], node: Point, goal: Callable[[Point], bool]) -> List[Tuple[Point, int]]:
        """
        Generate successors according to JPS4 algorithm.
        
        Args:
            parent: The parent node (None for starting node)
            node: The current node
            goal: A function that returns True if a node is a goal
            
        Returns:
            List of (successor, cost) tuples
        """
        print(f"DEBUG: Finding successors for {node} with parent {parent}")
        successors = []
        
        # Special case: if this is the goal node, return it immediately
        if goal(node):
            print(f"DEBUG: Current node {node} is the goal")
            return [(node, 0)]
            
        # Check if we're adjacent to the goal - direct path is preferred
        for adj in node.neumann_neighborhood():
            if self.can_move_to(adj) and goal(adj):
                print(f"DEBUG: Found goal {adj} adjacent to {node}")
                return [(adj, C)]
        
        # Get pruned neighbors according to JPS4 rules
        pruned = self.jps_prune_neighbors(parent, node)
        print(f"DEBUG: Pruned neighbors for {node}: {pruned}")
        
        # For each pruned neighbor, try to jump
        for neighbor in pruned:
            # Calculate the direction from current node to neighbor
            direction = node.direction_to(neighbor)
            
            # Try to jump in that direction
            jump_result = self.jps_jump(node, direction, goal)
            
            # If we found a jump point, add it to successors
            if jump_result is not None:
                cost = C * node.manhattan_distance(jump_result)
                print(f"DEBUG: Jump from {node} found successor {jump_result} with cost {cost}")
                successors.append((jump_result, cost))
                
        # If no successors were found and this isn't the start node, add fallback options
        if not successors and parent is not None:
            print(f"DEBUG: No successors found for {node}, adding fallbacks")
            # Add regular neighbors that aren't the parent
            for adj in node.neumann_neighborhood():
                if self.can_move_to(adj) and adj != parent:
                    print(f"DEBUG: Adding fallback neighbor {adj}")
                    successors.append((adj, C))
                    
        print(f"DEBUG: Final successors for {node}: {[s[0] for s in successors]}")
        return successors

    def can_potentially_reach_goal(self, node: Point, goal: Callable[[Point], bool]) -> bool:
        """Check if this node has potential to reach the goal (simple heuristic check)"""
        # Check a few sample points to see if the goal is potentially reachable
        to_visit = [node]
        visited = {node}
        depth = 0
        max_depth = 5  # Limit the depth to avoid excessive computation
        
        while to_visit and depth < max_depth:
            depth += 1
            next_to_visit = []
            for current in to_visit:
                if goal(current):
                    return True
                for neighbor in self.neighborhood_points(current):
                    if neighbor not in visited and self.can_move_to(neighbor):
                        visited.add(neighbor)
                        next_to_visit.append(neighbor)
            to_visit = next_to_visit
                
        return False

    def generate_components(self):
        """Reset and regenerate the UnionFind components to check connectivity between cells."""
        self.components = UnionFind(self.width * self.height)
        self.components_dirty = False
        
        # For each free cell, connect it to its free neighbors
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x]:  # If this cell is free
                    idx = y * self.width + x
                    # Check all 4-connected neighbors
                    for nx, ny in [(x+1, y), (x-1, y), (x, y+1), (x, y-1)]:
                        if 0 <= nx < self.width and 0 <= ny < self.height and not self.grid[ny][nx]:
                            nidx = ny * self.width + nx
                            self.components.union(idx, nidx)
        
        # Print debug information about connectivity
        print("DEBUG: Updated UnionFind components for connectivity")

    def reachable(self, start: Point, end: Point) -> bool:
        """
        Check if there is a path from start to end using the UnionFind connectivity components.
        
        Args:
            start: The starting point
            end: The ending point
            
        Returns:
            bool: True if there's a path, False otherwise
        """
        # First check if both points are in bounds and not blocked
        if not (self.in_bounds(start.x, start.y) and self.in_bounds(end.x, end.y)):
            return False
            
        if self.grid[start.y][start.x] or self.grid[end.y][end.x]:
            return False
            
        # If the components are dirty, regenerate them
        if self.components_dirty:
            self.generate_components()
            
        # Check if start and end are in the same connected component
        start_idx = start.y * self.width + start.x
        end_idx = end.y * self.width + end.x
        
        are_connected = self.components.equiv(start_idx, end_idx)
        print(f"DEBUG: Testing reachability from {start} to {end}: {are_connected}")
        return are_connected

    def get_waypoints_single_goal(self, start: Point, goal: Point, mode: str = "astar") -> List[Point]:
        """Find a path from start to goal using the specified algorithm."""
        print("\nDEBUG: Printing Path Grid (X=blocked, .=free):")
        for row in range(self.height):
            print("".join("X" if self.grid[row][col] else "." for col in range(self.width)))
        print(f"DEBUG: Searching path from {start} to {goal} (approx={mode})")
        
        # Try a direct path check first - the most common case for adjacent cells
        if start.manhattan_distance(goal) == 1:
            print(f"DEBUG: Direct adjacent path found from {start} to {goal}")
            return [start, goal]
        
        # Check for direct line paths (horizontal, vertical, or diagonal)
        direct_path = self.find_direct_path(start, goal)
        if direct_path:
            print(f"DEBUG: Direct path found: {direct_path}")
            return direct_path
        
        # Check if the goal is reachable using UnionFind
        if not self.reachable(start, goal):
            print("DEBUG: Not reachable (UnionFind check)!")
            return None
            
        # Define a goal test function
        goal_test = lambda pt: pt == goal
        
        # Create heuristic function
        heuristic = lambda point: int(self.heuristic(point, goal) * self.heuristic_factor)
        
        # First, try with JPS4
        if mode == "jps4":
            result = self.context.astar_jps(
                start,
                lambda p, n: self.jps_successors(p, n, goal_test),
                heuristic,
                goal_test
            )
        else:
            # Standard A* fallback
            result = self.context.astar_jps(
                start,
                lambda p, n: self.standard_astar_successors(n, goal_test),
                heuristic,
                goal_test
            )
        
        # If no path found, attempt standard A* as fallback
        if result is None and mode == "jps4":
            print("DEBUG: JPS4 failed, falling back to standard A*")
            result = self.context.astar_jps(
                start,
                lambda p, n: self.standard_astar_successors(n, goal_test),
                heuristic,
                goal_test
            )
            
        # If still no path found, return None
        if result is None:
            print("DEBUG: No path found!")
            return None
            
        (path, cost) = result
        print(f"DEBUG: Raw path found: {path}")
        
        # Make sure the goal is in the path
        if path and path[-1] != goal and path[-1].manhattan_distance(goal) == 1:
            path.append(goal)
            print(f"DEBUG: Extended path with goal: {path}")
            
        # Handle the case of adjacent start and goal
        if start.manhattan_distance(goal) == 1 and len(path) < 2:
            path = [start, goal]
            print(f"DEBUG: Forced adjacent path: {path}")
            
        # Try to find post-processing optimizations
        optimized = self.post_process_path(path)
        print(f"DEBUG: Post-processed path: {optimized}")
        
        # Validate the optimized path
        if not self.validate_path(optimized):
            print("DEBUG: Optimized path validation failed, returning original path")
            # If optimization fails, check the original path
            if self.validate_path(path):
                return path
            else:
                print("DEBUG: Original path also invalid. Path finding failed.")
                return None
                
        return optimized
    
    def find_direct_path(self, start: Point, goal: Point) -> Optional[List[Point]]:
        """
        Check if there's a direct, unobstructed path between start and goal.
        Returns a list of points forming the path if one exists, None otherwise.
        """
        # Check if they are equal or adjacent
        if start == goal:
            return [start]
        if start.manhattan_distance(goal) == 1:
            return [start, goal]
            
        # Check for horizontal line
        if start.y == goal.y:
            min_x, max_x = min(start.x, goal.x), max(start.x, goal.x)
            # Check if path is clear
            for x in range(min_x, max_x + 1):
                if x != start.x and x != goal.x and self.grid[start.y][x]:
                    return None
            # Generate the path
            path = []
            step = 1 if goal.x > start.x else -1
            for x in range(start.x, goal.x + step, step):
                path.append(Point(x, start.y))
            return path
            
        # Check for vertical line
        if start.x == goal.x:
            min_y, max_y = min(start.y, goal.y), max(start.y, goal.y)
            # Check if path is clear
            for y in range(min_y, max_y + 1):
                if y != start.y and y != goal.y and self.grid[y][start.x]:
                    return None
            # Generate the path
            path = []
            step = 1 if goal.y > start.y else -1
            for y in range(start.y, goal.y + step, step):
                path.append(Point(start.x, y))
            return path
        
        # Not a direct line path
        return None
    
    def post_process_path(self, path: List[Point]) -> List[Point]:
        """
        Post-process the path to find a more optimal route while maintaining adjacency.
        Uses a combination of line-of-sight checks and path smoothing.
        """
        if len(path) <= 2:
            return path
            
        # Create a new optimized path starting with the first point
        optimized = [path[0]]
        current_idx = 0
        
        while current_idx < len(path) - 1:
            current = optimized[-1]
            
            # Try to find the furthest point in the path that has a direct line of sight
            furthest_idx = current_idx + 1
            for i in range(len(path) - 1, current_idx, -1):
                direct_path = self.find_direct_path(current, path[i])
                if direct_path:
                    furthest_idx = i
                    break
            
            # If we found a furthest visible point
            if furthest_idx > current_idx + 1:
                # Add the direct path (skipping the current point which is already added)
                direct_path = self.find_direct_path(current, path[furthest_idx])
                if direct_path:
                    optimized.extend(direct_path[1:])  # Skip the first point as it's already in the path
                current_idx = furthest_idx
            else:
                # Just add the next point
                optimized.append(path[current_idx + 1])
                current_idx += 1
        
        return optimized

    def standard_astar_successors(self, node: Point, goal: Callable[[Point], bool]) -> List[Tuple[Point, int]]:
        """Standard A* successor function that returns all valid neighbors."""
        successors = []
        
        # If this is the goal node, return it immediately
        if goal(node):
            return [(node, 0)]
            
        # Return all valid neighbors
        for neighbor in self.neighborhood_points(node):
            if self.can_move_to(neighbor):
                successors.append((neighbor, C))
                
        return successors

    def get_path_single_goal(self, start: Point, goal: Point, mode: str = "astar") -> Optional[List[Point]]:
        waypoints = self.get_waypoints_single_goal(start, goal, mode)
        if waypoints is None or len(waypoints) == 0:
            return None
            
        # Convert waypoints to a path with adjacent steps
        path = self.waypoints_to_path(waypoints)
        print(f"DEBUG: Final path after waypoints_to_path: {path}")
        
        # Validate the final path
        if not self.validate_path(path):
            print("DEBUG: WARNING - Path validation failed!")
            # If validation failed, we'll still return the path but log the warning
        
        return path

    def waypoints_to_path(self, waypoints: List[Point]) -> List[Point]:
        """Convert waypoints to a path where each step is adjacent to the previous one."""
        if not waypoints:
            return []
        
        # Start with the first waypoint
        path = [waypoints[0]]
        
        # Process each waypoint
        for i in range(1, len(waypoints)):
            current = path[-1]
            target = waypoints[i]
            
            # Add intermediate steps until we reach the target
            while current.manhattan_distance(target) > 0:
                # Take a single step toward the target
                direction = current.direction_to(target)
                current = current + direction
                path.append(current)
                
                # If we've reached the target, break
                if current.x == target.x and current.y == target.y:
                    break
        
        return path

    def optimize_path(self, path: List[Point]) -> List[Point]:
        """
        Optimize the path by removing unnecessary waypoints while ensuring all steps are adjacent.
        
        Args:
            path: List of Points representing the path
            
        Returns:
            List[Point]: Optimized path where each point is adjacent to the previous point
        """
        # Just delegate to the new post_process_path method
        return self.post_process_path(path)

    def validate_optimized_path(self, path: List[Point]) -> bool:
        """Verify that an optimized path has only adjacent steps."""
        if not path or len(path) < 2:
            return True
            
        for i in range(len(path) - 1):
            if path[i].manhattan_distance(path[i+1]) > 1:
                return False
                
        return True

    def validate_path(self, path: List[Point]) -> bool:
        """Verify that a path has only adjacent steps and all points are valid."""
        if not path or len(path) < 2:
            return True
            
        for i in range(len(path) - 1):
            # Check adjacency
            if path[i].manhattan_distance(path[i+1]) > 1:
                print(f"DEBUG: Non-adjacent points in path: {path[i]} -> {path[i+1]}")
                return False
                
            # Check that both points are valid (in bounds and not blocked)
            for p in [path[i], path[i+1]]:
                if not self.in_bounds(p.x, p.y) or self.grid[p.y][p.x]:
                    print(f"DEBUG: Invalid point in path: {p}")
                    return False
                    
        return True
