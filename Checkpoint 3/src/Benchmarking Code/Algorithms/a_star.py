import heapq
from typing import List, Optional, Tuple
from Algorithms.helper import Point, SearchNode

class AStar:
    """
    Grid implementation for the A* algorithm.
    """
    def __init__(self, width: int, height: int, default_blocked: bool = False, allow_diagonal: bool = False):
        self.width = width
        self.height = height

        # True means blocked, False means passable
        self.grid = [[default_blocked for _ in range(width)] for _ in range(height)]

        # Whether to allow diagonal movement
        self.allow_diagonal = allow_diagonal

        # Movement costs
        self.cardinal_cost = 1.0
        self.diagonal_cost = 1.414  # √2
    
    def in_bounds(self, x: int, y: int) -> bool:
        """Check if a point is within grid bounds."""
        return 0 <= x < self.width and 0 <= y < self.height
    
    def is_blocked(self, x: int, y: int) -> bool:
        """Check if a cell is blocked (obstacle)."""
        return not self.in_bounds(x, y) or self.grid[y][x]
    
    def can_move_to(self, point: Point) -> bool:
        """Check if a point is a valid move target."""
        return self.in_bounds(point.x, point.y) and not self.grid[point.y][point.x]
    
    def set(self, x: int, y: int, blocked: bool) -> None:
        """Set a cell's blocked status."""
        if self.in_bounds(x, y):
            self.grid[y][x] = blocked
    
    def get_neighbors(self, point: Point) -> List[Tuple[Point, float]]:
        """Get valid neighbors of a point with movement costs."""
        neighbors = []
        
        # Cardinal neighbors (N, E, S, W)
        for neighbor in point.cardinal_neighbors():
            if self.can_move_to(neighbor):
                neighbors.append((neighbor, self.cardinal_cost))
        
        # Diagonal neighbors (NE, SE, SW, NW)
        if self.allow_diagonal:
            # Get diagonal neighbors
            diagonals = [
                Point(1, -1), Point(1, 1), Point(-1, 1), Point(-1, -1)
            ]
            
            for diagonal in diagonals:
                neighbor = point + diagonal
                if self.can_move_to(neighbor):
                    # Check if we can move diagonally (no corner cutting)
                    if not self.is_blocked(point.x + diagonal.x, point.y) and \
                       not self.is_blocked(point.x, point.y + diagonal.y):
                        neighbors.append((neighbor, self.diagonal_cost))
        
        return neighbors
    
    def heuristic(self, p1: Point, p2: Point) -> float:
        """Heuristic function for A* search."""
        if self.allow_diagonal:
            # Diagonal distance (allows diagonal movement)
            dx = abs(p1.x - p2.x)
            dy = abs(p1.y - p2.y)
            return self.cardinal_cost * max(dx, dy) + (self.diagonal_cost - self.cardinal_cost) * min(dx, dy)
        else:
            # Manhattan distance (for cardinal-only movement)
            return self.cardinal_cost * p1.manhattan_distance(p2)
    
    def find_path(self, start: Point, goal: Point, debug: bool = False, on_node_expansion=None) -> Optional[List[Point]]:
        """
        Find a path from start to goal using A* search.
        Returns a list of points forming the path, or None if no path exists.
        
        Args:
            start: Starting point
            goal: Goal point
            debug: Whether to print debug info
            on_node_expansion: Optional callback when a node is expanded (for benchmarking)
        """
        # Edge case: start and goal are the same
        if start == goal:
            return [start]
        
        # Edge case: start or goal is blocked
        if not self.can_move_to(start) or not self.can_move_to(goal):
            if debug:
                print("Start or goal is blocked")
            return None
        
        if debug:
            print(f"Starting A* search from {start} to {goal}")
        
        # Initialize A* search
        open_set = []  # Priority queue for nodes to explore
        closed_set = set()  # Set of explored nodes
        
        # Dictionary to track g_scores (cost from start)
        g_scores = {start: 0.0}
        
        # Dictionary to track parent nodes (for path reconstruction)
        parents = {start: None}
        
        # Add start node to open set
        f_score = self.heuristic(start, goal)
        heapq.heappush(open_set, SearchNode(f_score, 0.0, start))
        
        # A* search
        while open_set:
            # Get node with lowest f_score
            current = heapq.heappop(open_set)
            current_pos = current.position
            
            if debug:
                print(f"Processing node {current_pos} with f_score={current.f_score}")
            
            # Skip if already processed
            if current_pos in closed_set:
                continue
            
            # Call benchmark callback if provided
            if on_node_expansion:
                on_node_expansion(current)
            
            # Goal reached, reconstruct path
            if current_pos == goal:
                if debug:
                    print("Goal reached")
                path = []
                while current_pos:
                    path.append(current_pos)
                    current_pos = parents.get(current_pos)
                return list(reversed(path))
            
            # Mark as explored
            closed_set.add(current_pos)
            
            # Check all neighbors
            for neighbor, cost in self.get_neighbors(current_pos):
                if neighbor in closed_set:
                    continue
                
                # Calculate tentative g_score
                tentative_g = g_scores[current_pos] + cost
                
                # If we found a better path to neighbor
                if neighbor not in g_scores or tentative_g < g_scores[neighbor]:
                    # Update path and scores
                    parents[neighbor] = current_pos
                    g_scores[neighbor] = tentative_g
                    f_score = tentative_g + self.heuristic(neighbor, goal)
                    
                    if debug:
                        print(f"  Adding {neighbor} with f_score={f_score}")
                    
                    # Add to open set
                    heapq.heappush(open_set, SearchNode(f_score, tentative_g, neighbor))
        
        if debug:
            print("No path found")
        return None
    
    def complete_path(self, waypoints: List[Point]) -> List[Point]:
        """
        Convert a list of waypoints to a complete path with all intermediate steps.
        This is useful for visualizing paths with one step per grid cell.
        """
        if not waypoints or len(waypoints) < 2:
            return waypoints
        
        result = [waypoints[0]]
        
        for i in range(1, len(waypoints)):
            current = waypoints[i-1]
            next_point = waypoints[i]
            
            # Calculate steps between current and next
            dx = next_point.x - current.x
            dy = next_point.y - current.y
            
            # Determine number of steps (max of dx and dy for diagonal movement)
            steps = max(abs(dx), abs(dy))
            
            for j in range(1, steps):
                # Calculate intermediate point
                if abs(dx) > abs(dy):
                    # More horizontal movement
                    x = current.x + j * (1 if dx > 0 else -1)
                    y = current.y + j * dy // abs(dx) if dx != 0 else current.y
                else:
                    # More vertical movement
                    y = current.y + j * (1 if dy > 0 else -1)
                    x = current.x + j * dx // abs(dy) if dy != 0 else current.x
                
                result.append(Point(x, y))
            
            # Add the destination point
            result.append(next_point)
        
        return result
    
    def __str__(self) -> str:
        """String representation of the grid."""
        result = ""
        for y in range(self.height):
            for x in range(self.width):
                if self.grid[y][x]:
                    result += "# "  # Blocked
                else:
                    result += ". "  # Passable
            result += "\n"
        return result


# Example usage
if __name__ == "__main__":
    # Create a 10x10 grid with some obstacles
    grid = AStar(10, 10)
    
    # Set up some obstacles
    for i in range(1, 8):
        grid.set(i, 5, True)  # Horizontal wall
    
    for i in range(5, 8):
        grid.set(3, i, True)  # Vertical wall
    
    start = Point(1, 1)
    goal = Point(5, 9)
    
    print("Grid:")
    print(grid)
    
    # Find path
    path = grid.find_path(start, goal, debug=True)
    
    if path:
        print("\nPath found:")
        for point in path:
            print(f"  {point}")
        
        # Visualize the path
        visual_grid = [[' ' for _ in range(grid.width)] for _ in range(grid.height)]
        
        # Mark obstacles
        for y in range(grid.height):
            for x in range(grid.width):
                if grid.grid[y][x]:
                    visual_grid[y][x] = '#'
        
        # Mark path
        complete_path = grid.complete_path(path)
        for i, point in enumerate(complete_path):
            if i == 0:
                visual_grid[point.y][point.x] = 'S'  # Start
            elif i == len(complete_path) - 1:
                visual_grid[point.y][point.x] = 'G'  # Goal
            else:
                visual_grid[point.y][point.x] = '*'  # Path
        
        print("\nPath visualization:")
        for row in visual_grid:
            print("  " + "".join(row))
    else:
        print("No path found") 