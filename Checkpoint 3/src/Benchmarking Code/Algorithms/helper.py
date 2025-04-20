from dataclasses import dataclass
from typing import List, Any, Optional

# This is a placeholder for the path to the directory where the benchmarking files are located.
path_dir = "../../../Checkpoint 3//tests//"

def clamp(value: int, min_val: int, max_val: int) -> int:
        return max(min_val, min(max_val, value))

@dataclass
class Point:
    """
    A simple 2D point with x, y coordinates.
    """
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y
    
    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)
        
    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)
    
    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __hash__(self) -> int:
        return hash((self.x, self.y))
    
    def __repr__(self) -> str:
        return f"Point({self.x}, {self.y})"
    
    def manhattan_distance(self, other: 'Point') -> int:
        """Calculate Manhattan distance between two points."""
        return abs(self.x - other.x) + abs(self.y - other.y)
    
    def euclidean_distance(self, other: 'Point') -> float:
        """Calculate Euclidean distance between two points."""
        return ((self.x - other.x) ** 2 + (self.y - other.y) ** 2) ** 0.5
    
    def cardinal_neighbors(self) -> List['Point']:
        """Get the 4 cardinal neighbors (N, E, S, W)."""
        return [
            Point(self.x, self.y - 1),  # North
            Point(self.x + 1, self.y),  # East
            Point(self.x, self.y + 1),  # South
            Point(self.x - 1, self.y),  # West
        ]
    
    def all_neighbors(self) -> List['Point']:
        """Get all 8 neighbors (cardinal + diagonal)."""
        return [
            Point(self.x, self.y - 1),      # North
            Point(self.x + 1, self.y - 1),  # Northeast
            Point(self.x + 1, self.y),      # East
            Point(self.x + 1, self.y + 1),  # Southeast
            Point(self.x, self.y + 1),      # South
            Point(self.x - 1, self.y + 1),  # Southwest
            Point(self.x - 1, self.y),      # West
            Point(self.x - 1, self.y - 1),  # Northwest
        ]
    
    def direction_to(self, other: 'Point') -> 'Point':
        """Get unit direction from this point to another."""
        dx = clamp(other.x - self.x, -1, 1)
        dy = clamp(other.y - self.y, -1, 1)
        return Point(dx, dy)
    
# JPS (Jump Point Search) specific function
class UnionFind:
    """
    Union-Find data structure for connected components.
    """
    def __init__(self, n: int):
        self.parent = list(range(n))
    
    def find(self, x: int) -> int:
        """Find the root of x with path compression."""
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        """Merge the sets containing x and y."""
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[ry] = rx
    
    def equiv(self, x: int, y: int) -> bool:
        """Check if x and y are in the same set."""
        return self.find(x) == self.find(y)
    
@dataclass(order=False)
class SearchNode:
    estimated_cost: float  # f = g + h
    cost: float  # g
    node: Point
    parent: Optional[Point] = None

    def __post_init__(self):
        self.f_score = self.estimated_cost
        self.g_score = self.cost
        self.position = self.node

    def __lt__(self, other: 'SearchNode') -> bool:
        if self.estimated_cost == other.estimated_cost:
            return self.cost > other.cost  # Tie-breaker
        return self.estimated_cost < other.estimated_cost
