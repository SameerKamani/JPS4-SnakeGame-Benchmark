from typing import List, Any, Optional, Tuple, Dict, Callable
import heapq

class Point:
    """
    Point class representing a 2D point with x and y coordinates.
    """
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def manhattan_distance(self, other: 'Point') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def neumann_neighborhood(self) -> List['Point']:
        # Cardinal (4-connected) moves only.
        directions = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return [self + d for d in directions]

    def direction_to(self, other: 'Point') -> 'Point':
        dx = other.x - self.x
        dy = other.y - self.y
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        return Point(dx, dy)

# --- Union-Find ---
class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
    
    def find(self, x: int) -> int:
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x: int, y: int) -> None:
        rx = self.find(x)
        ry = self.find(y)
        if rx != ry:
            self.parent[ry] = rx
    
    def equiv(self, x: int, y: int) -> bool:
        return self.find(x) == self.find(y)

# --- A* Search Context for JPS ---
class SearchNode:
    def __init__(self, estimated_cost: int, cost: int, node: Any, parent: Optional[Any]):
        self.estimated_cost = estimated_cost
        self.cost = cost
        self.node = node
        self.parent = parent

    def __lt__(self, other: 'SearchNode') -> bool:
        if self.estimated_cost == other.estimated_cost:
            return self.cost < other.cost
        return self.estimated_cost < other.estimated_cost

class AstarContext:
    def __init__(self):
        self.fringe: List[SearchNode] = []
        # Map each node to a tuple (parent, cost)
        self.parents: Dict[Any, Tuple[Optional[Any], int]] = {}

    def astar_jps(
        self,
        start: Any,
        successors: Callable[[Optional[Any], Any], List[Tuple[Any, int]]],
        heuristic: Callable[[Any], int],
        success: Callable[[Any], bool],
    ) -> Optional[Tuple[List[Any], int]]:
        self.fringe.clear()
        self.parents.clear()
        self.parents[start] = (None, 0)
        heapq.heappush(self.fringe, SearchNode(heuristic(start), 0, start, None))
        
        while self.fringe:
            current = heapq.heappop(self.fringe)
            if success(current.node):
                path = []
                node = current.node
                while node is not None:
                    path.append(node)
                    node = self.parents[node][0]
                path.reverse()
                return (path, current.cost)
            for succ, move_cost in successors(self.parents[current.node][0], current.node):
                new_cost = current.cost + move_cost
                if succ not in self.parents or self.parents[succ][1] > new_cost:
                    self.parents[succ] = (current.node, new_cost)
                    heapq.heappush(self.fringe, SearchNode(new_cost + heuristic(succ), new_cost, succ, current.node))
        return None
