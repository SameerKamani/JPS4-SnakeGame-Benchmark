import heapq
from collections import deque
from typing import Any, Callable, List, Optional, Tuple, Dict

# Constants (using values from the Rust code)
EQUAL_EDGE_COST = False
GRAPH_PRUNING = True
N_SMALLVEC_SIZE = 8

if EQUAL_EDGE_COST:
    D = 1
    C = 1
else:
    D = 99
    C = 70
E = 2 * C - D  # = 41

############################
# Basic data structures
############################

class Point:
    """A simple 2D point."""
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y

    def __add__(self, other: 'Point') -> 'Point':
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: 'Point') -> 'Point':
        return Point(self.x - other.x, self.y - other.y)

    def __eq__(self, other: Any) -> bool:
        return isinstance(other, Point) and self.x == other.x and self.y == other.y

    def __hash__(self) -> int:
        return hash((self.x, self.y))

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

    def manhattan_distance(self, other: 'Point') -> int:
        return abs(self.x - other.x) + abs(self.y - other.y)

    def moore_neighborhood(self) -> List['Point']:
        # 8-connected neighbors (Moore neighborhood)
        directions = [
            Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
            Point(0, 1), Point(-1, 1), Point(-1, 0), Point(-1, -1)
        ]
        return [self + d for d in directions]

    def neumann_neighborhood(self) -> List['Point']:
        # 4-connected neighbors (Von Neumann neighborhood)
        directions = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
        return [self + d for d in directions]

    def move_distance(self, other: 'Point') -> float:
        # Here we use Manhattan distance as a proxy for “distance”
        return self.manhattan_distance(other)

    def direction_to(self, other: 'Point') -> 'Point':
        # Return a unit step from self to other (cardinal or diagonal)
        dx = other.x - self.x
        dy = other.y - self.y
        # Normalize to at most 1 step in each axis
        if dx != 0:
            dx = dx // abs(dx)
        if dy != 0:
            dy = dy // abs(dy)
        return Point(dx, dy)

    def neighbor_in_direction(self, d: int) -> 'Point':
        # Return neighbor in one of 8 directions.
        # We'll assume directions as follows:
        # 0: up, 1: up-right, 2: right, 3: down-right,
        # 4: down, 5: down-left, 6: left, 7: up-left
        mapping = [
            Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1),
            Point(0, 1), Point(-1, 1), Point(-1, 0), Point(-1, -1)
        ]
        return self + mapping[d % 8]

############################
# Union-Find (Disjoint Set)
############################

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

############################
# A* Search Context for JPS
############################

class SearchNode:
    def __init__(self, estimated_cost: int, cost: int, index: int):
        self.estimated_cost = estimated_cost
        self.cost = cost
        self.index = index

    def __lt__(self, other: 'SearchNode') -> bool:
        # Lower estimated_cost has higher priority; tie-break on cost
        if self.estimated_cost == other.estimated_cost:
            return self.cost < other.cost
        return self.estimated_cost < other.estimated_cost

    def __repr__(self):
        return f"SearchNode(est={self.estimated_cost}, cost={self.cost}, idx={self.index})"

class AstarContext:
    """
    Maintains the fringe (priority queue) and a parent mapping.
    In Python, we use a list for the fringe with heapq.
    The parents dictionary maps nodes to a tuple (parent_index, cost).
    """
    def __init__(self):
        self.fringe: List[SearchNode] = []
        self.parents: Dict[Any, Tuple[Optional[Any], int]] = {}  # key: node, value: (parent, cost)

    def astar_jps(
        self,
        start: Any,
        successors: Callable[[Optional[Any], Any], List[Tuple[Any, int]]],
        heuristic: Callable[[Any], int],
        success: Callable[[Any], bool],
    ) -> Optional[Tuple[List[Any], int]]:
        self.fringe.clear()
        self.parents.clear()

        # Initialize start node; we use the start itself as key.
        self.parents[start] = (None, 0)
        heapq.heappush(self.fringe, SearchNode(heuristic(start), 0, 0))
        
        while self.fringe:
            current = heapq.heappop(self.fringe)
            # Find the corresponding node by iterating parents (in Python we use keys directly)
            # Here, we store the node itself as key.
            # For simplicity, we iterate through our parents dict to get the node with matching cost.
            # (A more efficient implementation would store (node, ...) in the SearchNode.)
            current_node = None
            for node, (p, cost) in self.parents.items():
                if cost == current.cost:
                    current_node = node
                    break
            if current_node is None:
                continue

            if success(current_node):
                # Reconstruct path by following parent pointers.
                path = []
                node = current_node
                while node is not None:
                    path.append(node)
                    node = self.parents[node][0]
                path.reverse()
                return (path, current.cost)

            # Get parent's node if available
            parent_node = None
            # (In this version, we simply pass the current node as parent to the successor function.)
            succs = successors(parent_node, current_node)
            for succ, move_cost in succs:
                new_cost = current.cost + move_cost
                # If this node is new or we've found a better path, update.
                if succ not in self.parents or self.parents[succ][1] > new_cost:
                    self.parents[succ] = (current_node, new_cost)
                    est = new_cost + heuristic(succ)
                    heapq.heappush(self.fringe, SearchNode(est, new_cost, 0))
        print("Warning: No reachable goal found.")
        return None

def astar_jps(
    start: Any,
    successors: Callable[[Optional[Any], Any], List[Tuple[Any, int]]],
    heuristic: Callable[[Any], int],
    success: Callable[[Any], bool],
) -> Optional[Tuple[List[Any], int]]:
    context = AstarContext()
    return context.astar_jps(start, successors, heuristic, success)

############################
# PathingGrid and related functionality
############################

class PathingGrid:
    """
    A grid-based pathfinding system.
    - grid: 2D list of booleans (True = blocked, False = free)
    - neighbours: 2D list of ints acting as a bitmask (for available neighbor directions)
    - components: UnionFind structure for connected components.
    """
    def __init__(self, width: int, height: int, default_value: bool):
        self.width = width
        self.height = height
        self.grid: List[List[bool]] = [[default_value for _ in range(width)] for _ in range(height)]
        # Initialize all neighbor masks to 0b11111111 (all directions available)
        self.neighbours: List[List[int]] = [[0xFF for _ in range(width)] for _ in range(height)]
        self.components = UnionFind(width * height)
        self.components_dirty = False
        self.heuristic_factor = 1.0
        self.improved_pruning = True
        self.allow_diagonal_move = True
        self.context = AstarContext()

        # Emulate border obstacles by updating neighbours around the grid border.
        for i in range(-1, width + 1):
            self.update_neighbours(i, -1, True)
            self.update_neighbours(i, height, True)
        for j in range(-1, height + 1):
            self.update_neighbours(-1, j, True)
            self.update_neighbours(width, j, True)

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.width and 0 <= y < self.height

    def get(self, x: int, y: int) -> bool:
        return self.grid[y][x]

    def set(self, x: int, y: int, blocked: bool):
        p = Point(x, y)
        if self.grid[y][x] != blocked and blocked:
            self.components_dirty = True
        else:
            # For simplicity, union the point with its free neighbors.
            idx = y * self.width + x
            for np in self.neighborhood_points(p):
                if self.can_move_to(np):
                    nidx = np.y * self.width + np.x
                    self.components.union(idx, nidx)
        self.update_neighbours(x, y, blocked)
        self.grid[y][x] = blocked

    def neighborhood_points(self, point: Point) -> List[Point]:
        if self.allow_diagonal_move:
            return point.moore_neighborhood()
        else:
            return point.neumann_neighborhood()

    def neighborhood_points_and_cost(self, pos: Point) -> List[Tuple[Point, int]]:
        result = []
        for p in self.neighborhood_points(pos):
            if self.can_move_to(p):
                # Cost: if move is diagonal then cost = D; else cost = C.
                delta = pos.direction_to(p)
                # Check if move is diagonal:
                if abs(delta.x) == 1 and abs(delta.y) == 1:
                    cost = D
                else:
                    cost = C
                result.append((p, cost))
        return result

    def heuristic(self, p1: Point, p2: Point) -> int:
        if self.allow_diagonal_move:
            dx = abs(p1.x - p2.x)
            dy = abs(p1.y - p2.y)
            return (E * abs(dx - dy) + D * (dx + dy)) // 2
        else:
            return p1.manhattan_distance(p2) * C

    def can_move_to(self, pos: Point) -> bool:
        return self.in_bounds(pos.x, pos.y) and (not self.grid[pos.y][pos.x])

    def update_neighbours(self, x: int, y: int, blocked: bool):
        # Update neighbour bitmask for cell (x,y) in the grid (if within bounds).
        p = Point(x, y)
        for i in range(8):
            np = p.neighbor_in_direction(i)
            if self.in_bounds(np.x, np.y):
                # In this simplified version, we set or clear the bit for the opposite direction.
                opposite = (i + 4) % 8
                # Update neighbour mask at np accordingly.
                mask = self.neighbours[np.y][np.x]
                if blocked:
                    mask &= ~(1 << opposite)
                else:
                    mask |= (1 << opposite)
                self.neighbours[np.y][np.x] = mask

    def is_forced(self, direction: Point, node: Point) -> bool:
        # A simplified forced neighbor check: if a neighbour in a perpendicular direction is blocked.
        # For demonstration, we simply check one adjacent cell.
        # (The actual logic in Rust uses bitmask lookups.)
        perp = Point(-direction.y, direction.x)  # perpendicular vector
        check = node + perp
        return not self.can_move_to(check)

    def jump_straight(self, initial: Point, cost: int, direction: Point, goal: Callable[[Point], bool]) -> Optional[Tuple[Point, int]]:
        # Only for cardinal moves (non-diagonal)
        while True:
            initial = initial + direction
            if not self.can_move_to(initial):
                return None
            if goal(initial) or self.is_forced(direction, initial):
                return (initial, cost)
            cost += C

    def jump(self, initial: Point, cost: int, direction: Point, goal: Callable[[Point], bool]) -> Optional[Tuple[Point, int]]:
        while True:
            initial = initial + direction
            if not self.can_move_to(initial):
                return None
            if goal(initial) or self.is_forced(direction, initial):
                return (initial, cost)
            # For diagonal moves, also check for horizontal/vertical jump possibilities.
            if abs(direction.x) == 1 and abs(direction.y) == 1:
                if (self.jump_straight(initial, 1, Point(direction.x, 0), goal) is not None or 
                    self.jump_straight(initial, 1, Point(0, direction.y), goal) is not None):
                    return (initial, cost)
            cost += (D - C if abs(direction.x) == 1 and abs(direction.y) == 1 else 0) + C

    def jps_neighbours(self, parent: Optional[Point], node: Point, goal: Callable[[Point], bool]) -> List[Tuple[Point, int]]:
        succ = []
        if parent is not None:
            # Determine movement direction from parent to node.
            direction = parent.direction_to(node)
            # In a simplified version, get the pruned neighborhood.
            for nbr, c in self.neighborhood_points_and_cost(node):
                d = node.direction_to(nbr)
                res = self.jump(node, c, d, goal)
                if res is not None:
                    jumped_node, jump_cost = res
                    # If improved pruning is enabled and move is diagonal and not forced:
                    if self.improved_pruning and abs(d.x) == 1 and abs(d.y) == 1 and not goal(jumped_node) and not self.is_forced(d, jumped_node):
                        # For simplicity, we do not recursively expand in this example.
                        succ.append((jumped_node, jump_cost))
                    else:
                        succ.append((jumped_node, jump_cost))
        else:
            succ = self.neighborhood_points_and_cost(node)
        return succ

    def get_ix_point(self, p: Point) -> int:
        return p.y * self.width + p.x

    def generate_components(self):
        # Recompute connected components via UnionFind.
        self.components = UnionFind(self.width * self.height)
        self.components_dirty = False
        for y in range(self.height):
            for x in range(self.width):
                if not self.grid[y][x]:
                    idx = y * self.width + x
                    for nbr in self.neighborhood_points(Point(x, y)):
                        if self.in_bounds(nbr.x, nbr.y) and not self.grid[nbr.y][nbr.x]:
                            nidx = nbr.y * self.width + nbr.x
                            self.components.union(idx, nidx)

    def reachable(self, start: Point, goal: Point) -> bool:
        if self.in_bounds(start.x, start.y) and self.in_bounds(goal.x, goal.y):
            return self.components.equiv(self.get_ix_point(start), self.get_ix_point(goal))
        return False

    def get_waypoints_single_goal(self, start: Point, goal: Point, approximate: bool) -> Optional[List[Point]]:
        # For approximate pathing, check if a neighbor of goal is reachable.
        if approximate:
            # A simplified check: if goal is not reachable from start directly.
            if not self.reachable(start, goal):
                return None
            # Otherwise, use astar_jps with a relaxed goal condition.
            result = self.context.astar_jps(
                start,
                lambda parent, node: self.jps_neighbours(parent, node, lambda p: self.heuristic(p, goal) <= (1 if EQUAL_EDGE_COST else 99)),
                lambda point: int(self.heuristic(point, goal) * self.heuristic_factor),
                lambda point: self.heuristic(point, goal) <= (1 if EQUAL_EDGE_COST else 99)
            )
        else:
            if not self.reachable(start, goal):
                return None
            result = self.context.astar_jps(
                start,
                lambda parent, node: self.jps_neighbours(parent, node, lambda p: p == goal),
                lambda point: int(self.heuristic(point, goal) * self.heuristic_factor),
                lambda point: point == goal
            )
        if result is None:
            return None
        (path, cost) = result
        return path

    def get_path_single_goal(self, start: Point, goal: Point, approximate: bool) -> Optional[List[Point]]:
        waypoints = self.get_waypoints_single_goal(start, goal, approximate)
        if waypoints is None:
            return None
        return self.waypoints_to_path(waypoints)

    def waypoints_to_path(self, waypoints: List[Point]) -> List[Point]:
        if not waypoints:
            return []
        path = [waypoints[0]]
        for next_pt in waypoints[1:]:
            current = path[-1]
            while current.move_distance(next_pt) >= 1:
                delta = current.direction_to(next_pt)
                current = current + delta
                path.append(current)
        return path

    def __str__(self):
        s = "Grid:\n"
        for row in self.grid:
            s += " ".join("1" if cell else "0" for cell in row) + "\n"
        s += "Neighbours:\n"
        for row in self.neighbours:
            s += " ".join(f"{cell:08b}" for cell in row) + "\n"
        return s

############################
# Testing (if run as main)
############################

if __name__ == '__main__':
    # A simple test: 3x3 grid with center blocked, path from (0,0) to (2,2)
    pg = PathingGrid(3, 3, False)
    pg.set(1, 1, True)
    pg.generate_components()
    start = Point(0, 0)
    end = Point(2, 2)
    path = pg.get_path_single_goal(start, end, approximate=False)
    print("Path:", path)
