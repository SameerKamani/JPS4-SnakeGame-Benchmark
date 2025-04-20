from helper import Point, AstarContext
from pathing_grid import PathingGrid

from typing import Any, Callable, List, Optional, Tuple
from tkinter import *
from random import choice

import numpy as np
import time
import winsound

path_dir = "Checkpoint 3//src//Snake Game Code//"

def astar_jps(start: Any,
              successors: Callable[[Optional[Any], Any], List[Tuple[Any, int]]],
              heuristic: Callable[[Any], int],
              success: Callable[[Any], bool]) -> Optional[Tuple[List[Any], int]]:

    context = AstarContext()
    return context.astar_jps(start, successors, heuristic, success)


############################
# Snake Game Integration
############################

# Colors and cell types
COLORS = ['white', 'maroon', 'red', 'yellow', 'grey', 'white']
EMPTY = 0
BODY = 1
FOOD = 2
HEAD = 3
WALL = 4

# Define two boards: EASY and HARD
EASY_BOARD_LAYOUT = [
    [4]*20,
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4,0,0,0,4,4,4,0,0,0,4,0,0,0,0,0,0,0,0,4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4,0,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4,0,0,0,0,0,4,0,0,0,0,0,4,0,0,0,0,0,0,4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4] + [0]*18 + [4],
    [4,0,0,4,4,4,0,0,0,0,0,0,0,0,0,0,4,0,0,4],
    [4] + [0]*18 + [4],
    [4]*20
]

HARD_BOARD_LAYOUT = [
    [4]*20,
    [4, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 0, 4, 0, 4, 0, 4],
    [4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 4, 0, 0, 0, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 4],
    [4, 0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 4, 4, 0, 0, 0, 4, 0, 0, 4],
    [4, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 0, 0, 0, 4, 0, 4, 0, 0, 4],
    [4, 0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 4],
    [4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 0, 0, 4, 0, 0, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 0, 0, 4, 0, 4, 0, 4, 0, 4, 0, 0, 4],
    [4, 0, 0, 0, 4, 4, 0, 0, 0, 4, 0, 0, 0, 4, 4, 0, 0, 0, 0, 4],
    [4, 0, 4, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 4],
    [4, 0, 0, 0, 0, 0, 0, 0, 4, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 4],
    [4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 4],
    [4, 0, 4, 0, 0, 0, 4, 0, 0, 0, 4, 0, 4, 0, 0, 0, 4, 0, 4, 4],
    [4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4],
    [4, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 0, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 4],
    [4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 0, 4, 4],
    [4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, 0, 4],
    [4]*20
]

# Global references for the board and snake
BOARD = None
main_snake = None

# Place the initial food at a random free cell.
food = (10, 10)

# Direction functions (using (row, col))
LEFT = lambda pos: (pos[0], pos[1] - 1)
RIGHT = lambda pos: (pos[0], pos[1] + 1)
UP = lambda pos: (pos[0] - 1, pos[1])
DOWN = lambda pos: (pos[0] + 1, pos[1])

CELL_WIDTH = 32
win_width = 20 * CELL_WIDTH
win_height = 20 * CELL_WIDTH

# Default snake spawn in a free area
default_location = [(2,2), (2,3)]

def update_pathing_grid(pg: PathingGrid, board: np.ndarray):
    height, width = board.shape
    for row in range(height):
        for col in range(width):
            # Treat WALL and BODY as blocked; HEAD and FOOD remain free.
            blocked = (board[row, col] in [WALL, BODY])
            pg.set(col, row, blocked)

class Snake:
    def __init__(self, board: np.ndarray, locations=default_location, virtual=False):
        self.locations = locations  # list of (row, col); head is first element
        self.board = board          # shared board (numpy array)
        self.virtual = virtual
        self.score = 0
        self.no_path_count = 0  # Count of consecutive turns with no valid path

    def play(self):
        # Start the music
        winsound.PlaySound(f"{path_dir}Assets/roku_snake.wav", winsound.SND_LOOP + winsound.SND_ASYNC)
        while self.alive():
            movement = self.food_search()
            if movement is None:
                self.no_path_count += 1
                print("No path found to food; attempting emergency movement.")
                # Try emergency movement to avoid getting trapped
                movement = self.find_safest_move()
                if movement is None:
                    print("No safe moves available; skipping move this turn.")
                    time.sleep(0.3)
                    if self.no_path_count >= 5:
                        break
                    continue
                else:
                    print(f"Emergency move chosen: {movement.__name__}")
            else:
                self.no_path_count = 0
            self.change_positions(movement)
            time.sleep(0.2)
        self.game_over()

    def food_search(self):
        """Find a path to the food using JPS4 pathfinding."""
        # Get the current state
        head_loc = self.head()
        tail_loc = self.tail()
        
        # Create a PathingGrid
        pg = PathingGrid(self.board.shape[1], self.board.shape[0], False)
        
        # Temporarily remove tail so it doesn't block the path
        temp_board = self.board.copy()
        temp_board[tail_loc] = EMPTY
        
        # Update the pathing grid
        update_pathing_grid(pg, temp_board)
        
        # Convert locations to Point objects
        start = Point(head_loc[1], head_loc[0])  # Convert (row, col) to (x, y)
        goal_pt = Point(food[1], food[0])        # Convert (row, col) to (x, y)
        
        # First, check for a direct path to food (fastest option)
        direct_path = pg.find_direct_path(start, goal_pt)
        if direct_path and len(direct_path) > 1:
            print(f"\nDEBUG: Direct path to food: {direct_path}")
            next_pt = direct_path[1]
            new_pos = (next_pt.y, next_pt.x)  # Convert back to (row, col)
            return self.get_direction(head_loc, new_pos)
        
        # Try to find a path using JPS4
        print(f"\nDEBUG: Snake is at {head_loc}, food is at {food}")
        path = pg.get_path_single_goal(start, goal_pt, mode="jps4")
        
        # If no path was found, return None
        if not path or len(path) < 2:
            print("DEBUG: No path found to food")
            return None
            
        # Print the path for debugging
        print(f"DEBUG: Path found to food: {path}")
        
        # The next step is the second point in the path (first point is current position)
        next_pt = path[1]
        new_pos = (next_pt.y, next_pt.x)  # Convert back to (row, col)
        
        # Validate the next position to ensure it's a legal move
        if (not 0 <= new_pos[0] < self.board.shape[0] or 
            not 0 <= new_pos[1] < self.board.shape[1] or
            self.board[new_pos] not in [EMPTY, FOOD]):
            print(f"WARNING: Pathfinding suggested illegal move to {new_pos}. Falling back to emergency movement.")
            return None
            
        return self.get_direction(head_loc, new_pos)
        
    def get_direction(self, current_pos: Tuple[int, int], next_pos: Tuple[int, int]):
        """Determine which direction to move based on current and next positions."""
        if next_pos[0] < current_pos[0]:  # Moving up
            return UP
        elif next_pos[0] > current_pos[0]:  # Moving down
            return DOWN
        elif next_pos[1] < current_pos[1]:  # Moving left
            return LEFT
        elif next_pos[1] > current_pos[1]:  # Moving right
            return RIGHT
        
        # Should never reach here
        print(f"ERROR: Unable to determine direction from {current_pos} to {next_pos}")
        return None

    def validate_path(self, path: List[Point]) -> Optional[List[Point]]:
        """Ensure the path is valid for the snake by checking each step"""
        if not path or len(path) < 2:
            return None
            
        # Start from the head position
        head_loc = self.head()
        current_pos = (head_loc[0], head_loc[1])
        
        # Create a copy of the board to simulate movement
        sim_board = self.board.copy()
        
        # Build a new path starting from the current position
        valid_path = [path[0]]  # First point is the start (head position)
        
        for i in range(1, len(path)):
            next_pt = path[i]
            next_pos = (next_pt.y, next_pt.x)
            
            # Check if this is a valid move (adjacent and not blocked)
            is_adjacent = False
            for d in [LEFT, RIGHT, UP, DOWN]:
                if d(current_pos) == next_pos:
                    is_adjacent = True
                    break
                    
            if not is_adjacent:
                print(f"WARNING: Path contains non-adjacent move from {current_pos} to {next_pos}")
                return None
                
            # Check if the position is valid (not blocked)
            if (not 0 <= next_pos[0] < sim_board.shape[0] or 
                not 0 <= next_pos[1] < sim_board.shape[1] or
                (sim_board[next_pos] not in [EMPTY, FOOD] and next_pos != self.tail())):
                print(f"WARNING: Path contains blocked position at {next_pos}")
                return None
                
            # Add this point to the validated path
            valid_path.append(next_pt)
            
            # Update for next iteration
            current_pos = next_pos
            
        return valid_path

    def find_safest_move(self):
        """When no path to food exists, find a move that keeps the most future options open."""
        head_loc = self.head()
        # Get only VALID possible moves
        possible_moves = []
        for d in [LEFT, RIGHT, UP, DOWN]:
            new_loc = d(head_loc)
            if (0 <= new_loc[0] < self.board.shape[0] and 
                0 <= new_loc[1] < self.board.shape[1] and
                self.board[new_loc] in [EMPTY, FOOD]):
                possible_moves.append(d)
        
        if not possible_moves:
            return None
            
        # Evaluate each move with a more sophisticated scoring system
        best_move = None
        best_score = -1
        
        for move in possible_moves:
            # Simulate the move
            new_head = move(head_loc)
            
            # Double-check this is a valid move (should always be true due to the filtering above)
            if not (0 <= new_head[0] < self.board.shape[0] and 
                   0 <= new_head[1] < self.board.shape[1] and
                   self.board[new_head] in [EMPTY, FOOD]):
                print(f"WARNING: Skipping invalid move to {new_head}")
                continue
            
            # 1. Count immediate moves available (basic freedom)
            future_moves_count = 0
            future_moves = []
            for d in [LEFT, RIGHT, UP, DOWN]:
                future_pos = d(new_head)
                if 0 <= future_pos[0] < self.board.shape[0] and 0 <= future_pos[1] < self.board.shape[1]:
                    # Check if this position would be valid after the move
                    # (excluding the current tail which will move)
                    if future_pos != self.tail() and self.board[future_pos] in [EMPTY, FOOD]:
                        future_moves_count += 1
                        future_moves.append(future_pos)
            
            # 2. Evaluate open space - avoid corners and narrow corridors
            # Count empty spaces in a larger area around the new position
            open_space_score = 0
            visited = set()
            if future_moves_count > 0:  # Only bother if there's at least one immediate move
                # Use a simple flood fill to measure open space
                to_visit = [new_head]
                visited.add(new_head)
                depth = 0
                max_depth = 6  # Limit search depth to avoid too much computation
                
                while to_visit and depth < max_depth:
                    depth += 1
                    next_to_visit = []
                    for pos in to_visit:
                        for d in [LEFT, RIGHT, UP, DOWN]:
                            neighbor = d(pos)
                            if (neighbor not in visited and 
                                0 <= neighbor[0] < self.board.shape[0] and 
                                0 <= neighbor[1] < self.board.shape[1] and
                                self.board[neighbor] in [EMPTY, FOOD]):
                                visited.add(neighbor)
                                next_to_visit.append(neighbor)
                                # Give higher score to spaces found earlier in the search
                                open_space_score += (max_depth - depth + 1)
                    to_visit = next_to_visit
            
            # 3. Prefer moves away from walls and the snake's body
            wall_distance = 0
            # Check distance to walls/obstacles in each direction
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue  # Skip the center
                    
                    # Look outward in this direction
                    r, c = new_head
                    distance = 0
                    while (0 <= r < self.board.shape[0] and 
                           0 <= c < self.board.shape[1] and
                           self.board[(r, c)] in [EMPTY, FOOD, HEAD] and
                           distance < 4):  # Limit distance check
                        distance += 1
                        r += dr
                        c += dc
                    
                    wall_distance += distance
            
            # Calculate overall score with weighted components
            # Weight immediate freedom more heavily
            score = (future_moves_count * 10) + (open_space_score * 2) + wall_distance
            
            # Debug output
            print(f"Evaluating move {move.__name__}: future_moves={future_moves_count}, open_space={open_space_score}, wall_distance={wall_distance}, total_score={score}")
            
            if score > best_score:
                best_score = score
                best_move = move
                
        return best_move

    def change_positions(self, direction: Callable[[Tuple[int, int]], Tuple[int, int]]):
        head_loc = self.head()
        new_head = direction(head_loc)
        
        # Safety check: make sure we're not moving into a wall or our own body
        if (not 0 <= new_head[0] < self.board.shape[0] or 
            not 0 <= new_head[1] < self.board.shape[1] or
            self.board[new_head] in [WALL, BODY]):
            print(f"ERROR: Attempted illegal move to {new_head} which contains: {self.board[new_head] if 0 <= new_head[0] < self.board.shape[0] and 0 <= new_head[1] < self.board.shape[1] else 'out of bounds'}")
            # Choose a random legal move instead
            legal_moves = self.possible_moves_list(head_loc)
            if legal_moves:
                return self.change_positions(legal_moves[0])
            else:
                # No legal moves left
                return
                
        tail_loc = self.tail()
        self.board[tail_loc] = EMPTY
        self.locations = [new_head] + self.locations[:-1]
        if new_head == food:
            self.locations.append(tail_loc)
            self.score += 1
            self.make_food()
        self.update_board()

    def make_food(self):
        global food
        free_cells = []
        for r in range(self.board.shape[0]):
            for c in range(self.board.shape[1]):
                if self.board[r, c] == EMPTY:
                    free_cells.append((r, c))
        if free_cells:
            food = choice(free_cells)
        else:
            food = None

    def alive(self):
        return len(self.possible_moves_list(self.head())) > 0

    def head(self):
        return self.locations[0]

    def tail(self):
        return self.locations[-1]

    def possible_moves_list(self, location):
        moves = []
        for d in [LEFT, RIGHT, UP, DOWN]:
            new_loc = d(location)
            if 0 <= new_loc[0] < self.board.shape[0] and 0 <= new_loc[1] < self.board.shape[1]:
                if self.board[new_loc] in [EMPTY, FOOD]:
                    moves.append(d)
        return moves

    def update_board(self):
        self.board[self.board == BODY] = EMPTY
        self.board[self.board == HEAD] = EMPTY
        for i, pos in enumerate(self.locations):
            if i == 0:
                self.board[pos] = HEAD
            else:
                self.board[pos] = BODY
        if food is not None:
            self.board[food] = FOOD
        if not self.virtual:
            self.update_canvas()

    def update_canvas(self):
        canvas.delete('all')
        canvas.config(width=win_width, height=win_height)
        for col in range(self.board.shape[1]):
            for row in range(self.board.shape[0]):
                x_val = col * CELL_WIDTH
                y_val = row * CELL_WIDTH
                cell_val = self.board[row, col]
                color = COLORS[cell_val]
                canvas.create_rectangle(x_val, y_val, x_val + CELL_WIDTH, y_val + CELL_WIDTH,
                                        fill=color, outline='white')
        canvas.update()

    def game_over(self):
        winsound.PlaySound(f"{path_dir}Assets/price.wav", winsound.SND_ASYNC)
        canvas.delete("all")
        canvas.create_text(win_width // 2, win_height // 2,
                           text=f"Game Over\nScore: {self.score}",
                           font=("Arial", 30), fill="red")
        canvas.update()
        print("Game Over! Final Score:", self.score)

############################
# Tkinter Setup
############################

banana = Tk()
banana.title('JPS4 Snake Game')
canvas = Canvas(banana, bg="black")
canvas.pack()

def start_menu():
    """Display a start menu with snake_game.png as a full-window background and two difficulty buttons."""
    # Hide the canvas so it doesn't appear at the top
    canvas.pack_forget()

    # Force the main window to the desired size
    banana.geometry(f"{win_width}x{win_height}")

    # Create a separate frame for the menu
    menu_frame = Frame(banana, width=win_width, height=win_height)
    menu_frame.pack(fill="both", expand=True)

    # Attempt to load the background image
    try:
        bg_image_original = PhotoImage(file=f"{path_dir}Assets/snake_game.png")
        img_width = bg_image_original.width()
        img_height = bg_image_original.height()

        # Compute how many times we can scale the image (integer factor only).
        scale_factor_x = win_width // img_width  if img_width  else 1
        scale_factor_y = win_height // img_height if img_height else 1
        # Pick the smaller of the two to ensure we don't exceed window boundaries.
        scale_factor = min(scale_factor_x, scale_factor_y)
        
        # If scale_factor is >= 1, we enlarge using .zoom().
        # If scale_factor is 0, we do a sub-sample using an integer factor (for smaller images).
        if scale_factor >= 1:
            bg_image = bg_image_original.zoom(scale_factor)
        else:
            # We'll invert the factor for sub-sampling (avoiding division by zero).
            # e.g., if scale_factor is 0, use at least 2 or more as sub-sample factor.
            subsample_factor = max(2, (img_width // win_width) + 1, (img_height // win_height) + 1)
            bg_image = bg_image_original.subsample(subsample_factor)
    except Exception as e:
        print("Could not load 'snake_game.png':", e)
        bg_image = None

    # If the image loaded, create a label to hold it as a background
    if bg_image:
        bg_label = Label(menu_frame, image=bg_image)
        bg_label.image = bg_image  # Keep a reference to avoid garbage collection
        bg_label.place(x=0, y=0, relwidth=1, relheight=1)  # Fill the frame
    else:
        # Fallback color if no image is available
        menu_frame.config(bg="lightgreen")

    # Functions for difficulty choices
    def choose_easy():
        menu_frame.destroy()
        # Now show the canvas
        canvas.pack(fill="both", expand=True)
        on_difficulty_chosen(EASY_BOARD_LAYOUT)

    def choose_hard():
        menu_frame.destroy()
        # Now show the canvas
        canvas.pack(fill="both", expand=True)
        on_difficulty_chosen(HARD_BOARD_LAYOUT)

    # Create buttons on top of the background
    easy_button = Button(menu_frame, text="Easy", font=("Arial", 16, "bold"), 
                         fg="white", bg="green", command=choose_easy)
    hard_button = Button(menu_frame, text="Hard", font=("Arial", 16, "bold"), 
                         fg="white", bg="red", command=choose_hard)

    # Place buttons at positions over the background
    # Adjust these relx/rely values to position them exactly where you want
    easy_button.place(relx=0.4, rely=0.9, anchor="center")
    hard_button.place(relx=0.6, rely=0.9, anchor="center")


def on_difficulty_chosen(board_layout):
    global BOARD, main_snake
    # Convert the chosen layout to a numpy array
    BOARD = np.array(board_layout)
    # Create the snake and start the game
    main_snake = Snake(BOARD.copy(), locations=default_location)
    main_snake.update_board()
    main_snake.play()

# Show the start menu first
start_menu()

banana.mainloop()
