# Remember to adjust your student ID in meta.xml
import numpy as np
import pickle
import random
import gym
from gym import spaces
import matplotlib.pyplot as plt
import copy
import random
import math


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()

        self.size = 4  # 4x4 2048 board
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0

        # Action space: 0: up, 1: down, 2: left, 3: right
        self.action_space = spaces.Discrete(4)
        self.actions = ["up", "down", "left", "right"]

        self.last_move_valid = True  # Record if the last move was valid

        self.reset()

    def reset(self):
        """Reset the environment"""
        self.board = np.zeros((self.size, self.size), dtype=int)
        self.score = 0
        self.add_random_tile()
        self.add_random_tile()
        return self.board

    def add_random_tile(self):
        """Add a random tile (2 or 4) to an empty cell"""
        empty_cells = list(zip(*np.where(self.board == 0)))
        if empty_cells:
            x, y = random.choice(empty_cells)
            self.board[x, y] = 2 if random.random() < 0.9 else 4

    def compress(self, row):
        """Compress the row: move non-zero values to the left"""
        new_row = row[row != 0]  # Remove zeros
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')  # Pad with zeros on the right
        return new_row

    def merge(self, row):
        """Merge adjacent equal numbers in the row"""
        for i in range(len(row) - 1):
            if row[i] == row[i + 1] and row[i] != 0:
                row[i] *= 2
                row[i + 1] = 0
                self.score += row[i]
        return row

    def move_left(self):
        """Move the board left"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            new_row = self.compress(self.board[i])
            new_row = self.merge(new_row)
            new_row = self.compress(new_row)
            self.board[i] = new_row
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_right(self):
        """Move the board right"""
        moved = False
        for i in range(self.size):
            original_row = self.board[i].copy()
            # Reverse the row, compress, merge, compress, then reverse back
            reversed_row = self.board[i][::-1]
            reversed_row = self.compress(reversed_row)
            reversed_row = self.merge(reversed_row)
            reversed_row = self.compress(reversed_row)
            self.board[i] = reversed_row[::-1]
            if not np.array_equal(original_row, self.board[i]):
                moved = True
        return moved

    def move_up(self):
        """Move the board up"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            col = self.compress(self.board[:, j])
            col = self.merge(col)
            col = self.compress(col)
            self.board[:, j] = col
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def move_down(self):
        """Move the board down"""
        moved = False
        for j in range(self.size):
            original_col = self.board[:, j].copy()
            # Reverse the column, compress, merge, compress, then reverse back
            reversed_col = self.board[:, j][::-1]
            reversed_col = self.compress(reversed_col)
            reversed_col = self.merge(reversed_col)
            reversed_col = self.compress(reversed_col)
            self.board[:, j] = reversed_col[::-1]
            if not np.array_equal(original_col, self.board[:, j]):
                moved = True
        return moved

    def is_game_over(self):
        """Check if there are no legal moves left"""
        # If there is any empty cell, the game is not over
        if np.any(self.board == 0):
            return False

        # Check horizontally
        for i in range(self.size):
            for j in range(self.size - 1):
                if self.board[i, j] == self.board[i, j+1]:
                    return False

        # Check vertically
        for j in range(self.size):
            for i in range(self.size - 1):
                if self.board[i, j] == self.board[i+1, j]:
                    return False

        return True

    def step(self, action):
        """Execute one action"""
        assert self.action_space.contains(action), "Invalid action"

        if action == 0:
            moved = self.move_up()
        elif action == 1:
            moved = self.move_down()
        elif action == 2:
            moved = self.move_left()
        elif action == 3:
            moved = self.move_right()
        else:
            moved = False

        self.last_move_valid = moved  # Record if the move was valid

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board, self.score, done, {}

    def render(self, mode="human", action=None):
        """
        Render the current board using Matplotlib.
        This function does not check if the action is valid and only displays the current board state.
        """
        fig, ax = plt.subplots(figsize=(4, 4))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlim(-0.5, self.size - 0.5)
        ax.set_ylim(-0.5, self.size - 0.5)

        for i in range(self.size):
            for j in range(self.size):
                value = self.board[i, j]
                color = COLOR_MAP.get(value, "#3c3a32")  # Default dark color
                text_color = TEXT_COLOR.get(value, "white")
                rect = plt.Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=color, edgecolor="black")
                ax.add_patch(rect)

                if value != 0:
                    ax.text(j, i, str(value), ha='center', va='center',
                            fontsize=16, fontweight='bold', color=text_color)
        title = f"score: {self.score}"
        if action is not None:
            title += f" | action: {self.actions[action]}"
        plt.title(title)
        plt.gca().invert_yaxis()
        plt.show()

    def simulate_row_move(self, row):
        """Simulate a left move for a single row"""
        # Compress: move non-zero numbers to the left
        new_row = row[row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        # Merge: merge adjacent equal numbers (do not update score)
        for i in range(len(new_row) - 1):
            if new_row[i] == new_row[i + 1] and new_row[i] != 0:
                new_row[i] *= 2
                new_row[i + 1] = 0
        # Compress again
        new_row = new_row[new_row != 0]
        new_row = np.pad(new_row, (0, self.size - len(new_row)), mode='constant')
        return new_row

    def is_move_legal(self, action):
        """Check if the specified move is legal (i.e., changes the board)"""
        # Create a copy of the current board state
        temp_board = self.board.copy()

        if action == 0:  # Move up
            for j in range(self.size):
                col = temp_board[:, j]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col
        elif action == 1:  # Move down
            for j in range(self.size):
                # Reverse the column, simulate, then reverse back
                col = temp_board[:, j][::-1]
                new_col = self.simulate_row_move(col)
                temp_board[:, j] = new_col[::-1]
        elif action == 2:  # Move left
            for i in range(self.size):
                row = temp_board[i]
                temp_board[i] = self.simulate_row_move(row)
        elif action == 3:  # Move right
            for i in range(self.size):
                row = temp_board[i][::-1]
                new_row = self.simulate_row_move(row)
                temp_board[i] = new_row[::-1]
        else:
            raise ValueError("Invalid action")

        # If the simulated board is different from the current board, the move is legal
        return not np.array_equal(self.board, temp_board)

import copy
import random
import math
import numpy as np
from collections import defaultdict


# -------------------------------
# TODO: Define transformation functions (rotation and reflection), i.e., rot90, rot180, ..., etc.
# -------------------------------
def rot90(pattern):
    """Rotates a pattern 90 degrees clockwise."""
    return [(y, 3 - x) for x, y in pattern]

def rot180(pattern):
    """Rotates a pattern 180 degrees."""
    return [(3 - x, 3 - y) for x, y in pattern]

def rot270(pattern):
    """Rotates a pattern 270 degrees clockwise."""
    return [(3 - y, x) for x, y in pattern]

def reflect(pattern):
    """Reflects the pattern horizontally."""
    return [(x, 3 - y) for x, y in pattern]


class NTupleApproximator:
    def __init__(self, board_size, patterns):
        """
        重點：我們只對「每個 pattern」配置一個權重字典，
        而非對 pattern 的每個對稱形都建權重。
        """
        self.board_size = board_size
        self.patterns = patterns

        self.weights = [defaultdict(float) for _ in patterns]
        # Generate symmetrical transformations for each pattern
        self.symmetry_patterns = []
        for pattern in self.patterns:
            syms = self.generate_symmetries(pattern)
            for syms_ in syms:
                self.symmetry_patterns.append(syms_)
            # print(syms)

    def generate_symmetries(self, pattern):
        """
        對單一個 pattern 產生所有對稱形 (至多 8 種)。
        """
        transformations = set()
        transformations.add(tuple(pattern))

        for rot_func in [rot90, rot180, rot270]:
            transformations.add(tuple(rot_func(pattern)))

        refl = reflect(pattern)
        transformations.add(tuple(refl))
        for rot_func in [rot90, rot180, rot270]:
            transformations.add(tuple(rot_func(refl)))

        return list(transformations)

    def tile_to_index(self, tile):
        """
        2048 tile -> 指數; e.g. 2->1,4->2,8->3,... 如果是0或出錯就返回0
        """
        if tile == 0:
            return 0
        else:
            return int(math.log(tile, 2))

    def get_feature(self, board, coords):
        """
        從 board 上根據 coords 取出 tile 值，轉成 tile index。
        回傳一個 feature tuple，例如 (1, 2, 3, 0)
        """
        return tuple(self.tile_to_index(board[x, y]) for x, y in coords)


    def value(self, board):
        """
        對所有 pattern 的 8 種對稱形，各自取出 feature 並查權重值，加總為整體 board 估計值。
        """
        total_value = 0.0
        for i, pattern in enumerate(self.patterns):
            symmetries = self.generate_symmetries(pattern)
            for coords in symmetries:
                feature = self.get_feature(board, coords)
                total_value += self.weights[i][feature]
        return total_value

    def update(self, board, delta, alpha):
        """
        對所有 pattern 的對稱形各自取出 feature 並根據 TD 誤差平均更新對應的權重值。
        """
        for i, pattern in enumerate(self.patterns):
            symmetries = self.generate_symmetries(pattern)
            for coords in symmetries:
                feature = self.get_feature(board, coords)
                self.weights[i][feature] += alpha * (delta / len(symmetries))


    def save(self, filename="/content/drive/MyDrive/approximator.pkl"):
        import pickle
        with open(filename, "wb") as f:
            pickle.dump((self.weights,), f)
        print(f"NTupleApproximator saved to {filename}")

    def load(self, path):
        """Load weights from a pickle file."""
        import pickle
        with open(path, "rb") as f:
            loaded_data = pickle.load(f)

        # Handle different save formats
        if isinstance(loaded_data, tuple):
            # Old format: (weights,)
            self.weights = loaded_data[0]
        elif isinstance(loaded_data, NTupleApproximator):
            # New format: entire approximator object
            self.weights = loaded_data.weights
        else:
            raise TypeError(f"Unknown format in pickle file: {type(loaded_data)}")

        print(f"Weights loaded from {path}")

approximator = NTupleApproximator(board_size=4, patterns=patterns)
approximator.load("./8_6tuple_alpha0_ep24000.pkl")

def get_action(state, score):
    env = Game2048Env()
    legal_moves = [a for a in range(4) if env.is_move_legal(a)]
    if not legal_moves:
        break

    if random.random() < epsilon:
        action = random.choice(legal_moves)
    else:
        best_value = float('-inf')
        best_action = None
        for a in legal_moves:
            env_copy = copy.deepcopy(env)
            _, sim_score, _, info = env_copy.step(a)
            after_state = info["after"]
            val = (sim_score - previous_score) + 0.99 * approximator.value(after_state)
            if val > best_value:
                best_value = val
                best_action = a
        action = best_action
    
    return action
    # return random.choice([0, 1, 2, 3]) # Choose a random action
    
    # You can submit this random agent to evaluate the performance of a purely random strategy.


