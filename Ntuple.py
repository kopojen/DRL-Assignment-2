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