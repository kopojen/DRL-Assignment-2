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
from Ntuple import NTupleApproximator
import gc


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

        self.last_move_valid = moved

        after_board = self.board.copy()

        if moved:
            self.add_random_tile()

        done = self.is_game_over()

        return self.board.copy(), self.score, done, {"after": after_board}

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
    def step_without_random_tile(self, action):
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

        self.last_move_valid = moved
        done = self.is_game_over()
        return self.board.copy(), self.score, done, {}

import copy
import random
import math
import numpy as np

# Note: This MCTS implementation is almost identical to the previous one,
# except for the rollout phase, which now incorporates the approximator.

# Node for TD-MCTS using the TD-trained value approximator
class TD_MCTS_Node:
    def __init__(self, state, score, parent=None, action=None, prob=0):
        """
        state: current board state (numpy array)
        score: cumulative score at this node
        parent: parent node (None for root)
        action: action taken from parent to reach this node
        """
        self.state = state
        self.score = score
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.total_reward = 0.0
        self.untried_actions = []
        self.prob = prob

        env_copy = Game2048Env()
        env_copy.board = state
        self.untried_actions = [a for a in range(4) if env_copy.is_move_legal(a)]

    def fully_expanded(self):
        # A node is fully expanded if no legal actions remain untried.
        return len(self.untried_actions) == 0


# TD-MCTS class utilizing a trained approximator for leaf evaluation
class TD_MCTS:
    def __init__(self, env, approximator, iterations=500, exploration_constant=1.41, rollout_depth=3, gamma=0.99):
        self.env = env
        self.approximator = approximator
        self.iterations = iterations
        self.c = exploration_constant
        self.rollout_depth = rollout_depth
        self.gamma = gamma

    def create_env_from_state(self, state, score):
        # Create a deep copy of the environment with the given state and score.
        new_env = copy.deepcopy(self.env)
        new_env.board = state.copy()
        new_env.score = score
        return new_env

    def select_child(self, node):
        # TODO: Use the UCT formula: Q + c * sqrt(log(parent.visits)/child.visits) to select the best child.
        log_parent_visits = math.log(node.visits + 1e-6)
        best_child = None
        best_uct = float('-inf')

        for child in node.children.values():
            # Q-value estimate
            if child.visits > 0:
                
                q_value = (child.total_reward / child.visits) / 50000
                # print(child.total_reward, child.visits)
            else:
                q_value = 0.0

            # Exploration term
            uct_exploration = self.c * math.sqrt(log_parent_visits / (child.visits + 1e-6))

            # Final UCT
            # print(child.total_reward)
            # print(q_value, uct_exploration)
            uct_value = q_value + uct_exploration
            
            if uct_value > best_uct:
                best_uct = uct_value
                best_child = child

        return best_child


    def rollout(self, sim_env, depth):
        # TODO: Perform a random rollout until reaching the maximum depth or a terminal state.
        # TODO: Use the approximator to evaluate the final state.

        for step in range(depth):
            # Check if game is already over
            legal_moves = [a for a in range(4) if sim_env.is_move_legal(a)]
            if not legal_moves:
                break

            # Random move for exploration
            action = random.choice(legal_moves)
            _, _, done, _ = sim_env.step(action)

        best_action_value = float('-inf')
        legal_actions = [a for a in range(4) if sim_env.is_move_legal(a)]
        
        if not legal_actions:
            return sim_env.score
            
        for action in legal_actions:
            temp_env = copy.deepcopy(sim_env)

            prev_score = temp_env.score
            _, _, _, _ = temp_env.step(action)
            
            # print("approximator", self.approximator.value(temp_env.board))
            total = (temp_env.score + self.approximator.value(temp_env.board))

            if total > best_action_value:
                best_action_value = total
        
        return best_action_value

    def backpropagate(self, node, reward):
        # TODO: Propagate the obtained reward back up the tree.
        while node is not None:
            node.visits += 1
            node.total_reward += reward
            # if node.total_reward < 0:
            #     print(reward)
            #     print("total_reward", node.total_reward)
            node = node.parent

    def run_simulation(self, root):
        node = root
        sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Selection: Traverse the tree until reaching an unexpanded node.
        while node.fully_expanded() and node.children:
            selected_child = self.select_child(node)
            if selected_child is None:
                print("error error error error error error error error error error error error")
                break
            if selected_child.children:
                children = list(selected_child.children.values())
                probs = [c.prob for c in children]
                node = random.choices(children, weights=probs, k=1)[0]
                sim_env = self.create_env_from_state(node.state, node.score)

        # TODO: Expansion: If the node is not terminal, expand an untried action.
        if node.untried_actions:
            action = node.untried_actions.pop()
            sim_env.step_without_random_tile(action)
            new_state = sim_env.board.copy()
            new_score = sim_env.score
            
            new_node = TD_MCTS_Node(
                new_state, new_score, parent=node,
                action=action, prob=-1
            )
            
            node.children[action] = new_node
            
            node = new_node
            
            empty_cells = list(zip(*np.where(sim_env.board == 0)))
            num_empty = len(empty_cells)
            i = 0
            for x, y in empty_cells:
                for tile_value, prob in [(2, 0.9), (4, 0.1)]:
                    temp_board = sim_env.board.copy()
                    temp_board[x, y] = tile_value
                    child_node = TD_MCTS_Node(
                        temp_board,
                        sim_env.score,
                        parent=node,
                        action=-1, 
                        prob=(prob / num_empty)
                    )
                    node.children[i] = child_node
                    i += 1

            if node.children:
                children = list(node.children.values())
                probs = [child.prob for child in children]
                node = random.choices(children, weights=probs, k=1)[0]

        # Rollout: Simulate a random game from the expanded node.
        rollout_reward = self.rollout(sim_env, self.rollout_depth)
        # Backpropagate the obtained reward.
        self.backpropagate(node, rollout_reward)

    def best_action_distribution(self, root):
        # Compute the normalized visit count distribution for each child of the root.
        total_visits = sum(child.visits for child in root.children.values())
        distribution = np.zeros(4)
        best_visits = -1
        best_action = None
        for action, child in root.children.items():
            distribution[action] = child.visits / total_visits if total_visits > 0 else 0
            if child.visits > best_visits:
                best_visits = child.visits
                best_action = action
        return best_action, distribution
    
approximator = None
td_mcts = None
patterns = [
    [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
    [(0, 1), (0, 2), (1, 1), (1, 2), (2, 1), (3, 1)],
    [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1)],
    [(0, 0), (0, 1), (1, 1), (1, 2), (1, 3), (2, 2)],
    [(0, 0), (0, 1), (0, 2), (1, 1), (2, 1), (2, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (3, 2)],
    [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1), (2, 0)],
    [(0, 0), (1, 0), (0, 1), (0, 2), (1, 2), (2, 2)]
]

def init_model():
    global approximator
    if approximator is None:
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except:
            pass
        with open("../restart_0.5/8_6tuple_alpha0_ep1000.pkl", "rb") as f:
            approximator = pickle.load(f)
        print("Weights loaded from ../restart_0.5/8_6tuple_alpha0_ep1000.pkl")

# MCTS + TD
def get_action(state, score):
    """
    用 MCTS 來決定要走哪個 action
    """
    init_model()
    # 1) 初始化 Game2048Env 及 Node
    env = Game2048Env()
    env.board = copy.deepcopy(state)
    env.score = score

    root = TD_MCTS_Node(env.board, env.score)
    td_mcts = TD_MCTS(env, approximator, iterations=1000, exploration_constant=1.41, rollout_depth=3, gamma=0.99)

    # 3) run multiple simulations
    for _ in range(td_mcts.iterations):
        td_mcts.run_simulation(root)

    # 4) 拿到最常訪問的 action
    best_act, _ = td_mcts.best_action_distribution(root)

    return best_act

# TD only
# def get_action(state, score):
#     init_model()

#     env = Game2048Env()
#     env.board = copy.deepcopy(state)
#     env.score = score

#     legal_moves = [a for a in range(4) if env.is_move_legal(a)]

#     best_value = float('-inf')
#     best_action = None
    
#     print(type(approximator))
#     for a in legal_moves:
#         env_copy = copy.deepcopy(env)
#         _, sim_score, _, info = env_copy.step(a)
#         after_state = info["after"]
#         val = (sim_score - score) + 0.99 * approximator.value(after_state)
#         if val > best_value:
#             best_value = val
#             best_action = a

#     return best_action