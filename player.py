#!/usr/bin/env python3
import random
import sys

from fishing_game_core.game_tree import Node
from fishing_game_core.player_utils import PlayerController
from fishing_game_core.shared import ACTION_TO_STR
from time import time

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)


class PlayerControllerHuman(PlayerController):
    def player_loop(self):
        """
        Function that generates the loop of the game. In each iteration
        the human plays through the keyboard and send
        this to the game through the sender. Then it receives an
        update of the game through receiver, with this it computes the
        next movement.
        :return:
        """

        while True:
            # send message to game that you are ready
            msg = self.receiver()
            if msg["game_over"]:
                return


class PlayerControllerMinimax(PlayerController):

    def __init__(self):
        super(PlayerControllerMinimax, self).__init__()
        self.MAX_DEPTH = 7

        self.alpha = 3 # Multiplier of difference of scores
        self.beta = 2  # Multiplier of difference of caught fish scores
        self.gamma = 0.1 # Decaying factor for larger depths
        self.threshold = 0.065 # Time limit

        self.max_depth = 1
        self.table = {}

    def encode_state(self, current_node):
        player = current_node.state.get_player()
        fish_positions = current_node.state.get_fish_positions().values()
        fish_scores = current_node.state.get_fish_scores().values()
        hooks_coords = current_node.state.get_hook_positions()
        (our_hook_x, our_hook_y) = hooks_coords[0]
        (their_hook_x, their_hook_y) = hooks_coords[1]

        encoding = str(player) + "_" + str(our_hook_x) + "_" + str(our_hook_y) + "_" + str(their_hook_x) + "_" + str(their_hook_y)
        fish = sorted(zip(fish_positions, fish_scores), key = lambda x: (x[0], x[1]))
        for elem in fish:
            encoding += "_" + str(elem[0][0]) + "_" + str(elem[0][1]) + "_" + str(elem[1])
        return encoding

    def player_loop(self):
        """
        Main loop for the minimax next move search.
        :return:
        """

        # Generate first message (Do not remove this line!)
        first_msg = self.receiver()

        while True:
            msg = self.receiver()

            # Create the root node of the game tree
            node = Node(message=msg, player=0)

            # Possible next moves: "stay", "left", "right", "up", "down"
            best_move = self.search_best_next_move(initial_tree_node=node)

            # Execute next action
            self.sender({"action": best_move, "search_time": None})

    def search_best_next_move(self, initial_tree_node):
        """
        Use minimax (and extensions) to find best possible next move for player 0 (green boat)
        :param initial_tree_node: Initial game tree node
        :type initial_tree_node: game_tree.Node
            (see the Node class in game_tree.py for more information!)
        :return: either "stay", "left", "right", "up" or "down"
        :rtype: str
        """

        self.start = time()
        self.out_of_time = False
        self.best_values = {0:0, 1:0, 2:0, 3:0, 4:0}
        self.child_indices = [0, 1, 2, 3, 4]

        next_move = self.init_minimax_ids(initial_tree_node)

        return ACTION_TO_STR[next_move]
    
    def init_minimax_ids(self, root):
        root.compute_and_get_children()
        best_index = 0
        if len(root.children) > 1:
            for max_depth in range(1,self.MAX_DEPTH+1):
                self.max_depth = max_depth
                alpha = -float('inf')
                beta = float('inf')
                for idx in self.child_indices:
                    child = root.children[idx]
                    v = self.minimax(child, 1, alpha, beta, idx)
                    self.best_values[idx] += v
                    best_index = list(self.best_values.values()).index(max(list(self.best_values.values())))
                    if self.out_of_time:
                        break
                if self.out_of_time:
                    break
                self.child_indices = sorted(self.child_indices, key=lambda x : -self.best_values[x])
        return root.children[best_index].move
    
    def minimax(self, current_node, current_depth, alpha, beta, index):
        self.current_time = time()
        if current_depth == self.max_depth or (len(current_node.state.get_fish_positions()) == 1 and current_node.state.get_caught() != (None, None)):
            val = self.heuristic(current_node, current_depth)
            return val
        elif (self.current_time - self.start) > self.threshold:
            self.out_of_time = True
            return self.best_values[index]     
        else:
            encoding = self.encode_state(current_node)
            remaining_depth = self.max_depth - current_depth
            if encoding in self.table and self.table[encoding][1] == remaining_depth:
                return self.table[encoding][0]
            else:
                current_node.compute_and_get_children()
                if current_node.state.get_player() == 0: # MAX
                    best = -float('inf')
                    for child in current_node.children:
                        v = self.minimax(child, current_depth+1, alpha, beta, index)
                        best = max(best, v)
                        alpha = max(alpha, best)
                        if beta <= alpha:
                            break
                else: # MIN
                    best = float('inf')
                    for child in current_node.children:
                        v = self.minimax(child, current_depth+1, alpha, beta, index)
                        best = min(best,v)
                        beta = min(beta, best)
                        if beta <= alpha:
                            break
                    self.table[encoding] = (best, self.max_depth - current_depth)
                return best
    
    def heuristic(self, current_node, depth):
        fish_positions = current_node.state.get_fish_positions()
        fish_scores = current_node.state.get_fish_scores()
        hook_positions = current_node.state.get_hook_positions()
        (our_score, their_score) = current_node.state.get_player_scores()
        our_hook = hook_positions[0]
        their_hook = hook_positions[1]

        our_position_term, our_caught_fish = self.compute_position_term(fish_positions, fish_scores, our_hook)
        their_position_term, their_caught_fish = self.compute_position_term(fish_positions, fish_scores, their_hook)

        return ((our_position_term - their_position_term) + self.alpha*(our_score - their_score) + self.beta*(our_caught_fish - their_caught_fish)) * (1 - depth*self.gamma)

    def compute_position_term(self, fish_positions, fish_scores, hook_coord):
        acc = 0
        caught_fish = 0
        for fish in fish_positions:
            score = fish_scores[fish]
            fish_coord = fish_positions[fish]
            dist = self.L1_wrapping(fish_coord, hook_coord)
            if dist != 0:
                acc += score / dist
            else:
                caught_fish = score
        return acc, caught_fish
    
    def L1_wrapping(self, fish_coord, hook_cord):
        x_diff = abs(fish_coord[0] - hook_cord[0])
        y_diff = abs(fish_coord[1] - hook_cord[1])
        half_width = 10
        if x_diff > half_width:
            x_diff = 2 * half_width - x_diff
        return x_diff + y_diff



