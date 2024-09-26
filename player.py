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
        self.MAX_DEPTH = 3
        self.floor = 1e100
        self.gamma = 1

        self.max_depth = 1

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

        # EDIT THIS METHOD TO RETURN BEST NEXT POSSIBLE MODE USING MINIMAX ###

        # NOTE: Don't forget to initialize the children of the current node
        #       with its compute_and_get_children() method!

        start = time()
        next_move = self.init_minimax(initial_tree_node)
        # random_move = random.randrange(5)
        end = time()
        eprint((end-start) * 1000, " miliseconds.")
        return ACTION_TO_STR[next_move]
    
    def init_minimax(self, root):
        root.compute_and_get_children()
        global_scores = [0 for _ in range(len(root.children))]
        for max_depth in range(1,self.MAX_DEPTH):
            self.max_depth = max_depth
            scores = []
            alpha = -float('inf')
            beta = float('inf')
            for idx, child in enumerate(root.children):
                scores.append(self.minimax(child, 1, alpha, beta))
                global_scores[idx] += self.gamma / max_depth * scores[idx]
            eprint("Max_Depth: ", max_depth)
            eprint(scores)
        best_child = global_scores.index(max(global_scores))
        eprint(global_scores)
        eprint("Index: ", best_child, " Movement: ", ACTION_TO_STR[root.children[best_child].move])
        return root.children[best_child].move
    
    def minimax(self, current_node, current_depth, alpha, beta):
        if current_depth == self.max_depth or len(current_node.state.get_fish_positions()) == 0:
            # eprint("Final Depth: " + str(current_depth))
            return self.heuristic(current_node)
        else:
            current_node.compute_and_get_children()
            if current_node.state.get_player() == 0: # MAX
                best = -float('inf')
                for child in current_node.children:
                    v = self.minimax(child, current_depth+1, alpha, beta)
                    best = max(best, v)
                    alpha = max(alpha, best)
                    if beta <= alpha:
                        break
            else: # MIN
                best = float('inf')
                for child in current_node.children:
                    v = self.minimax(child, current_depth+1, alpha, beta)
                    best = min(best,v)
                    beta = min(beta, best)
                    if beta <= alpha:
                        break
            return best

    def heuristic(self, current_node):
        fish_positions = current_node.state.get_fish_positions()
        fish_scores = current_node.state.get_fish_scores()
        hook_positions = current_node.state.get_hook_positions()
        # scores = current_node.state.get_player_scores()
        # max_score = 0
        
        sum_scores = 0
        hook_x, hook_y = hook_positions[0]
        other_hook_x, other_hook_y = hook_positions[1]
        eprint("Entering for loop, move: ", ACTION_TO_STR[current_node.move])
        for idx in fish_positions.keys():
            fish_x, fish_y = fish_positions[idx]
            if not(fish_x == other_hook_x and fish_y == other_hook_y):
                fish_score = fish_scores[idx]
                # L1 Norm
                diff_x = abs(fish_x - hook_x)
                diff_y = abs(fish_y - hook_y)
                sum_diff = diff_x + diff_y
                if sum_diff == 0:
                    score = fish_score * self.floor
                else:
                    if fish_score < 0:
                        score = fish_score * sum_diff
                    else:
                        score = fish_score / sum_diff
                sum_scores += score
                eprint("Fish score: ", fish_score, ", L1 Distance:", sum_diff, ", Index: ", idx, ", Score: ", score)
                # if max_score < score:
                    # max_score = score
        return sum_scores



