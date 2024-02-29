# This file is where game logic lives. No input
# or output happens here. The logic in this file
# should be unit-testable.
import os
import time
import numpy as np
import pandas as pd
from random import choice
from treelib import Tree, Node

cond_to_chr = {0: ' ', 1: 'O', 2: 'X'}
role_to_int = {'O': 1, 'X': 2, None: 0}
meta_game_completed = [0] * 9

def make_empty_board():
    return np.zeros([9,3,3])


class Game:
    def __init__(self, game_mode='pvp', bot_type='Minimax', start_first=True, args=None, player_names=None):
        self.board = make_empty_board()
        self.game_mode = game_mode
        self.id_to_name = {1: 'O', 2: 'X'}
        if args is None:
            if game_mode == 'pvp':
                self.player_one = HumanPlayer(1, player_name=player_names[0] if player_names is not None else None)
                self.player_two = HumanPlayer(2, player_name=player_names[1] if player_names is not None else None)
            elif game_mode == 'pve':
                self.player_one = HumanPlayer(1, player_name=player_names[0] if player_names is not None else None)
                bot_type = bot_type.lower()
                if bot_type in name2class:
                    self.player_two = name2class[bot_type](2)
                else:
                    self.player_two = None
            elif game_mode == 'eve':
                self.player_one = MinimaxBot(1)
                self.player_two = MinimaxBot(2)
            elif game_mode == 'evr':
                self.player_one = MinimaxBot(1)
                self.player_two = RandomBot(2)
        else:
            self.player_one = name2class[args.player1](1)
            self.player_two = name2class[args.player2](2)
            start_first = True
        if start_first:
            self.player_now = self.player_one
        else:
            self.player_now = self.player_two
        if not isinstance(self.player_now, HumanPlayer):
            self.player_now.name += '_first'
        if not isinstance(self.other_player(self.player_now), HumanPlayer):
            self.other_player(self.player_now).name += '_second'
        self.starter = self.player_now.name
        self.name2player = {
            self.player_one.name: self.player_one,
            self.player_two.name: self.player_two,
        }
        self.id_to_name = {
            self.player_one.player_id: self.player_one.name,
            self.player_two.player_id: self.player_two.name,
        }
        self.savegame: pd.DataFrame = self.read_savegame("savegame.csv")
        print(self.id_to_name)

    def read_savegame(self, filename):
        if os.path.exists(filename):
            return pd.read_csv(filename)
        else:
            return pd.DataFrame({
                "game_id": [],
                "winner": [],
                "rounds": [],
                "starter": [],
                "players": [],
            })
            
    def check_local_winner(self, player, board):
        checks = []
        checks += [all(board[0] == player), all(board[1] == player), all(board[2] == player)]
        checks += [all(board[:, 0] == player), all(board[:, 1] == player), all(board[:, 2] == player)]
        checks += [all(board.flatten()[0::4] == player), all(board.flatten()[2:7:2] == player)]
        return any(checks)

    def check_winner(self, player, board):
        meta_checks = []
        for i in range(9):
            checks = []
            checks += [all(board[i][0] == player), all(board[i][1] == player), all(board[i][2] == player)]
            checks += [all(board[i][:, 0] == player), all(board[i][:, 1] == player), all(board[i][:, 2] == player)]
            checks += [all(board[i].flatten()[0::4] == player), all(board[i].flatten()[2:7:2] == player)]
            meta_checks.append(any(checks))
        # check the meta board
        meta_game_won = [i for i, v in enumerate(meta_checks) if v]
        for meta_game_won_index in meta_game_won:
            if not meta_game_completed[meta_game_won_index]:
                print("Game %s has been won by player %s" % (meta_game_won_index, self.id_to_name[player]))
                self.board[meta_game_won_index][:] = player
                meta_game_completed[meta_game_won_index] = player
        # Check if there's any board is a tie, which needs to be refreshed by filling 0
        for i in range(9):
            if all(board[i].flatten() != 0) and not meta_game_completed[i]:
                self.board[i][:] = 0
                print("Game %s has been drawed. Refresh the board!" % i)
        meta_checks = np.array(meta_checks).reshape(3, 3)
        meta_check_results = []
        meta_check_results += [all(meta_checks[0]), all(meta_checks[1]), all(meta_checks[2])]
        meta_check_results += [all(meta_checks[:, 0]), all(meta_checks[:, 1]), all(meta_checks[:, 2])]
        meta_check_results += [all(meta_checks.flatten()[0::4]), all(meta_checks.flatten()[2:7:2])]
        return any(meta_check_results)


    def check_draw(self, board):
        return all(board.flatten() != 0)


    def get_winner(self, board, get_global=True):
        """Determines the winner of the given board.
        Returns 'X', 'O', or None."""
        if get_global:
            if isinstance(board, list):
                board = np.array([[role_to_int[v] for v in l] for l in board])
            if self.check_winner(1, board):
                return 1
            elif self.check_winner(2, board):
                return 2
            else:
                return 0
        else:
            if isinstance(board, list):
                board = np.array([[role_to_int[v] for v in l] for l in board])
            if self.check_local_winner(1, board):
                return 1
            elif self.check_local_winner(2, board):
                return 2
            else:
                return 0


    def other_player(self, player):
        """Given the character for a player, returns the other player."""
        return self.player_one if player == self.player_two else self.player_two


    def move(self, player, position):
        if position[1] is not None and position[2] is not None and position in player.get_available_moves(self.board):
            self.board[position[0]][position[1]][position[2]] = player.player_id
            return True
        else:
            print("The place has already been taken!")
            return False
        
        
    def __repr__(self):
        separator = '--------------' * 3 + '\n'
        header = '  ' + '  0   1   2    3   4   5    6   7   8' + '\n'
        row_index='a'
        row_template = '%s ' + '| %s | %s | %s |' * 3 + '\n'
        border = '==============' * 3 + '\n'
        returned_str = header + border
        for j in range(3):
            row_strings = []
            for i in range(3):
                row_strings.append(row_template % (
                    chr(ord(row_index)+(j*3+i)),
                    cond_to_chr[self.board[j*3][i][0]], cond_to_chr[self.board[j*3][i][1]], cond_to_chr[self.board[j*3][i][2]],
                    cond_to_chr[self.board[j*3+1][i][0]], cond_to_chr[self.board[j*3+1][i][1]], cond_to_chr[self.board[j*3+1][i][2]],
                    cond_to_chr[self.board[j*3+2][i][0]], cond_to_chr[self.board[j*3+2][i][1]], cond_to_chr[self.board[j*3+2][i][2]],
                ))
            row_blocks = separator.join(row_strings)
            row_blocks += border
            returned_str += row_blocks
        return returned_str
    
    
    def add_savegame(self, winner, rounds, is_draw):
        game_id = max(self.savegame.game_id) + 1 if len(self.savegame) > 0 else 0
        self.savegame = self.savegame.append({
            "game_id": game_id,
            "winner": winner,
            "rounds": rounds,
            "is_draw": is_draw,
            "starter": self.starter,
            "players": [self.player_one.name, self.player_two.name],
        }, ignore_index=True)
    
    def start(self):
        winner = None
        rounds = 0
        last_time = time.time()
        while winner == None:
            rounds += 1
            # TODO: Show the board to the user.
            print(self)
            # TODO: Input a move from the player.
            meta_block_id, row, col = self.player_now.get_move(self)
            # TODO: Update the board.
            success = self.move(self.player_now, (meta_block_id, row, col))
            # TODO: Update who's turn it is.
            if success:
                self.player_now.num_moves += 1
                self.player_now.time_takes += time.time() - last_time
                last_time = time.time()
                self.player_now = self.other_player(self.player_now)
            checked = self.get_winner(self.board)
            if checked != 0:
                winner = checked
                print(self)
                print("Player %s has won!!" % self.id_to_name[checked])
                self.add_savegame(self.id_to_name[checked], rounds, False)
            if self.check_draw(self.board):
                print(self)
                print("Draw!")
                self.add_savegame(None, rounds, True)
                break
        self.save_game("savegame.csv")
        self.update_statistics(winner)
    
    def save_game(self, filename):
        self.savegame.to_csv(filename, index=False)
        
    def update_statistics(self, winner, filename="statistics.csv"):
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
        else:
            df = pd.DataFrame({
                "player_name": [],
                "wins": [],
                "played": [],
                "drawed": [],
                "thinking_time": [],
                "moves_take": [],
            }).set_index('player_name')
        winner_name = self.id_to_name[winner] if winner is not None else None
        for player_name in self.id_to_name.values():
            if player_name in df.index.tolist():
                df.loc[player_name, 'played'] += 1
                df.loc[player_name, 'thinking_time'] += self.name2player[player_name].time_takes
                df.loc[player_name, 'moves_take'] += self.name2player[player_name].num_moves
            else:
                df = df.append(pd.DataFrame({
                    "wins": [0],
                    "played": [1],
                    "drawed": [0],
                    "thinking_time": [self.name2player[player_name].time_takes],
                    "moves_take": [self.name2player[player_name].num_moves],
                }, index=[player_name]))
        if winner_name is not None:
            df.loc[winner_name, 'wins'] += 1
        else:
            for player_name in self.id_to_name.values():
                df.loc[player_name, 'drawed'] += 1
        df.to_csv(filename)
        df.sort_values(by=['wins', 'drawed'], ascending=False, inplace=True)
        print("Leaderboard:")
        print(df)
    


class Player:
    def __init__(self, player_id):
        self.player_id = player_id
        self.num_moves = 0
        self.time_takes = 0
    
    def get_available_moves(self, board: np.ndarray):
        indices = np.where(board == 0)
        available = [(i, j, k) for i, j, k in zip(indices[0], indices[1], indices[2])]
        return available

    def get_move(self):
        pass
    

class HumanPlayer(Player):
    def __init__(self, player_id, player_name=None):
        super().__init__(player_id)
        if player_name is None:
            self.name = input("Please input your name:")
        else:
            self.name = player_name
    
    def get_move(self, game=None):
        try:
            position = input("Player %s please input the position you want to take, for example, \"a 0\"" % self.player_id)
            row, col = position.split()
            if row not in ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i']:
                print("Wrong input! The index of rows should be 'a' to 'i'")
                return None, None, None
            if col not in ['0', '1', '2', '3', '4', '5', '6', '7', '8']:
                print("Wrong input! The index of columns should be 0-8")
                return None, None, None
            row = ord(row) - ord('a')
            col = int(col)
            meta_block_row, meta_block_col = row // 3, col // 3
            meta_block_id = meta_block_row * 3 + meta_block_col
            inner_block_row, inner_block_col = row % 3, col % 3
            return meta_block_id, inner_block_row, inner_block_col
        except Exception as e:
            print("Invalid input position command! Try again!")
            return None, None, None


class RandomBot(Player):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.name = "random_bot"
        
    def get_move(self, game: Game):
        available = self.get_available_moves(game.board)
        return choice(available)
    
class SequentialBot(Player):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.name = "sequential_bot"
        
    def get_move(self, game: Game):
        available = self.get_available_moves(game.board)
        return available[0]
    
    
class MinimaxBot(Player):
    def __init__(self, player_id):
        super().__init__(player_id)
        self.name = "minimax_bot"
        
    def get_move(self, game: Game):
        # First, decide which block to play (0-8)
        meta_move = minimax_algo(self, game, np.array(meta_game_completed).reshape(3, 3))
        # Then, decide which position to play (0-2, 0-2)
        meta_move_id = meta_move[0] * 3 + meta_move[1]
        inner_move = minimax_algo(self, game, game.board[meta_move_id])
        print("Move: ", meta_move_id, inner_move[0], inner_move[1])
        return meta_move_id, inner_move[0], inner_move[1]

    def get_available_moves(self, board: np.ndarray, get_global=True):
        if get_global:
            return super().get_available_moves(board)
        indices = np.where(board == 0)
        available = [(i, j) for i, j in zip(indices[0], indices[1])]
        return available


def minimax_algo(bot: MinimaxBot, game: Game, board):
    if (board == 0).all():
        return 1, 1
    status_stack = ["root"]
    search_tree = Tree()
    search_tree.create_node("root", "root", data = {
        'board': np.copy(board),
        'turn': 'max',
        'move': None,
        'score': None,
        'depth': 0,
    })
    while len(status_stack) != 0:
        status_id = status_stack.pop()
        status_node = search_tree.get_node(status_id)
        status_board = status_node.data['board']
        availables = bot.get_available_moves(status_board, get_global=False)
        for i, available in enumerate(availables):
            new_board = np.copy(status_board)
            new_board[available[0]][available[1]] = bot.player_id if status_node.data['turn'] == 'max' else game.other_player(bot).player_id
            new_board_data = {
                'board': new_board,
                'turn': 'min' if status_node.data['turn'] == 'max' else 'max',
                'move': available,
                'depth': status_node.data['depth'] + 1,
            }
            winner = game.get_winner(new_board, get_global=False)
            this_id = status_id + '-' + str(i)
            if winner != 0:
                new_board_data['score'] = 10 - new_board_data['depth'] if winner == bot.player_id else new_board_data['depth'] - 10
            elif game.check_draw(new_board):
                new_board_data['score'] = 0
            else:
                new_board_data['score'] = None
                status_stack.append(this_id)
            search_tree.create_node(
                this_id,
                this_id,
                parent=status_id,
                data=new_board_data
            )
            
    all_nodes = search_tree.all_nodes()
    all_nodes.reverse()
    for node in all_nodes:
        assert isinstance(node, Node)
        if node.data['score'] is not None:
            continue
        else:
            children_nodes = search_tree.children(node.identifier)
            node.data['score'] = max([child.data['score'] for child in children_nodes]) if node.data['turn'] == 'max' else min([child.data['score'] for child in children_nodes])
    
    move = None
    max_score = -100
    for children in search_tree.children('root'):
        if children.data['score'] > max_score:
            max_score = children.data['score']
            move = children.data['move']
    return move
                
name2class = {
    "human": HumanPlayer,
    "random": RandomBot,
    "minimax": MinimaxBot,
    "sequential": SequentialBot,
}
    
    
if __name__ == '__main__':
    game = Game(game_mode='pve', start_first=True)
    game.start()