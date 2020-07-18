import string

CHAIN_LENGTH_TO_WIN = 5
BOARD_WIDTH = 7
BOARD_HEIGHT = 7
EMPTY = '.'
RED = 'R'
BLUE = 'B'
RED_LABELS = list(string.ascii_lowercase)
RED_PLAY_LABELS = list(string.ascii_lowercase)
BLUE_LABELS = list(string.ascii_uppercase)
BLUE_PLAY_LABELS = list(string.ascii_uppercase)


class reflex_agent:
    def __init__(self, play_labels, agent_labels, opponent_labels):
        self.play_labels = play_labels
        self.labels = agent_labels
        self.opponent_labels = opponent_labels
        self.nodes_expanded = -1

    def make_move(self, board):
        move = self.try_end_4_chain(board, self.labels)
        if move:
            print('made winning move')
            return move

        move = self.try_end_4_chain(board, self.opponent_labels)
        if move:
            print('blocked winning move')
            return move

        move = self.try_block_3_chain(board)
        if move:
            print('blocked 3 chain')
            return move

        move = self.try_build_winning_block(board)
        if move:
            print('built up a winning block')
            return move

        print('no move was made')

    def try_end_4_chain(self, board, chain_labels):
        """Returns the coords of the move if the move was made, None otherwise"""
        # Check for a chain of 4 stones of the specified color and finish the chain if possible
        for x in range(board.width):
            for y in range(board.height):
                if not board.can_place(x, y): continue

                up_run_start = (x, y+1)
                up_run_end = (x, y + CHAIN_LENGTH_TO_WIN-1)

                down_run_start = (x, y-1)
                down_run_end = (x, y - (CHAIN_LENGTH_TO_WIN-1))

                right_run_start = (x+1, y)
                right_run_end = (x + CHAIN_LENGTH_TO_WIN-1, y)

                left_run_start = (x-1, y)
                left_run_end = (x - (CHAIN_LENGTH_TO_WIN-1), y)

                up_right_run_start = (x+1, y+1)
                up_right_run_end = (x + CHAIN_LENGTH_TO_WIN-1, y + CHAIN_LENGTH_TO_WIN-1)

                down_right_run_start = (x+1, y-1)
                down_right_run_end = (x + CHAIN_LENGTH_TO_WIN-1, y - (CHAIN_LENGTH_TO_WIN-1))

                up_left_run_start = (x-1, y+1)
                up_left_run_end = (x - (CHAIN_LENGTH_TO_WIN-1), y + CHAIN_LENGTH_TO_WIN-1)

                down_left_run_start = (x-1, y-1)
                down_left_run_end = (x - (CHAIN_LENGTH_TO_WIN-1), y - (CHAIN_LENGTH_TO_WIN-1))

                # list of tuples of (run_start_coords, run_end_coords)
                chain_coords = [
                    (up_run_start, up_run_end),
                    (down_run_start, down_run_end),
                    (right_run_start, right_run_end),
                    (left_run_start, left_run_end),
                    (up_right_run_start, up_right_run_end),
                    (down_right_run_start, down_right_run_end),
                    (up_left_run_start, up_left_run_end),
                    (down_left_run_start, down_left_run_end)
                ]

                for start_coords, end_coords in chain_coords:
                    chain = board.get_block(*start_coords, *end_coords)
                    if chain and all(spot in chain_labels for spot in chain):
                        board.place_stone(x, y, self.play_labels.pop(0))
                        print('placed stone at ({}, {})'.format(x, y))
                        return (x, y)

        return None

    def try_block_3_chain(self, board):
        """Returns the coords of the move if the move was made, None otherwise"""
        # Check for a chain of 3 of the opponent's stones with empty spaces on
        # both ends and block the chain
        for x in range(board.width):
            for y in range(board.height):
                if not board.can_place(x, y): continue

                up_run_start = (x, y+1)
                up_run_end = (x, y + (CHAIN_LENGTH_TO_WIN-2))
                up_run_next = (x, y + (CHAIN_LENGTH_TO_WIN-1))

                down_run_start = (x, y-1)
                down_run_end = (x, y - (CHAIN_LENGTH_TO_WIN-2))
                down_run_next = (x, y - (CHAIN_LENGTH_TO_WIN-1))

                right_run_start = (x+1, y)
                right_run_end = (x + (CHAIN_LENGTH_TO_WIN-2), y)
                right_run_next = (x + (CHAIN_LENGTH_TO_WIN-1), y)

                left_run_start = (x-1, y)
                left_run_end = (x - (CHAIN_LENGTH_TO_WIN-2), y)
                left_run_next = (x - (CHAIN_LENGTH_TO_WIN-1), y)

                up_right_run_start = (x+1, y+1)
                up_right_run_end = (x + (CHAIN_LENGTH_TO_WIN-2), y + (CHAIN_LENGTH_TO_WIN-2))
                up_right_run_next = (x + (CHAIN_LENGTH_TO_WIN-1), y + (CHAIN_LENGTH_TO_WIN-1))

                down_right_run_start = (x+1, y-1)
                down_right_run_end = (x + (CHAIN_LENGTH_TO_WIN-2), y - (CHAIN_LENGTH_TO_WIN-2))
                down_right_run_next = (x + (CHAIN_LENGTH_TO_WIN-1), y - (CHAIN_LENGTH_TO_WIN-1))

                up_left_run_start = (x-1, y+1)
                up_left_run_end = (x - (CHAIN_LENGTH_TO_WIN-2), y + (CHAIN_LENGTH_TO_WIN-2))
                up_left_run_next = (x - (CHAIN_LENGTH_TO_WIN-1), y + (CHAIN_LENGTH_TO_WIN-1))

                down_left_run_start = (x-1, y-1)
                down_left_run_end = (x - (CHAIN_LENGTH_TO_WIN-2), y - (CHAIN_LENGTH_TO_WIN-2))
                down_left_run_next = (x - (CHAIN_LENGTH_TO_WIN-1), y - (CHAIN_LENGTH_TO_WIN-1))

                # list of tuples of (run_start_coords, run_end_coords, coords of
                #                       spot one past the end coords)
                chain_coords = [
                    (up_run_start, up_run_end, up_run_next),
                    (down_run_start, down_run_end, down_run_next),
                    (right_run_start, right_run_end, right_run_next),
                    (left_run_start, left_run_end, left_run_next),
                    (up_right_run_start, up_right_run_end, up_right_run_next),
                    (down_right_run_start, down_right_run_end, down_right_run_next),
                    (up_left_run_start, up_left_run_end, up_left_run_next),
                    (down_left_run_start, down_left_run_end, down_left_run_next)
                ]

                for start_coords, end_coords, spot_past_end_coords in chain_coords:
                    chain = board.get_block(*start_coords, *end_coords)
                    if chain and all(spot in self.opponent_labels for spot in chain)\
                            and board.can_place(*spot_past_end_coords):
                        board.place_stone(x, y, self.play_labels.pop(0))
                        print('placed stone at ({}, {})'.format(x, y))
                        return (x, y)

        return None

    def try_build_winning_block(self, board):
        """Returns the coords of the move if the move was made, None otherwise"""
        # Check for winning blocks and place a stone in the winning block with the
        # largest number of the agent's stones

        # tuples of (block start coords, block end coords, block score)
        winning_block_tuple = None
        max_block_score = -1

        for x in range(board.width):
            for y in range(board.height):
                if not board.can_place(x, y): continue

                block_end_coords = [
                    (x-(CHAIN_LENGTH_TO_WIN-1),y-(CHAIN_LENGTH_TO_WIN-1)),
                    (x-(CHAIN_LENGTH_TO_WIN-1),y),
                    (x-(CHAIN_LENGTH_TO_WIN-1),y+(CHAIN_LENGTH_TO_WIN-1)),
                    (x,y-(CHAIN_LENGTH_TO_WIN-1)),
                    (x,y+(CHAIN_LENGTH_TO_WIN-1)),
                    (x+(CHAIN_LENGTH_TO_WIN-1),y-(CHAIN_LENGTH_TO_WIN-1)),
                    (x+(CHAIN_LENGTH_TO_WIN-1),y),
                    (x+(CHAIN_LENGTH_TO_WIN-1),y+(CHAIN_LENGTH_TO_WIN-1))
                ]

                for end_coords in block_end_coords:
                    block = board.get_block(x, y, *end_coords)
                    if block is None:
                        continue

                    block_score = 0
                    for spot in block:
                        if spot in self.labels:
                            block_score += 1
                        elif spot != EMPTY:
                            block_score = -1
                            break

                    block_tuple = ((x,y), end_coords, block_score)
                    if block_score > max_block_score:
                        winning_block_tuple = block_tuple
                        max_block_score = block_score

        if winning_block_tuple is None:
            return None

        start_coords, end_coords, _ = winning_block_tuple
        block_coords = board.get_block_coords(*start_coords, *end_coords)

        if max_block_score == 0:
            board.place_stone(*block_coords[0], self.play_labels.pop(0))
            print('placed stone at {}'.format(block_coords[0]))
            return block_coords[0]

        for idx, coords in enumerate(block_coords):
            if board.at(*coords) in self.labels:
                if idx-1 >= 0 and board.can_place(*block_coords[idx-1]):
                    board.place_stone(*block_coords[idx-1], self.play_labels.pop(0))
                    print('placed stone at {}'.format(block_coords[idx-1]))
                    return block_coords[idx-1]
                elif idx+1 < len(block_coords) and board.can_place(*block_coords[idx+1]):
                    board.place_stone(*block_coords[idx+1], self.play_labels.pop(0))
                    print('placed stone at {}'.format(block_coords[idx+1]))
                    return block_coords[idx+1]


class game_tree_search_agent:
    def __init__(self, play_labels, agent_labels, opponent_labels):
        self.play_labels = play_labels
        self.labels = agent_labels
        self.opponent_labels = opponent_labels
        self.nodes_expanded = 0

    def calculate_board_score(self, board):
        """Return a score representing how favorable the current board is to this agent.
        High scores represent desirable boards."""
        score = 0

        for x in range(board.width):
            for y in range(board.height):
                n = CHAIN_LENGTH_TO_WIN
                up_run_end = (x, y + n-1)
                right_run_end = (x + n-1, y)
                up_right_run_end = (x + n-1, y + n-1)
                down_right_run_end = (x + n-1, y - (n-1))

                chain_coords = [
                    up_run_end,
                    right_run_end,
                    up_right_run_end,
                    down_right_run_end,
                ]

                for end_coords in chain_coords:
                    block = board.get_block(x, y, *end_coords)
                    if not block: continue
                    if all(spot in self.labels for spot in block):
                        return 10000
                    elif all(spot in self.opponent_labels for spot in block):
                        return -10000
                    elif all(spot in self.opponent_labels or spot == EMPTY for spot in block):
                        score -= 40
                    elif all(spot in self.labels or spot == EMPTY for spot in block):
                        score += 50

        return score


class minimax_agent(game_tree_search_agent):
    def __init__(self, play_labels, agent_labels, opponent_labels):
        super().__init__(play_labels, agent_labels, opponent_labels)

    def make_move(self, board):
        """Returns True if the move was made, False otherwise"""
        # dict mapping from move to score of that move
        max_first_move_scores = {}
        for x in range(board.width):
            for y in range(board.height):
                if not board.can_place(x, y): continue
                board.place_stone(x, y, self.labels[0])

                # list of scores for moves that min can make after max made (x,y) move
                min_move_scores = []
                for xx in range(board.width):
                    for yy in range(board.height):
                        if not board.can_place(xx, yy): continue
                        board.place_stone(xx, yy, self.opponent_labels[0])

                        # list of scores for moves that max can make after
                        # max made (x,y) move and min made (xx,yy) move
                        max_second_move_scores = []
                        for xxx in range(board.width):
                            for yyy in range(board.height):
                                if not board.can_place(xxx, yyy): continue
                                board.place_stone(xxx, yyy, self.labels[0])

                                self.nodes_expanded += 1
                                score = self.calculate_board_score(board)
                                max_second_move_scores.append(score)
                                board.remove_stone(xxx, yyy)

                        min_move_scores.append(max(max_second_move_scores))
                        board.remove_stone(xx, yy)

                max_first_move_scores[(x,y)] = min(min_move_scores)
                board.remove_stone(x, y)

        if not max_first_move_scores:
            return False

        best_move = max(max_first_move_scores, key=lambda sequence: max_first_move_scores[sequence])

        board.place_stone(*best_move, self.play_labels.pop(0))
        print('placed stone at {}'.format(best_move))
        return best_move


class alpha_beta_agent(game_tree_search_agent):
    def __init__(self, play_labels, agent_labels, opponent_labels):
        super().__init__(play_labels, agent_labels, opponent_labels)

    def make_move(self, board):
        """Returns True if the move was made, False otherwise"""
        # dict mapping from move to score of that move
        max_first_move_scores = {}
        for x in range(board.width):
            for y in range(board.height):
                if not board.can_place(x, y): continue
                board.place_stone(x, y, self.labels[0])

                break_for = False

                # list of scores for moves that min can make after max made (x,y) move
                min_move_scores = []
                for xx in range(board.width):
                    if break_for: break
                    for yy in range(board.height):
                        if not board.can_place(xx, yy): continue
                        board.place_stone(xx, yy, self.opponent_labels[0])

                        # list of scores for moves that max can make after
                        # max made (x,y) move and min made (xx,yy) move
                        max_second_move_scores = []
                        for xxx in range(board.width):
                            for yyy in range(board.height):
                                if not board.can_place(xxx, yyy): continue
                                board.place_stone(xxx, yyy, self.labels[0])

                                self.nodes_expanded += 1
                                score = self.calculate_board_score(board)
                                max_second_move_scores.append(score)
                                board.remove_stone(xxx, yyy)
                        board.remove_stone(xx, yy)

                        min_move_score = max(max_second_move_scores)
                        if max_first_move_scores and min_move_score < max(max_first_move_scores.values()):
                            break_for = True
                            break
                        min_move_scores.append(min_move_score)

                print(min_move_scores)
                if min_move_scores:
                    max_first_move_scores[(x,y)] = min(min_move_scores)
                board.remove_stone(x, y)

        if not max_first_move_scores:
            return False

        best_move = max(max_first_move_scores, key=lambda sequence: max_first_move_scores[sequence])

        board.place_stone(*best_move, self.play_labels.pop(0))
        print('placed stone at {}'.format(best_move))
        return best_move


class board:
    def __init__(self, width, height):
        self.spots = [[EMPTY for _ in range(width)] for _ in range(height)]
        self.width = width
        self.height = height

    def is_in_bounds(self, x, y):
        return 0 <= y < len(self.spots) and 0 <= x < len(self.spots[y])

    def at(self, x, y):
        assert 0 <= y < len(self.spots)
        assert 0 <= x < len(self.spots[y])
        return self.spots[y][x]

    def get_block_coords(self, start_x, start_y, end_x, end_y):
        coords_gen = self.get_block_coords_gen(start_x, start_y, end_x, end_y)
        if coords_gen is None:
            return None
        return [coords for coords in coords_gen]

    def get_block_coords_gen(self, start_x, start_y, end_x, end_y):
        if not self.is_in_bounds(start_x, start_y) \
                or not self.is_in_bounds(end_x, end_y):
            return None

        if start_x == end_x:
            # vertical run
            if start_y < end_y:
                coords = ((start_x, y) for y in range(start_y, end_y+1))
            else:
                coords = ((start_x, y) for y in range(end_y, start_y+1))

        elif start_y == end_y:
            # horizontal run
            if start_x < end_x:
                coords = ((x, start_y) for x in range(start_x, end_x+1))
            else:
                coords = ((x, start_y) for x in range(end_x, start_x+1))

        elif abs(start_x - end_x) == abs(start_y - end_y):
            # diagonal run
            run_length = abs(start_x - end_x) + 1
            if start_y < end_y:
                if start_x < end_x:
                    # up right
                    coords = ((start_x+i, start_y+i) for i in range(run_length))
                else:
                    # up left
                    coords = ((start_x-i, start_y+i) for i in range(run_length))
            else:
                if start_x < end_x:
                    # down right
                    coords = ((start_x+i, start_y-i) for i in range(run_length))
                else:
                    # down left
                    coords = ((start_x-i, start_y-i) for i in range(run_length))

        else:
            return None

        return coords

    def get_block(self, start_x, start_y, end_x, end_y):
        coords = self.get_block_coords_gen(start_x, start_y, end_x, end_y)
        if coords is None:
            return None

        run = [self.at(x, y) for x, y in coords]
        return run

    def has_chain(self, start_x, start_y, end_x, end_y, chain_labels):
        chain = self.get_block(start_x, start_y, end_x, end_y)
        return all(spot in chain_labels for spot in chain) if chain is not None else False

    def num_n_chains(self, n, chain_labels):
        num_found = 0

        for x in range(self.width):
            for y in range(self.height):
                up_run_end = (x, y + n-1)
                right_run_end = (x + n-1, y)
                up_right_run_end = (x + n-1, y + n-1)
                down_right_run_end = (x + n-1, y - (n-1))

                chain_coords = [
                    up_run_end,
                    right_run_end,
                    up_right_run_end,
                    down_right_run_end,
                ]

                for end_coords in chain_coords:
                    if self.has_chain(x, y, *end_coords, chain_labels):
                        num_found += 1

        return num_found

    def check_winner(self):
        if self.num_n_chains(CHAIN_LENGTH_TO_WIN, RED_LABELS) > 0:
            return RED
        if self.num_n_chains(CHAIN_LENGTH_TO_WIN, BLUE_LABELS) > 0:
            return BLUE

    def can_place(self, x, y):
        return self.is_in_bounds(x, y) and self.spots[y][x] == EMPTY

    def place_stone(self, x, y, stone_label):
        assert self.can_place(x, y)
        self.spots[y][x] = stone_label

    def remove_stone(self, x, y):
        assert self.is_in_bounds(x, y)
        assert self.at(x, y) != EMPTY
        self.spots[y][x] = EMPTY

    def __repr__(self):
        string = []
        for row_idx, row in enumerate(reversed(self.spots)):
            row_idx = len(self.spots) - row_idx - 1
            string.append('{} '.format(str(row_idx)))
            string.append(' '.join(spot for spot in row))
            string.append('\n')
        string.append(' ')
        for col_idx in range(len(self.spots[-1])):
            string.append(' {}'.format(str(col_idx)))
        string.append('\n')
        return ''.join(string)


def main():
    #play_reflex_vs_reflex()
    play(minimax_agent, reflex_agent)
    #play(reflex_agent, alpha_beta_agent)

def play(red_agent_type, blue_agent_type):
    b = board(BOARD_WIDTH, BOARD_HEIGHT)
    print(b)

    red_agent = red_agent_type(play_labels=RED_PLAY_LABELS, agent_labels=RED_LABELS,
                                opponent_labels=BLUE_LABELS)

    blue_agent = blue_agent_type(play_labels=BLUE_PLAY_LABELS, agent_labels=BLUE_LABELS,
                                opponent_labels=RED_LABELS)

    turn = 0
    while True:
        move = None
        if turn % 2 == 0:
            print("red's turn")
            move = red_agent.make_move(b)
        else:
            print("blue's turn")
            move = blue_agent.make_move(b)
        print(b)
        winner = b.check_winner()
        if winner:
            print('{} won!'.format(winner))
            break
        turn += 1
        if not move:
            break

    print('red agent expanded {} nodes'.format(red_agent.nodes_expanded))
    print('blue agent expanded {} nodes'.format(blue_agent.nodes_expanded))


def play_reflex_vs_reflex():
    b = board(BOARD_WIDTH, BOARD_HEIGHT)
    print(b)

    b.place_stone(1, 1, RED_PLAY_LABELS.pop(0))
    b.place_stone(5, 5, BLUE_PLAY_LABELS.pop(0))

    red_agent = reflex_agent(play_labels=RED_PLAY_LABELS,
                                agent_labels=RED_LABELS, opponent_labels=BLUE_LABELS)
    blue_agent = reflex_agent(play_labels=BLUE_PLAY_LABELS,
                                agent_labels=BLUE_LABELS, opponent_labels=RED_LABELS)
    turn = 0
    while True:
        move = None
        if turn % 2 == 0:
            print("red's turn")
            move = red_agent.make_move(b)
        else:
            print("blue's turn")
            move = blue_agent.make_move(b)
        print(b)
        winner = b.check_winner()
        if winner:
            print('{} won!'.format(winner))
            break
        turn += 1
        if not move:
            break

if __name__ == '__main__':
    main()
