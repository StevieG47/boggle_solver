# Given board of letters, return possible words
# Ref: https://github.com/readjfb/boggle_player

def adjacent(val1, val2, lst):
    try:
        return (lst.index(val1) - lst.index(val2) in [-1, 1])
    except ValueError:
        return False
    return False


def solve_board(board, dictionary):
    def turtle(board, x, y, inv_spaces, letters=''):
        # crossing lets the path cross over itself diagonaly [if true];
        # min_num_lett is the min number of letters for a word
        crossing, min_num_lett = True, 3

        inv_spaces.append((x, y))
        letters += str(board[x][y])

        # checking to see if the current path has made a letter, and if it has
        # been the right length
        if dictionary.contains(letters) and len(letters) >= min_num_lett:
            all_words.add(letters)

        # if there are no more possible words
        elif not dictionary.one_autocomplete(letters):
            return 0
        '''
        p0 p1 p2
        p3 X  p4
        p5 p6 p7

        '''
        p0 = (x - 1, y - 1)
        p1 = (x, y - 1)
        p2 = (x + 1, y - 1)
        p3 = (x - 1, y)
        p4 = (x + 1, y)
        p5 = (x - 1, y + 1)
        p6 = (x, y + 1)
        p7 = (x + 1, y + 1)

        if p4 not in inv_spaces:
            turtle(board, *p4, list(inv_spaces), str(letters))

        if p3 not in inv_spaces:
            turtle(board, *p3, list(inv_spaces), str(letters))

        if p1 not in inv_spaces:
            turtle(board, *p1, list(inv_spaces), str(letters))

        if p6 not in inv_spaces:
            turtle(board, *p6, list(inv_spaces), str(letters))

        if not crossing:
            if p0 not in inv_spaces and not adjacent(p3, p1, inv_spaces):
                turtle(board, *p0, list(inv_spaces), str(letters))

            if p2 not in inv_spaces and not adjacent(p4, p1, inv_spaces):
                turtle(board, *p2, list(inv_spaces), str(letters))

            if p5 not in inv_spaces and not adjacent(p3, p6, inv_spaces):
                turtle(board, *p5, list(inv_spaces), str(letters))

            if p7 not in inv_spaces and not adjacent(p4, p6, inv_spaces):
                turtle(board, *p7, list(inv_spaces), str(letters))

        else:
            #diagonals, allows crossing over
            if p0 not in inv_spaces:
                turtle(board, *p0, list(inv_spaces), str(letters))

            if p2 not in inv_spaces:
                turtle(board, *p2, list(inv_spaces), str(letters))

            if p5 not in inv_spaces:
                turtle(board, *p5, list(inv_spaces), str(letters))

            if p7 not in inv_spaces:
                turtle(board, *p7, list(inv_spaces), str(letters))

    # all_words is global, as lists are global by default in python
    all_words = set([])
    # board exterior is a list of the points on the exterior of the board, such
    # that the turtle won't go to them
    board_exterior = [(-1, -1)]
    for i in range(len(board) + 1):
        board_exterior.append((-1, i))
        board_exterior.append((len(board), i))
        board_exterior.append((i, -1))
        board_exterior.append((i, len(board)))

    for x, y_l in enumerate(board):
        for y, x_l in enumerate(y_l):
            turtle(board, x, y, [i for i in board_exterior])
    return all_words