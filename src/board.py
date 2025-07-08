import copy

from src.const import *
from src.square import Square
from src.piece import *
from src.move import Move

class Board:
    def __init__(self):
        self.squares = [[0, 0, 0, 0, 0, 0, 0, 0, 0] for _ in range(ROWS)]
        self.last_move = None
        self._create()
        self._add_pieces('red')
        self._add_pieces('black')

    def move(self, piece, move):
        initial = move.initial
        final = move.final

        # Update the board
        self.squares[initial.row][initial.col].piece = None
        self.squares[final.row][final.col].piece = piece

        # Move
        piece.moved = True
        # Clear after move
        piece.clear_moves()

        # Set last move
        self.last_move = move

    def valid_move(self, piece, move):
        return move in piece.moves_rival or move in piece.moves_empty

    def checkmate(self, piece, move):
        # Giả lập bàn cờ sau khi đi nước "move"
        temp_board = copy.deepcopy(self)
        temp_piece = copy.deepcopy(piece)
        temp_board.move(temp_piece, move)

        # Tìm tướng bên mình sau nước đi
        general_row, general_col = None, None
        for r in range(ROWS):
            for c in range(COLS):
                p = temp_board.squares[r][c].piece
                if isinstance(p, General) and p.color == piece.color:
                    general_row, general_col = r, c
                    break

        # Kiểm tra quân địch có thể ăn tướng không
        for r in range(ROWS):
            for c in range(COLS):
                if temp_board.squares[r][c].has_rival_piece(piece.color):
                    enemy = temp_board.squares[r][c].piece
                    temp_board.cal_move(enemy, r, c, bool=False)
                    for m in enemy.moves_rival:
                        if m.final.row == general_row and m.final.col == general_col:
                            return True  # tướng bị chiếu

        return False  # an toàn

    def general_facing_each_other(self):
        red_pos = None
        black_pos = None

        # Tìm vị trí 2 tướng
        for r in range(ROWS):
            for c in range(COLS):
                piece = self.squares[r][c].piece
                if isinstance(piece, General):
                    if piece.color == 'red':
                        red_pos = (r, c)
                    elif piece.color == 'black':
                        black_pos = (r, c)

        if red_pos and black_pos and red_pos[1] == black_pos[1]:  # cùng cột
            col = red_pos[1]
            start = min(red_pos[0], black_pos[0]) + 1
            end = max(red_pos[0], black_pos[0])
            for r in range(start, end):
                if not self.squares[r][col].is_empty():
                    return False  # Có quân chặn giữa → không đối mặt
            return True  # Không có quân chặn → tướng đối mặt
        return False


    def cal_move(self, piece, row, col, bool=True):
        # Calculate all the posible valid moves of an specific piece
        def horse_move():
    # 8 possible moves with their corresponding "leg" blocks
            possible_moves = [
                ((row+2, col+1), (row+1, col)),  # down 2, right 1
                ((row+2, col-1), (row+1, col)),  # down 2, left 1
                ((row+1, col+2), (row, col+1)),  # down 1, right 2
                ((row+1, col-2), (row, col-1)),  # down 1, left 2
                ((row-1, col+2), (row, col+1)),  # up 1, right 2
                ((row-1, col-2), (row, col-1)),  # up 1, left 2
                ((row-2, col+1), (row-1, col)),  # up 2, right 1
                ((row-2, col-1), (row-1, col))   # up 2, left 1
            ]

            for (possible_row, possible_col), (block_row, block_col) in possible_moves:
                if not Square.in_range_other(possible_row, possible_col):
                    continue
                # Kiểm tra bị chắn chân mã
                if not self.squares[block_row][block_col].is_empty():
                    continue  # chân mã bị chặn

                initial = Square(row, col)

                if self.squares[possible_row][possible_col].is_empty():
                    final = Square(possible_row, possible_col)
                    move = Move(initial, final)
                    if bool:
                        if not self.checkmate(piece, move):
                            piece.add_move_empty(move)
                    else:
                        piece.add_move_empty(move)

                elif self.squares[possible_row][possible_col].has_rival_piece(piece.color):
                    final_piece = self.squares[possible_row][possible_col].piece
                    final = Square(possible_row, possible_col, final_piece)
                    move = Move(initial, final)
                    if bool:
                        if not self.checkmate(piece, move):
                            piece.add_move_rival(move)
                    else:
                        piece.add_move_rival(move)

                        

        def pawn_move():
            if piece.color == 'red' and row > 4:
                possible_move = (row-1, col)
                possible_row, possible_col = possible_move
                if Square.in_range_other(possible_row, possible_col):
                    if self.squares[possible_row][possible_col].is_empty():
                        # Create new moves
                        initial = Square(row, col)
                        final = Square(possible_row, possible_col)
                        move = Move(initial, final)
                        
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_empty(move)
                        else:
                            piece.add_move_empty(move)
                            
                    elif self.squares[possible_row][possible_col].has_rival_piece(piece.color):
                        # Create new moves
                        initial = Square(row, col)
                        final = Square(possible_row, possible_col)
                        move = Move(initial, final)
                        
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_rival(move)
                        else:
                            piece.add_move_rival(move)

            elif piece.color == 'red' and row <= 4:
                possible_moves = [
                    (row-1, col),
                    (row, col+1),
                    (row, col-1)            
                ]
                for possible_move in possible_moves:
                    possible_row, possible_col = possible_move
                    if Square.in_range_other(possible_row, possible_col):
                        if self.squares[possible_row][possible_col].is_empty():
                            # Create new moves
                            initial = Square(row, col)
                            final = Square(possible_row, possible_col)
                            move = Move(initial, final)
                            
                            if bool:
                                if not self.checkmate(piece, move):
                                    piece.add_move_empty(move)
                            else:
                                piece.add_move_empty(move)
                                
                        elif self.squares[possible_row][possible_col].has_rival_piece(piece.color):
                            # Create new moves
                            initial = Square(row, col)
                            final = Square(possible_row, possible_col)
                            move = Move(initial, final)
                            
                            if bool:
                                if not self.checkmate(piece, move):
                                    piece.add_move_rival(move)
                            else:
                                piece.add_move_rival(move)

            elif piece.color == 'black' and row >= 5:
                possible_moves = [
                    (row+1, col),
                    (row, col+1),
                    (row, col-1)            
                ]
                for possible_move in possible_moves:
                    possible_row, possible_col = possible_move
                    if Square.in_range_other(possible_row, possible_col):
                        if self.squares[possible_row][possible_col].is_empty():
                            # Create new moves
                            initial = Square(row, col)
                            final = Square(possible_row, possible_col)
                            move = Move(initial, final)
                            
                            if bool:
                                if not self.checkmate(piece, move):
                                    piece.add_move_empty(move)
                            else:
                                piece.add_move_empty(move)
                                
                        elif self.squares[possible_row][possible_col].has_rival_piece(piece.color):
                            # Create new moves
                            initial = Square(row, col)
                            final = Square(possible_row, possible_col)
                            move = Move(initial, final)
                            
                            if bool:
                                if not self.checkmate(piece, move):
                                    piece.add_move_rival(move)
                            else:
                                piece.add_move_rival(move)

            elif piece.color == 'black' and row <= 5:
                possible_move = (row+1, col)
                possible_row, possible_col = possible_move
                if Square.in_range_other(possible_row, possible_col):
                    if self.squares[possible_row][possible_col].is_empty():
                        # Create new moves
                        initial = Square(row, col)
                        final = Square(possible_row, possible_col)
                        move = Move(initial, final)
                        
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_empty(move)
                        else:
                            piece.add_move_empty(move)
                            
                    elif self.squares[possible_row][possible_col].has_rival_piece(piece.color):
                        # Create new moves
                        initial = Square(row, col)
                        final = Square(possible_row, possible_col)
                        move = Move(initial, final)
                        
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_rival(move)
                        else:
                            piece.add_move_rival(move)

        def general_move():
            possible_moves = [
                (row+1, col),
                (row-1, col),
                (row, col+1),
                (row, col-1)            
            ]

            for r, c in possible_moves:
                if Square.in_range_advisor_general(piece.color, r, c):
                    final_square = self.squares[r][c]
                    initial = Square(row, col)
                    final = Square(r, c, final_square.piece)
                    move = Move(initial, final)

                    # Kiểm tra nếu là bước di chuyển an toàn
                    temp_board = copy.deepcopy(self)
                    temp_piece = copy.deepcopy(piece)
                    temp_board.move(temp_piece, move)

                    if not temp_board.general_facing_each_other():
                        if bool:
                            if not self.checkmate(piece, move):
                                if final_square.is_empty():
                                    piece.add_move_empty(move)
                                elif final_square.has_rival_piece(piece.color):
                                    piece.add_move_rival(move)
                        else:
                            if final_square.is_empty():
                                piece.add_move_empty(move)
                            elif final_square.has_rival_piece(piece.color):
                                piece.add_move_rival(move)

        def advisor_move():
            possible_moves = [
                (row+1, col+1),
                (row+1, col-1),
                (row-1, col+1),
                (row-1, col-1)            
            ]
            for possible_move in possible_moves:
                possible_row, possible_col = possible_move
                if Square.in_range_advisor_general(piece.color, possible_row, possible_col):
                    if self.squares[possible_row][possible_col].is_empty():
                        # Create new moves
                        initial = Square(row, col)
                        final = Square(possible_row, possible_col)
                        move = Move(initial, final)
                        
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_empty(move)
                        else:
                            piece.add_move_empty(move)
                            
                    elif self.squares[possible_row][possible_col].has_rival_piece(piece.color):
                        # Create new moves
                        initial = Square(row, col)
                        final = Square(possible_row, possible_col)
                        move = Move(initial, final)
                        
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_rival(move)
                        else:
                            piece.add_move_rival(move)

        def elephant_move():
            possible_moves = [
                (row + 2, col + 2),
                (row + 2, col - 2),
                (row - 2, col + 2),
                (row - 2, col - 2)
            ]

            for possible_row, possible_col in possible_moves:
                # Kiểm tra ô kết thúc hợp lệ với màu quân tượng
                if not Square.in_range_elephant(piece.color, possible_row):
                    continue
                
                # Tính tọa độ ô chắn giữa (con mắt tượng)
                block_row = row + (possible_row - row) // 2
                block_col = col + (possible_col - col) // 2

                # Kiểm tra ô chắn nằm trong phạm vi bàn cờ
                if not Square.in_range_other(block_row, block_col):
                    continue
                
                # Nếu ô chắn bị chặn thì không đi được
                if not self.squares[block_row][block_col].is_empty():
                    continue
                
                initial = Square(row, col)
                final_square = self.squares[possible_row][possible_col]
                final = Square(possible_row, possible_col, final_square.piece)
                move = Move(initial, final)

                # Nếu ô kết thúc trống hoặc có quân đối phương thì thêm nước đi
                if final_square.is_empty():
                    if bool:
                        if not self.checkmate(piece, move):
                            piece.add_move_empty(move)
                    else:
                        piece.add_move_empty(move)
                elif final_square.has_rival_piece(piece.color):
                    if bool:
                        if not self.checkmate(piece, move):
                            piece.add_move_rival(move)
                    else:
                        piece.add_move_rival(move)

        
        def chariot_move():
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, Xuống, Trái, Phải

            for d_row, d_col in directions:
                r, c = row + d_row, col + d_col

                while Square.in_range_other(r, c):
                    target_square = self.squares[r][c]
                    initial = Square(row, col)
                    final = Square(r, c)
                    move = Move(initial, final)

                    if target_square.is_empty():
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_empty(move)
                        else:
                            piece.add_move_empty(move)

                    elif target_square.has_rival_piece(piece.color):
                        if bool:
                            if not self.checkmate(piece, move):
                                piece.add_move_rival(move)
                        else:
                            piece.add_move_rival(move)
                        break  # Gặp quân đối phương -> có thể ăn -> dừng tại đây

                    else:
                        break  # Gặp quân mình -> không được đi tiếp

                    r += d_row
                    c += d_col

        def cannon_move():
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Lên, Xuống, Trái, Phải

            for d_row, d_col in directions:
                r, c = row + d_row, col + d_col
                jumped = False  # Đã nhảy qua quân chưa

                while Square.in_range_other(r, c):
                    target_square = self.squares[r][c]
                    initial = Square(row, col)
                    final = Square(r, c)
                    move = Move(initial, final)

                    if target_square.is_empty():
                        if not jumped:
                            # Chưa nhảy -> có thể di chuyển bình thường
                            if bool:
                                if not self.checkmate(piece, move):
                                    piece.add_move_empty(move)
                            else:
                                piece.add_move_empty(move)
                        # Đã nhảy -> không thể di chuyển vào ô trống
                        
                    elif target_square.has_rival_piece(piece.color):
                        if jumped:
                            # Đã nhảy -> có thể ăn quân này
                            if bool:
                                if not self.checkmate(piece, move):
                                    piece.add_move_rival(move)
                            else:
                                piece.add_move_rival(move)
                            break  # Ăn quân -> dừng
                        else:
                            # Chưa nhảy -> đây là quân để nhảy
                            jumped = True
                            
                    else:
                        # Gặp quân mình
                        if not jumped:
                            jumped = True  # Đây là quân để nhảy
                        else:
                            break  # Đã nhảy và gặp quân mình -> dừng

                    r += d_row
                    c += d_col

        if piece.name == 'pawn':
            pawn_move()
        if piece.name == 'general':
            general_move()
        if piece.name == 'advisor':
            advisor_move()
        if piece.name == 'elephant':
            elephant_move()
        if piece.name == 'horse':
            horse_move()
        if piece.name == 'chariot':
            chariot_move()
        if piece.name == 'cannon':
            cannon_move()

    # Private method (call only inside class)
    def _create(self):
        for row in range(ROWS):
            for col in range(COLS):
                self.squares[row][col] = Square(row, col)

    def _add_pieces(self, color):
        if color == 'black':
            row_pawn, row_cannon, row_other = (3, 2, 0)
        else:
            row_pawn, row_cannon, row_other = (6, 7, 9)

        # Pawn
        for col in [0, 2, 4, 6, 8]:
            self.squares[row_pawn][col] = Square(row_pawn, col, Pawn(color))

        # Cannon
        for col in [1, 7]:
            self.squares[row_cannon][col] = Square(row_cannon, col, Cannon(color))
        
        # Chariot
        for col in [0, 8]:
            self.squares[row_other][col] = Square(row_other, col, Chariot(color))

        # Horse
        for col in [1, 7]:
            self.squares[row_other][col] = Square(row_other, col, Horse(color))

        # Elephant
        for col in [2, 6]:
            self.squares[row_other][col] = Square(row_other, col, Elephant(color))

        Advisor
        for col in [3, 5]:
            self.squares[row_other][col] = Square(row_other, col, Advisor(color))

        self.squares[row_other][4] = Square(row_other, 4, General(color))

        