
"""
I modified this code from https://github.com/everestwitman/Pygame-Checkers/blob/master/checkers.py
"""

"""
checkers.py

A simple checkers engine written in Python with the pygame 1.9.1 libraries.

Here are the rules I am using: http://boardgames.about.com/cs/checkersdraughts/ht/play_checkers.htm

I adapted some code from checkers.py found at 
http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py starting on line 159 of my program.

This is the final version of my checkers project for Programming Workshop at Marlboro College. The entire thing has been rafactored and made almost completely object oriented.

Funcitonalities include:

- Having the pieces and board drawn to the screen

- The ability to move pieces by clicking on the piece you want to move, then clicking on the square you would
  like to move to. You can change you mind about the piece you would like to move, just click on a new piece of yours.

- Knowledge of what moves are legal. When moving pieces, you'll be limited to legal moves.

- Capturing

- DOUBLE capturing etc.

- Legal move and captive piece highlighting

- Turn changes

- Automatic kinging and the ability for them to move backwords

- Automatic check for and end game. 

- A silky smoooth 60 FPS!

Everest Witman - May 2014 - Marlboro College - Programming Workshop 
"""

import pygame, sys
import numpy as np
from pygame.locals import *
import time
from muzero_agent import *


pygame.font.init()

##COLORS##
#             R    G    B 
WHITE    = (255, 255, 255)
BLUE     = (  0,   0, 255)
RED      = (255,   0,   0)
BLACK    = (  0,   0,   0)
GOLD     = (255, 215,   0)
HIGH     = (160, 190, 255)

##DIRECTIONS##
NORTHWEST = "northwest"
NORTHEAST = "northeast"
SOUTHWEST = "southwest"
SOUTHEAST = "southeast"


class Player:
	"""
	An agent to interact with the checkers environment
	"""
	def __init__(self, type="random_computer", color=BLUE, name=None, agent=None):
		self.name = name
		self.type = type
		self.color = color
		if "computer" in type and agent==None:
			self.Agent = None
		else:
			self.Agent = agent
		if type == "human":
			self.policy = None

	def evaluate(self, state, masked_actions, player_turn = None, reward=0, winner=None):
		if winner is None:
			return np.random.choice(np.where(masked_actions == 1)[0])
		else:
			return None


class Game:
	"""
	The main game control.
	"""

	def __init__(self, player_1_type, player_2_type, delay=.01, show_graphics=True, print_board=False, agent1=None, agent2=None):
		self.show_graphics = show_graphics
		if self.show_graphics:
			self.graphics = Graphics()
		self.board = Board()
		self.pos_map = {0: (-2, -2), 1: (-2, 2), 2: (-1, -1), 3: (-1, 1), 4: (1, -1), 5: (1, 1), 6: (2, -2), 7: (2, 2)}
		self.inv_pos_map = {v: k for k, v in self.pos_map.items()}
		self.turn = BLUE
		self.selected_piece = None # a board location. 
		self.hop = False
		self.selected_legal_moves = []
		self.players = [Player(type=player_1_type, color=BLUE, agent=agent1), Player(type=player_2_type, color=RED, agent=agent2)]
		self.print_board = print_board
		self.delay = delay
		self.game_over = False
		self.winner = None

	def setup(self):
		"""Draws the window and board at the beginning of the game"""
		self.graphics.setup_window()

	def human_event_loop(self):
		"""
		The event loop. This is where events are triggered 
		(like a mouse click) and then effect the game state.
		"""
		if self.game_over:
			return None

		assert self.show_graphics == True
		x = pygame.mouse.get_pos()[0]
		y = pygame.mouse.get_pos()[1]
		self.mouse_pos = self.graphics.board_coords(x, y) # what square is the mouse in?
		self.mouse_pos = int(self.mouse_pos[0]), int(self.mouse_pos[1])
		if self.selected_piece != None:
			self.selected_legal_moves = self.board.legal_moves(self.selected_piece, self.hop)

		for event in pygame.event.get():

			if event.type == QUIT:
				self.terminate_game()

			if event.type == MOUSEBUTTONDOWN:
				if self.hop == False:
					if self.board.location(self.mouse_pos).occupant != None and self.board.location(self.mouse_pos).occupant.color == self.turn:
						self.selected_piece = int(self.mouse_pos[0]), int(self.mouse_pos[1])

					elif self.selected_piece != None and self.mouse_pos in self.board.legal_moves(self.selected_piece):
						x, y = int(self.mouse_pos[0]), int(self.mouse_pos[1])
						self.board.move_piece(self.selected_piece, (x,y))
					
						if self.mouse_pos not in self.board.adjacent(self.selected_piece):
							self.board.remove_piece((self.selected_piece[0] + (self.mouse_pos[0] - self.selected_piece[0]) / 2, self.selected_piece[1] + (self.mouse_pos[1] - self.selected_piece[1]) / 2))
						
							self.hop = True
							self.selected_piece = self.mouse_pos

						else:
							self.end_turn()

				if self.hop == True:					
					if self.selected_piece != None and self.mouse_pos in self.board.legal_moves(self.selected_piece, self.hop):
						self.board.move_piece(self.selected_piece, self.mouse_pos)
						self.board.remove_piece((self.selected_piece[0] + (self.mouse_pos[0] - self.selected_piece[0]) / 2, self.selected_piece[1] + (self.mouse_pos[1] - self.selected_piece[1]) / 2))

					if self.board.legal_moves(self.mouse_pos, self.hop) == []:
							self.end_turn()

					else:
						self.selected_piece = self.mouse_pos

	def action_dict_to_array(self, action_dict):
		"""
		Converts the legal moves from the action dictionary into an array that the computer can easily handle
		:param action_dict:
		:return:
		"""
		available_actions = np.zeros(64 * 8)
		available_pieces = list(action_dict.keys())
		col_size = 8
		for i, piece in enumerate(available_pieces):
			legal_moves_for_piece = action_dict[piece]
			for j, available_action in enumerate(legal_moves_for_piece):
				unraveled_piece_position = (piece[0] * col_size + piece[1])
				movement = (available_action[0] - piece[0], available_action[1] - piece[1])
				pos = self.inv_pos_map[movement]
				available_actions[unraveled_piece_position * 8 + pos] = 1
		return available_actions


	def interpret_action(self, action):
		"""
		Interprets the action (an int) and changes that to a selected piece and selected move
		:param action:
		:return:
		"""
		unraveled_piece_pos = action // 8
		selected_piece = (unraveled_piece_pos // 8, unraveled_piece_pos % 8)

		unraveled_move_pos = action % 8
		i, j = self.pos_map[unraveled_move_pos]

		selected_move = (selected_piece[0] + i,  selected_piece[1] + j)

		return selected_piece, selected_move

	def computer_event_loop(self):
		"""
				The event loop. This is where events are triggered
				(like a mouse click) and then effect the game state.
				"""
		# x = pygame.mouse.get_pos()[0]
		# y = pygame.mouse.get_pos()[1]
		# self.mouse_pos = self.graphics.board_coords(x, y)  # what square is the mouse in?
		# self.mouse_pos = int(self.mouse_pos[0]), int(self.mouse_pos[1])
		self.selected_piece = None

		while True:
			if self.delay > 0:
				time.sleep(self.delay)
			all_legal_moves = self.board.all_legal_moves(self.current_player.color, self.hop, self.selected_piece)
			action_array = self.action_dict_to_array(all_legal_moves)

			if self.current_player.Agent is not None:
				action = self.current_player.Agent.evaluate(self.board.get_board(), action_array, self.current_player.color, winner=self.winner)
			else:
				action = self.current_player.evaluate(self.board.get_board(), action_array, self.current_player.color, winner=self.winner)

			if self.game_over:
				break

			self.selected_piece, self.selected_move = self.interpret_action(action)

			if self.hop == False:
				if self.selected_piece != None and self.selected_move in self.board.legal_moves(self.selected_piece):
					self.board.move_piece(self.selected_piece, self.selected_move)

					if self.selected_move not in self.board.adjacent(self.selected_piece):
						self.board.remove_piece(((self.selected_piece[0] + self.selected_move[0]) // 2,
																		 (self.selected_piece[1] + self.selected_move[1]) // 2))

						self.hop = True
						self.selected_piece = self.selected_move

						if self.board.legal_moves(self.selected_piece, self.hop) == []:
							break
						else:
							continue
					else:
						break
				else:
					raise AttributeError("Selected piece not none or selected move not available when hop is False.")

			if self.hop == True:
				if self.selected_piece != None and self.selected_move in self.board.legal_moves(self.selected_piece, self.hop):
					self.board.move_piece(self.selected_piece, self.selected_move)
					self.board.remove_piece((self.selected_piece[0] + (self.selected_move[0] - self.selected_piece[0]) / 2,
																	 self.selected_piece[1] + (self.selected_move[1] - self.selected_piece[1]) / 2))

				if self.board.legal_moves(self.selected_move, self.hop) == []:
					break
				else:
					self.selected_piece = self.selected_move

		self.end_turn() # If we break out of the while loop that means we have ended the turn

	def update(self):
		"""Calls on the graphics class to update the game display."""
		# if self.selected_piece is not None:
		# 	self.selected_piece = int(self.selected_piece[0]), int(self.selected_piece[1])

		if self.show_graphics:
			self.graphics.update_display(self.board, self.selected_legal_moves, self.selected_piece)
			if self.game_over:
				time.sleep(1)
		else:
			if self.print_board:
				print(self.board.get_board())

	def terminate_game(self):
		"""Quits the program and ends the game."""
		pygame.quit()
		sys.exit

	def get_current_player(self, player_color):
		for player in self.players:
			if player.color == player_color:
				return player

	def play(self):
		""""This executes the game and controls its flow."""
		if self.show_graphics:
			self.setup()
		moves = 0
		players_acknowledged_winner = 0
		while True: # main game loop

			self.current_player = self.get_current_player(self.turn)
			if self.current_player.type == "human":
				self.human_event_loop()
			else:
				self.computer_event_loop()

			self.update()

			if self.game_over:
				# Give both players the chance to acknowledge the winner
				if players_acknowledged_winner < 2:
					players_acknowledged_winner += 1
				else:
					if self.show_graphics:
						self.terminate_game()
					break

			moves += 1

		return self.winner, moves

	def end_turn(self):
		"""
		End the turn. Switches the current player. 
		end_turn() also checks for and game and resets a lot of class attributes.
		"""
		if self.turn == BLUE:
			self.turn = RED
		else:
			self.turn = BLUE

		self.selected_piece = None
		self.selected_legal_moves = []
		self.hop = False

		if self.check_for_endgame():
			if self.turn == BLUE:
				if self.show_graphics:
					self.graphics.draw_message("RED WINS!")
				else:
					if self.print_board:
						print("RED WINS")
				self.winner = RED
				self.game_over = True
			else:
				if self.show_graphics:
					self.graphics.draw_message("BLUE WINS!")
				else:
					if self.print_board:
						print("BLUE WINS")
				self.winner = BLUE
				self.game_over = True

	def check_for_endgame(self):
		"""
		Checks to see if a player has run out of moves or pieces. If so, then return True. Else return False.
		"""
		for x in range(8):
			for y in range(8):
				if self.board.location((x,y)).color == BLACK and self.board.location((x,y)).occupant != None and self.board.location((x,y)).occupant.color == self.turn:
					if self.board.legal_moves((x,y)) != []:
						return False

		return True

class Graphics:
	def __init__(self):
		self.caption = "Checkers"

		self.fps = 60
		self.clock = pygame.time.Clock()

		self.window_size = 600
		self.screen = pygame.display.set_mode((self.window_size, self.window_size))
		self.background = pygame.image.load('Resources/board.png')

		self.square_size = self.window_size / 8
		self.piece_size = self.square_size / 2

		self.message = False

	def setup_window(self):
		"""
		This initializes the window and sets the caption at the top.
		"""
		pygame.init()
		pygame.display.set_caption(self.caption)

	def update_display(self, board, legal_moves, selected_piece):
		"""
		This updates the current display.
		"""
		self.screen.blit(self.background, (0,0))
		
		self.highlight_squares(legal_moves, selected_piece)
		self.draw_board_pieces(board)

		if self.message:
			self.screen.blit(self.text_surface_obj, self.text_rect_obj)

		pygame.display.update()
		self.clock.tick(self.fps)

	def draw_board_squares(self, board):
		"""
		Takes a board object and draws all of its squares to the display
		"""
		for x in range(8):
			for y in range(8):
				pygame.draw.rect(self.screen, board[x][y].color, (x * self.square_size, y * self.square_size, self.square_size, self.square_size), )
	
	def draw_board_pieces(self, board):
		"""
		Takes a board object and draws all of its pieces to the display
		"""
		for x in range(8):
			for y in range(8):
				if board.matrix[x][y].occupant != None:
					pygame.draw.circle(self.screen, board.matrix[x][y].occupant.color, self.pixel_coords((x,y)), int(self.piece_size))

					if board.location((x,y)).occupant.king == True:
						pygame.draw.circle(self.screen, GOLD, self.pixel_coords((x,y)), int (self.piece_size / 1.7), int(self.piece_size / 4))


	def pixel_coords(self, board_coords):
		"""
		Takes in a tuple of board coordinates (x,y) 
		and returns the pixel coordinates of the center of the square at that location.
		"""
		return (int(board_coords[0] * self.square_size + self.piece_size), int(board_coords[1] * self.square_size + self.piece_size))

	def board_coords(self, pixel_x, pixel_y):
		"""
		Does the reverse of pixel_coords(). Takes in a tuple of of pixel coordinates and returns what square they are in.
		"""
		return (pixel_x / self.square_size, pixel_y / self.square_size)	

	def highlight_squares(self, squares, origin):
		"""
		Squares is a list of board coordinates. 
		highlight_squares highlights them.
		"""
		for square in squares:
			pygame.draw.rect(self.screen, HIGH, (square[0] * self.square_size, square[1] * self.square_size, self.square_size, self.square_size))	

		if origin != None:
			pygame.draw.rect(self.screen, HIGH, (origin[0] * self.square_size, origin[1] * self.square_size, self.square_size, self.square_size))

	def draw_message(self, message):
		"""
		Draws message to the screen. 
		"""
		self.message = True
		self.font_obj = pygame.font.Font('freesansbold.ttf', 44)
		self.text_surface_obj = self.font_obj.render(message, True, HIGH, BLACK)
		self.text_rect_obj = self.text_surface_obj.get_rect()
		self.text_rect_obj.center = (self.window_size / 2, self.window_size / 2)

class Board:
	def __init__(self):
		self.matrix = self.new_board()

	def new_board(self):
		"""
		Create a new board matrix.
		"""

		# initialize squares and place them in matrix

		matrix = [[None] * 8 for i in range(8)]

		# The following code block has been adapted from
		# http://itgirl.dreamhosters.com/itgirlgames/games/Program%20Leaders/ClareR/Checkers/checkers.py
		for x in range(8):
			for y in range(8):
				if (x % 2 != 0) and (y % 2 == 0):
					matrix[y][x] = Square(WHITE)
				elif (x % 2 != 0) and (y % 2 != 0):
					matrix[y][x] = Square(BLACK)
				elif (x % 2 == 0) and (y % 2 != 0):
					matrix[y][x] = Square(WHITE)
				elif (x % 2 == 0) and (y % 2 == 0): 
					matrix[y][x] = Square(BLACK)

		# initialize the pieces and put them in the appropriate squares

		for x in range(8):
			for y in range(3):
				if matrix[x][y].color == BLACK:
					matrix[x][y].occupant = Piece(RED)
			for y in range(5, 8):
				if matrix[x][y].color == BLACK:
					matrix[x][y].occupant = Piece(BLUE)

		return matrix

	def get_board(self):
		"""
		Takes a board and returns a matrix of the board space colors. Used for testing new_board()

		Empty is 0, Blue is 1, Blue King is 2, Red is 3, Red King is 4
		"""
		board = self.matrix
		board_string = np.empty((8,8), dtype=object)

		for x in range(8):
			for y in range(8):
				if board[x][y].occupant is not None:
					if board[x][y].occupant.color == BLUE:
						if board[x][y].occupant.king:
							board_string[x,y] = "BK"
						else:
							board_string[x, y] = "B-"
					else:
						if board[x][y].occupant.king:
							board_string[x,y] = "RK"
						else:
							board_string[x, y] = "R-"
				else:
					board_string[x, y] = "[]"

		return board_string
	
	def rel(self, dir, pos):
		"""
		Returns the coordinates one square in a different direction to (x,y).

		===DOCTESTS===

		>>> board = Board()

		>>> board.rel(NORTHWEST, (1,2))
		(0,1)

		>>> board.rel(SOUTHEAST, (3,4))
		(4,5)

		>>> board.rel(NORTHEAST, (3,6))
		(4,5)

		>>> board.rel(SOUTHWEST, (2,5))
		(1,6)
		"""
		x, y = pos[0], pos[1]
		if dir == NORTHWEST:
			return (x - 1, y - 1)
		elif dir == NORTHEAST:
			return (x + 1, y - 1)
		elif dir == SOUTHWEST:
			return (x - 1, y + 1)
		elif dir == SOUTHEAST:
			return (x + 1, y + 1)
		else:
			return 0

	def adjacent(self, pos):
		"""
		Returns a list of squares locations that are adjacent (on a diagonal) to (x,y).
		"""
		return [self.rel(NORTHWEST, pos), self.rel(NORTHEAST, pos),self.rel(SOUTHWEST, pos),self.rel(SOUTHEAST, pos)]

	def location(self, pos):
		"""
		Takes a set of coordinates as arguments and returns self.matrix[x][y]
		This can be faster than writing something like self.matrix[coords[0]][coords[1]]
		"""
		return self.matrix[pos[0]][pos[1]]

	def blind_legal_moves(self, pos):
		"""
		Returns a list of blind legal move locations from a set of coordinates (x,y) on the board. 
		If that location is empty, then blind_legal_moves() return an empty list.
		"""
		x, y = pos[0], pos[1]

		if self.matrix[x][y].occupant != None:
			
			if self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == BLUE:
				blind_legal_moves = [self.rel(NORTHWEST, pos), self.rel(NORTHEAST, pos)]
				
			elif self.matrix[x][y].occupant.king == False and self.matrix[x][y].occupant.color == RED:
				blind_legal_moves = [self.rel(SOUTHWEST, pos), self.rel(SOUTHEAST, pos)]

			else:
				blind_legal_moves = [self.rel(NORTHWEST, pos), self.rel(NORTHEAST, pos), self.rel(SOUTHWEST, pos), self.rel(SOUTHEAST, pos)]

		else:
			blind_legal_moves = []

		return blind_legal_moves

	def legal_moves(self, pos, hop = False):
		"""
		Returns a list of legal move locations from a given set of coordinates (x,y) on the board.
		If that location is empty, then legal_moves() returns an empty list.
		"""
		x, y = pos[0], pos[1]
		blind_legal_moves = self.blind_legal_moves((x,y)) 
		legal_moves = []

		if hop == False:
			for move in blind_legal_moves:
				if self.on_board(move):
					if self.location(move).occupant == None:
						legal_moves.append(move)

					elif self.location(move).occupant.color != self.location((x,y)).occupant.color and self.on_board((move[0] + (move[0] - x), move[1] + (move[1] - y))) and self.location((move[0] + (move[0] - x), move[1] + (move[1] - y))).occupant == None: # is this location filled by an enemy piece?
						legal_moves.append((move[0] + (move[0] - x), move[1] + (move[1] - y)))
		else: # hop == True
			for move in blind_legal_moves:
				if self.on_board(move) and self.location(move).occupant != None:
					if self.location(move).occupant.color != self.location((x,y)).occupant.color and self.on_board((move[0] + (move[0] - x), move[1] + (move[1] - y))) and self.location((move[0] + (move[0] - x), move[1] + (move[1] - y))).occupant == None: # is this location filled by an enemy piece?
						legal_moves.append((move[0] + (move[0] - x), move[1] + (move[1] - y)))

		return legal_moves

	def all_legal_moves(self, color, hop = False, selected_piece=None):
		piece_to_action = dict()
		if selected_piece is None:
			for i, matrix_row in enumerate(self.matrix):
				for j, matrix_column in enumerate(matrix_row):
					if self.matrix[i][j].occupant is not None:
						if self.matrix[i][j].occupant.color == color:
							pos = (i, j)
							piece_to_action[pos] = self.legal_moves(pos, hop)
		else: # Only activated if we are hopping
			piece_to_action[selected_piece] = self.legal_moves(selected_piece, hop)
		return piece_to_action

	def remove_piece(self, pos):
		x, y = int(pos[0]), int(pos[1])
		"""
		Removes a piece from the board at position (x,y). 
		"""
		self.matrix[x][y].occupant = None

	def move_piece(self, pos1, pos2):
		"""
		Move a piece from (start_x, start_y) to (end_x, end_y).
		"""
		start_x, start_y = pos1[0], pos1[1]
		end_x, end_y = pos2[0], pos2[1]
		self.matrix[end_x][end_y].occupant = self.matrix[start_x][start_y].occupant
		self.remove_piece((start_x, start_y))

		self.king((end_x, end_y))

	def is_end_square(self, coords):
		"""
		Is passed a coordinate tuple (x,y), and returns true or 
		false depending on if that square on the board is an end square.

		===DOCTESTS===

		>>> board = Board()

		>>> board.is_end_square((2,7))
		True

		>>> board.is_end_square((5,0))
		True

		>>>board.is_end_square((0,5))
		False
		"""

		if coords[1] == 0 or coords[1] == 7:
			return True
		else:
			return False

	def on_board(self, pos):
		"""
		Checks to see if the given square (x,y) lies on the board.
		If it does, then on_board() return True. Otherwise it returns false.

		===DOCTESTS===
		>>> board = Board()

		>>> board.on_board((5,0)):
		True

		>>> board.on_board(-2, 0):
		False

		>>> board.on_board(3, 9):
		False
		"""
		if pos[0] < 0 or pos[1] < 0 or pos[0] > 7 or pos[1] > 7:
			return False
		else:
			return True


	def king(self, pos):
		"""
		Takes in (x,y), the coordinates of square to be considered for kinging.
		If it meets the criteria, then king() kings the piece in that square and kings it.
		"""
		if self.location(pos).occupant != None:
			if (self.location(pos).occupant.color == BLUE and pos[1] == 0) or (self.location(pos).occupant.color == RED and pos[1] == 7):
				self.location(pos).occupant.king = True

class Piece:
	def __init__(self, color, king = False):
		self.color = color
		self.king = king

class Square:
	def __init__(self, color, occupant = None):
		self.color = color # color is either BLACK or WHITE
		self.occupant = occupant # occupant is a Square object

def main():
	start = time.time()
	num_games = 1
	all_moves = []
	agent1 = MuZero_Agent(agent_num=0, device="cuda:0", save_data=False)
	agent2 = MuZero_Agent(agent_num=0, device="cuda:0", save_data=False)
	for _ in range(num_games):
		game = Game("computer", "random_computer", show_graphics=True, print_board=False, delay=0, agent1=agent1, agent2=agent2)
		all_moves.append(game.play()[1])
	end = time.time()
	print(num_games, "games in", end-start, "seconds")
	print("Average of {} moves per game.".format(np.mean(all_moves)))
	# board = Board()
	# print(board.get_board())
	# your_pieces = np.squeeze(np.dstack(np.where(board.get_board() == 'R-')))
	# # choice = np.random.randint(your_pieces.shape[0])
	# your_choice = (1,5)
	# # options = board.legal_moves(tuple(your_choice))
	#
	# options = board.all_legal_moves(color=BLUE)
	# print(options)
	#
	# n = len(options)
	#
	# move = np.random.randint(2)
	# print(your_choice, options)
	#
	# board.move_piece(your_choice, options[your_choice][move])
	# print(board.get_board())


if __name__ == "__main__":
	import cProfile

	cProfile.run('main()')
	# main()