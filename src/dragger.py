import pygame

from src.const import *

class Dragger:

    def __init__(self):
        self.mouseX = 0
        self.mouseY = 0
        self.initial_row = 0
        self.initial_col = 0
        self.piece = None
        self.dragging = False

    def update_blit(self, surface):
        texture = self.piece.set_texture()
        img = pygame.image.load(texture)
        img = pygame.transform.scale(img, (90, 90))
        self.piece.texture_rect = img.get_rect(center=(self.mouseX, self.mouseY))
        surface.blit(img, self.piece.texture_rect)


    def update_mouse(self, pos):
        self.mouseX, self.mouseY = pos

    def save_initial(self, pos):
        self.initial_row = pos[1] // SQSIZEY
        self.initial_col = pos[0] // SQSIZEX

    def drag_piece(self, piece):
        self.piece = piece
        self.dragging = True

    def undrag_piece(self):
        self.piece = None
        self.dragging = False