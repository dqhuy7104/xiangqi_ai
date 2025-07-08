import pygame

class Sound:
    def __init__(self):
        self.sound_move = pygame.mixer.Sound('./assets/sound/move.mp3')
        self.sound_capture = pygame.mixer.Sound('./assets/sound/capture.mp3')

    def play_move(self):
        pygame.mixer.Sound.play(self.sound_move)

    def play_capture(self):
        pygame.mixer.Sound.play(self.sound_capture)