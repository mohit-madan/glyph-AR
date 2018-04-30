import pygame
from pygame.locals import *
from objloader import *


class App:
    def __init__(self):
        self._running = True
        self._display_surf = None
        self.size = self.weight, self.height = 640, 480
        self._image_surf = None
        self._cube_obj = None

    def on_init(self):
        pygame.init()
        self._display_surf = pygame.display.set_mode(self.size, pygame.HWSURFACE | pygame.DOUBLEBUF)
        self._running = True
        # load image as surface
        self._cube_obj = OBJ('data/3d_tree/3d_tree.obj', swapyz=True)
        self._image_surf = pygame.image.load("background.jpg").convert()

    def on_event(self, event):
        if event.type == pygame.QUIT:
            self._running = False

    def on_loop(self):
        pass

    def on_render(self):
        # args: image, position, (start of crop, size of crop)
        self._display_surf.blit(self._image_surf, (0, 0))
        self._image_surf = pygame.image.load("data/tree_texture.jpg").convert()
        glCallList(self._cube_obj.gl_list)
        pygame.display.flip()

    def on_cleanup(self):
        pygame.quit()

    def on_execute(self):
        if self.on_init() == False: # todo: what does on_init return? Why does "not self.on_init()" not work?
            self._running = False

        while self._running:
            for event in pygame.event.get():
                self.on_event(event)
            self.on_loop()
            self.on_render()
        self.on_cleanup()

if __name__ == "__main__" :
    theApp = App()
    theApp.on_execute()
