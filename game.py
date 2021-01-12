import pygame as pg, random
import numpy as np

class Settings:
    def __init__(self):
        self.w = 28
        self.h = 28
        self.rect_len = 15

class Snake:
    def __init__(self):
        
        self.image_up = pg.image.load('images/head_up.bmp')
        self.image_down = pg.image.load('images/head_down.bmp')
        self.image_left = pg.image.load('images/head_left.bmp')
        self.image_right = pg.image.load('images/head_right.bmp')

        self.tail_up = pg.image.load('images/tail_up.bmp')
        self.tail_down = pg.image.load('images/tail_down.bmp')
        self.tail_left = pg.image.load('images/tail_left.bmp')
        self.tail_right = pg.image.load('images/tail_right.bmp')
            
        self.image_body =pg.image.load('images/body.bmp')

        self.zig = "right"
        self.initialize()

    def initialize(self):
        self.position = [6, 6]
        self.segments = [[6 - i, 6] for i in range(3)]
        self.score = 0

    def blit_body(self, x, y, screen):
        screen.blit(self.image_body, (x, y))
        
    def blit_head(self, x, y, screen):
        if self.zig == "up":
            screen.blit(self.image_up, (x, y))
        elif self.zig == "down":
            screen.blit(self.image_down, (x, y))
        elif self.zig == "left":
            screen.blit(self.image_left, (x, y))
        else:
            screen.blit(self.image_right, (x, y))
            
    def blit_tail(self, x, y, screen):
        t_direction = [self.segments[-2][i] - self.segments[-1][i] for i in range(2)]
        
        if t_direction == [0, -1]:
            screen.blit(self.tail_up, (x, y))
        elif t_direction == [0, 1]:
            screen.blit(self.tail_down, (x, y))
        elif t_direction == [-1, 0]:
            screen.blit(self.tail_left, (x, y))
        else:
            screen.blit(self.tail_right, (x, y))
    
    def blit(self, rect_len, screen):
        self.blit_head(self.segments[0][0]*rect_len, self.segments[0][1]*rect_len, screen)
        for position in self.segments[1:-1]:
            self.blit_body(position[0]*rect_len, position[1]*rect_len, screen)
        self.blit_tail(self.segments[-1][0]*rect_len, self.segments[-1][1]*rect_len, screen)
            
    
    def update(self):
        if self.zig == 'right':
            self.position[0] += 1
        if self.zig == 'left':
            self.position[0] -= 1
        if self.zig  == 'up':
            self.position[1] -= 1
        if self.zig  == 'down':
            self.position[1] += 1
        self.segments.insert(0, list(self.position))
        
class Apple():
    def __init__(self, set):
        self.set = set
        
        self.style = str(random.randint(1, 8))
        self.image = pg.image.load('images/apple' + str(self.style) + '.bmp')
        self.initialize()
        
    def random_pos(self, S):
        self.style = str(random.randint(1, 8))
        self.image = pg.image.load('images/apple' + str(self.style) + '.bmp')
        
        self.position[0] = random.randint(0, self.set.w-1)
        self.position[1] = random.randint(0, self.set.h-1)

        self.position[0] = random.randint(9, 19)
        self.position[1] = random.randint(9, 19)
        
        if self.position in S.segments:
            self.random_pos(S)

    def blit(self, screen):
        screen.blit(self.image, [p * self.set.rect_len for p in self.position])
   
    def initialize(self):
        self.position = [15, 10]
      
        
class Game:

    def __init__(self):
        self.settings = Settings()
        self.Snake= Snake()
        self.apple = Apple(self.settings)
        self.move_dict = {0 : 'up', 1 : 'down', 2 : 'left', 3 : 'right'}
        
    def restart_game(self):
        self.Snake.initialize()
        self.apple.initialize()

    def current_state(self):         
        state = np.zeros((self.settings.w+2, self.settings.h+2, 2))
        expand = [[0, 1], [0, -1], [-1, 0], [1, 0], [0, 2], [0, -2], [-2, 0], [2, 0]]
        
        for position in self.Snake.segments:
            state[position[1], position[0], 0] = 1
        
        state[:, :, 1] = -0.5        

        state[self.apple.position[1], self.apple.position[0], 1] = 0.5
        for d in expand:
            state[self.apple.position[1]+d[0], self.apple.position[0]+d[1], 1] = 0.5
        return state
    
    def zig_to_int(self, direction):
        direction_dict = {value : key for key,value in self.move_dict.items()}
        return direction_dict[direction]
        
    def do_move(self, move):
        move_dict = self.move_dict
        
        change_direction = move_dict[move]
        
        if change_direction == 'right' and not self.Snake.zig == 'left':
            self.Snake.zig= change_direction
        if change_direction == 'left' and not self.Snake.zig == 'right':
            self.Snake.zig = change_direction
        if change_direction == 'up' and not self.Snake.zig == 'down':
            self.Snake.zig = change_direction
        if change_direction == 'down' and not self.Snake.zig == 'up':
            self.Snake.zig = change_direction

        self.Snake.update()
        
        if self.Snake.position == self.apple.position:
            self.apple.random_pos(self.Snake)
            reward = 1
            self.Snake.score += 1
        else:
            self.Snake.segments.pop()
            reward = 0
                
        if self.game_end():
            return -1
                    
        return reward
    
    def game_end(self):
        end = False
        if self.Snake.position[0] >= self.settings.w or self.Snake.position[0] < 0:
            end = True
        if self.Snake.position[1] >= self.settings.h or self.Snake.position[1] < 0:
            end = True
        if self.Snake.segments[0] in self.Snake.segments[1:]:
            end = True

        return end
    
    def blit_score(self, c, screen):
        f = pg.font.SysFont(None, 25)
        t = f.render('Score: ' + str(self.Snake.score), True, c)
        screen.blit(t, (0, 0))

