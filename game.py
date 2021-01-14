import pygame as pg, random
import numpy as np

class Settings:
    def __init__(self):
        self.w = 28 #폭
        self.h = 28 #높이
        self.rect_len = 15 #사각형 길이


class Snake:
    def __init__(self):
        #뱀 머리
        self.head_up = pg.image.load('images/head_up.bmp')
        self.head_down = pg.image.load('images/head_down.bmp')
        self.head_left = pg.image.load('images/head_left.bmp')
        self.head_right = pg.image.load('images/head_right.bmp')

        #뱀 꼬리
        self.tail_up = pg.image.load('images/tail_up.bmp')
        self.tail_down = pg.image.load('images/tail_down.bmp')
        self.tail_left = pg.image.load('images/tail_left.bmp')
        self.tail_right = pg.image.load('images/tail_right.bmp')

        #뱀 몸
        self.body = pg.image.load('images/body.bmp')

        #방향
        self.zig = "right"

        self.Reset()

    #초기화
    def Reset(self):
        self.position = [6, 6] #현재 위치 좌표
        self.segments = [[6 - i, 6] for i in range(3)] #분할
        self.score = 0 #점수

    #화면에 머리 블록 전송
    def Blit_Head(self, x, y, screen):
        if self.zig == "up":
            screen.blit(self.head_up, (x, y))
        elif self.zig == "down":
            screen.blit(self.head_down, (x, y))
        elif self.zig == "left":
            screen.blit(self.head_left, (x, y))
        else:
            screen.blit(self.head_right, (x, y))

    # 화면에 몸통 블록 전송
    def Blit_Body(self, x, y, screen):
        screen.blit(self.body, (x, y))

    # 화면에 꼬리 블록 전송
    def Blit_Tail(self, x, y, screen):
        tail_direction = [self.segments[-2][i] - self.segments[-1][i] for i in range(2)]

        if tail_direction == [0, -1]:
            screen.blit(self.tail_up, (x, y))
        elif tail_direction == [0, 1]:
            screen.blit(self.tail_down, (x, y))
        elif tail_direction == [-1, 0]:
            screen.blit(self.tail_left, (x, y))
        else:
            screen.blit(self.tail_right, (x, y))

    # 화면에 머리,몸통,꼬리 블록 연결시킨다.
    def Blit(self, rect_len, screen):
        self.Blit_Head(self.segments[0][0] * rect_len, self.segments[0][1] * rect_len, screen)
        for position in self.segments[1:-1]:
            self.Blit_Body(position[0] * rect_len, position[1] * rect_len, screen)
        self.Blit_Tail(self.segments[-1][0] * rect_len, self.segments[-1][1] * rect_len, screen)

    #방향 업데이트
    def Position_Update(self):
        if self.zig == 'right':
            self.position[0] += 1
        if self.zig == 'left':
            self.position[0] -= 1
        if self.zig == 'up':
            self.position[1] -= 1
        if self.zig == 'down':
            self.position[1] += 1
        self.segments.insert(0, list(self.position))

class Apple():
    def __init__(self, settings):
        self.settings = settings
        self.apple_images = pg.image.load('images/apple.bmp')
        self.Reset()

    #현재 사과 위치 초기화
    def Reset(self):
        self.position = [15, 10]

    #사과 위치 랜덤
    def Random_Position(self, snake):
        self.apple_images = pg.image.load('images/apple.bmp')

        self.position[0] = random.randint(0, self.settings.w - 1)
        self.position[1] = random.randint(0, self.settings.h - 1)

        self.position[0] = random.randint(9, 19)
        self.position[1] = random.randint(9, 19)

        if self.position in snake.segments:
            self.Random_Position(snake)

    #화면에 사과를 보여준다.
    def Blit(self, screen):
        screen.blit(self.apple_images, [p * self.settings.rect_len for p in self.position])


class Game:
    def __init__(self):
        self.settings = Settings()
        self.snake = Snake()
        #
        self.apple = Apple(self.settings)
        #방향 딕셔너리
        self.move_dict = {0: 'up', 1: 'down', 2: 'left',3: 'right'}

    #현재 상태
    def Current_State(self):
        S = np.zeros((self.settings.w + 2, self.settings.h + 2, 2)) #상태
        set_S = [[0, 1], [0, -1], [-1, 0], [1, 0], [0, 2], [0, -2], [-2, 0], [2, 0]] #상태의 집합
        for position in self.snake.segments:
            S[position[1], position[0], 0] = 1

        S[:, :, 1] = -0.5

        S[self.apple.position[1], self.apple.position[0], 1] = 0.5
        for d in set_S:
            S[self.apple.position[1] + d[0], self.apple.position[0] + d[1], 1] = 0.5
        return S

    def Direction(self, direction):
        direction_dict = {value: key for key, value in self.move_dict.items()}
        return direction_dict[direction]

    def Move(self, move):
        move_dict = self.move_dict

        change_direction = move_dict[move]

        if change_direction == 'right' and not self.snake.zig == 'left':
            self.snake.zig = change_direction
        if change_direction == 'left' and not self.snake.zig== 'right':
            self.snake.zig = change_direction
        if change_direction == 'up' and not self.snake.zig == 'down':
            self.snake.zig = change_direction
        if change_direction == 'down' and not self.snake.zig == 'up':
            self.snake.zig = change_direction

        self.snake.Position_Update()

        if self.snake.position == self.apple.position:
            self.apple.Random_Position(self.snake)
            Reward = 1
            self.snake.score += 1
        else:
            self.snake.segments.pop()
            Reward = 0

        if self.Game_End():
            return -1

        return Reward

    def Game_End(self):
        end = False
        if self.snake.position[0] >= self.settings.w or self.snake.position[0] < 0:
            end = True
        if self.snake.position[1] >= self.settings.h or self.snake.position[1] < 0:
            end = True
        if self.snake.segments[0] in self.snake.segments[1:]:
            end = True

        return end

    #점수를 화면에 보여준다.
    def Blit_Score(self, color, screen):
        font = pg.font.SysFont(None, 25)
        text = font.render('Score: ' + str(self.snake.score), True, color)
        screen.blit(text, (0, 0))

    # 게임 다시 시작
    def Restart_Game(self):
        self.snake.Reset()
        self.apple.Reset()