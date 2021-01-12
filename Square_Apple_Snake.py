import pygame as pg, time
from game import Game

black = (0, 0, 0)
white = (255, 255, 255)

green = (0, 200, 0)
b_green = (0, 255, 0)
red = (200, 0, 0)
b_red = (255, 0, 0)
blue = (32, 178, 170)
b_blue = (32, 200, 200)
yellow = (255, 205, 0)
b_yellow = (255, 255, 0)

Game = Game()
rect_len = Game.settings.rect_len
snake = Game.Snake #snake
pg.init()
fpsClock = pg.time.Clock()
screen = pg.display.set_mode((Game.settings.w*15, Game.settings.h*15))
pg.display.set_caption('Square Apple')

crash_sound = pg.mixer.Sound('./sound/crash.wav')

def text_objects(text, font, color = black):
    text_surface = font.render(text, True, color)
    return text_surface, text_surface.get_rect()

def message_display(text, x, y, color = black):
    large_text = pg.font.SysFont('comicsansms', 50)
    text_surf, text_rect = text_objects(text, large_text, color)
    text_rect.center = (round(x), round(y))
    screen.blit(text_surf, text_rect)
    pg.display.update()
    
def button(msg, x, y, w, h, inactive_color, active_color, action = None, parameter = None):
    cursor = pg.mouse.get_pos()
    click = pg.mouse.get_pressed()
    if x + w > cursor[0] > x and y + h > cursor[1] > y:
        pg.draw.rect(screen, active_color, (x, y, w, h))
        if click[0] == 1 and action != None:
            if parameter != None:
                action(parameter)    
            else:
                action() 
    else:
        pg.draw.rect(screen, inactive_color, (x, y, w, h))

    small_text =pg.font.SysFont('comicsansms', 20)
    text_Surf, text_Rect = text_objects(msg, small_text)
    text_Rect.center = (round(x) + round(w / 2), round(y) + round(h / 2))
    screen.blit(text_Surf, text_Rect)

def quit_game():
    pg.quit()
    quit()
    
def crash():    
    pg.mixer.Sound.play(crash_sound)
    message_display('crash', Game.settings.w/2*15, Game.settings.h/3*15, black)
    time.sleep(1)
   
def initial_interface():
    intro = True
    while intro:
                
        for event in pg.event.get():
            if event.type == pg.QUIT:
                pg.quit()
                
        screen.fill(white)
        message_display('Square Apple', Game.settings.w/2*15, Game.settings.h/4*15)

        button('DQN', 175, 210, 80, 40, yellow, b_yellow, DQN)
        button('Quit', 175, 270, 80, 40, red, b_red, quit_game)
        
        pg.display.update()
        pg.time.Clock().tick(15)
        

def DQN():
    import tensorflow as tf
    from DQN import Deep_Q_Network
    import numpy as np
    
    Game.restart_game()
    
    tf.compat.v1.reset_default_graph()
    sess = tf.compat.v1.Session()

    dqn = Deep_Q_Network(sess, Game)
        
    g_s = Game.current_state()

    start_state = np.concatenate((g_s, g_s, g_s, g_s), axis=2)
    s_t = start_state
    
    while not Game.game_end():

        _, action_index = dqn.choose_action(s_t)
        
        move = action_index
        Game.do_move(move)
        
        pg.event.pump()
        
        g_s = Game.current_state()
        s_t = np.append(g_s, s_t[:, :, :-2], axis=2)
        
        screen.fill(white)
        
        snake.blit(rect_len, screen)
        Game.apple.blit(screen)
        Game.blit_score(black, screen)
        
        pg.display.flip()
        
        fpsClock.tick(15)  

    crash()        
                  
if __name__ == "__main__":
    initial_interface()
