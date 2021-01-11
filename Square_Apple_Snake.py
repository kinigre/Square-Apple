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
s = Game.S #snake
pg.init()
fpsClock = pg.time.Clock()
screen = pg.display.set_mode((Game.settings.w*15, Game.settings.h*15))
pg.display.set_caption('Square Apple')

crash_sound = pg.mixer.Sound('./sound/crash.wav')

def t_objects(t, f, c = black): #(text, font, color)
    t_surface = f.render(t, True, c)
    return t_surface, t_surface.get_rect()

def message_display(t, x, y, c = black):
    large_t = pg.font.SysFont('comicsansms', 50)
    t_surf, t_rect = t_objects(t, large_t, c)
    t_rect.center = (x, y)
    screen.blit(t_surf, t_rect)
    pg.display.update()
    
def button(msg, x, y, w, h, inactive_c, active_c, action = None, parameter = None):
    cursor = pg.mouse.get_pos()
    click = pg.mouse.get_pressed()
    if x + w > cursor[0] > x and y + h > cursor[1] > y:
        pg.draw.rect(screen, active_c, (x, y, w, h))
        if click[0] == 1 and action != None:
            if parameter != None:
                action(parameter)    
            else:
                action() 
    else:
        pg.draw.rect(screen, inactive_c, (x, y, w, h))

    small_t =pg.font.SysFont('comicsansms', 20)
    t_Surf, t_Rect = t_objects(msg, small_t)
    t_Rect.center = (x + (w / 2), y + (h / 2))
    screen.blit(t_Surf, t_Rect)

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
    
    tf.reset_default_graph()
    sess = tf.Session()

    dqn = Deep_Q_Network(sess, Game)
        
    g_s = Game.current_state()

    start_state = np.concatenate((g_s, g_s, g_s, g_s), axis=2)
    s_t = start_state
    
    while not Game.g_end():

        _, action_index = dqn.choose_action(s_t)
        
        move = action_index
        Game.do_move(move)
        
        pg.event.pump()
        
        g_s = Game.current_state()
        s_t = np.append(g_s, s_t[:, :, :-2], axis=2)
        
        screen.fill(white)
        
        s.blit(rect_len, screen)
        Game.apple.blit(screen)
        Game.blit_score(black, screen)
        
        pg.display.flip()
        
        fpsClock.tick(15)  

    crash()        
                  
if __name__ == "__main__":
    initial_interface()
