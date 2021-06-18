import pygame
import random
import math
from pygame.locals import (
    K_UP,
    K_DOWN,
    K_LEFT,
    K_RIGHT,
    K_r,
    K_ESCAPE,
    KEYDOWN,
    QUIT,
)
import numpy as np
from enum import IntEnum
import gym
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam
from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory
from gym import error, spaces
import copy
import argparse
import sys

WIDTH = 512
HEIGHT = 512
PLAT_WIDTH = 16
PLAT_HEIGHT = 80
PLAT_BORDER_OFFSET = 8
ENEMY_PLAT_COLOR = (255, 255, 255)
PLAYER_PLAT_COLOR = (255, 255, 255)
PLAT_SPEED = 3.3
BALL_SPEED = 6
BALL_COLOR = (255, 0, 0)
BALL_RADIUS = 8
BOUNDS = {
    'left': 0,
    'right': WIDTH,
    'top': 0,
    'bottom': HEIGHT,
}
MODEL_FILE = 'dqn_weights.h5f'
BALL_BOUNCE_Y_ANGLE = BALL_SPEED
PADDING = 5

class Difficulty(IntEnum):
    Easy = 0
    Hard = 1

state = {
    'enemy_pos': None,
    'player_pos': None,
    'player_vel': None,
    'ball_pos': None,
    'ball_vel': None,
    'enemy_score': 0,
    'player_score': 0,
    'difficulty': Difficulty.Hard,
}

def respawn_ball():
    state['ball_pos'] = [int(WIDTH / 2), int(HEIGHT / 2)]
    K = 0.05
    state['ball_vel'] = pygame.math.Vector2([
        random.choice([-1, 1]) * random.uniform(0.5, 1.0) * 2 * BALL_SPEED,
        random.uniform(0.0, 0.25) * 2 * BALL_BOUNCE_Y_ANGLE,
    ])

def reset():
    state['enemy_pos'] = [
        PLAT_BORDER_OFFSET,
        int(HEIGHT / 2)
    ]
    state['player_pos'] = [
        WIDTH - PLAT_BORDER_OFFSET - PLAT_WIDTH,
        int(HEIGHT / 2)
    ]
    state['player_vel'] = PLAT_SPEED
    respawn_ball()
    state['player_score'] = 0
    state['enemy_score'] = 0

def player_move_up():
    new_y = state['player_pos'][1] - PLAT_SPEED
    if new_y <= 0:
        state['player_pos'][1] = 0
    else:
        state['player_pos'][1] = new_y

def player_move_down():
    new_y = state['player_pos'][1] + PLAT_HEIGHT + PLAT_SPEED
    if new_y >= HEIGHT:
        state['player_pos'][1] = HEIGHT - PLAT_HEIGHT
    else:
        state['player_pos'][1] += PLAT_SPEED

def collides_with_platform(ball_pos, platform, mod):
    p_startx = platform[0]
    p_endx = p_startx + PLAT_WIDTH
    p_starty = platform[1]
    p_endy = p_starty + PLAT_HEIGHT
    bp = [ball_pos[0] + mod * BALL_RADIUS, ball_pos[1] + BALL_RADIUS]
    if mod == 1:
        collides_x = bp[0] >= p_startx
    else:
        collides_x = bp[0] <= (p_startx + PLAT_WIDTH)
    collides_y = bp[1] >= p_starty and bp[1] <= p_endy
    collides = collides_x and collides_y
    return collides

def update():
    # Update enemy pos
    if state['difficulty'] == Difficulty.Easy:
        state['enemy_pos'][1] += (state['ball_pos'][1] - state['enemy_pos'][1]) / random.uniform(2, 5)
    else:
        d = (state['ball_pos'][1] - state['enemy_pos'][1])
        k = 1 if d > 0 else -1 if d < 0 else 0
        state['enemy_pos'][1] += k * PLAT_SPEED
    if state['enemy_pos'][1] <= 0:
        state['enemy_pos'][1] = 0
    elif state['enemy_pos'][1] + PLAT_HEIGHT >= HEIGHT:
        state['enemy_pos'][1] = HEIGHT - PLAT_HEIGHT

    bpmrx = state['ball_pos'][0] - BALL_RADIUS
    bpprx = state['ball_pos'][0] + BALL_RADIUS
    bpmry = state['ball_pos'][1] - BALL_RADIUS
    bppry = state['ball_pos'][1] + BALL_RADIUS

    if bpmrx < BOUNDS['left']:
        # enemy loss
        state['player_score'] += 1
        respawn_ball()

    if (bpprx > BOUNDS['right']):
        # player loss
        state['enemy_score'] += 1
        respawn_ball()

    if (bpmry < BOUNDS['top']) or (bppry > BOUNDS['bottom']):
        state['ball_vel'].y *= -1

    if collides_with_platform(state['ball_pos'], state['player_pos'], 1):
        state['ball_vel'].x *= -1 * BALL_SPEED
        state['ball_vel'].y = random.uniform(-BALL_BOUNCE_Y_ANGLE, BALL_BOUNCE_Y_ANGLE)
    elif collides_with_platform(state['ball_pos'], state['enemy_pos'], -1):
        state['ball_vel'].x *= -1 * BALL_SPEED
        state['ball_vel'].y = random.uniform(-BALL_BOUNCE_Y_ANGLE, BALL_BOUNCE_Y_ANGLE)

    # print('ball velocity: '.format(state['ball_vel']))
    state['ball_pos'][0] += state['ball_vel'].x
    state['ball_pos'][1] += state['ball_vel'].y

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
enemy_surf = pygame.Surface((PLAT_WIDTH, PLAT_HEIGHT))
enemy_surf.fill(ENEMY_PLAT_COLOR)
player_surf = pygame.Surface((PLAT_WIDTH, PLAT_HEIGHT))
player_surf.fill(PLAYER_PLAT_COLOR)
font = pygame.font.Font(pygame.font.get_default_font(), 18)
def render(state):
    # Reset screen
    screen.fill((0, 0, 0))
    # Opponent platform
    screen.blit(enemy_surf, (state['enemy_pos'][0], state['enemy_pos'][1]))
    # Player platform
    screen.blit(player_surf, (state['player_pos'][0], state['player_pos'][1]))
    # Ball
    pygame.draw.circle(
        screen,
        BALL_COLOR,
        (state['ball_pos'][0], state['ball_pos'][1]),
        BALL_RADIUS
    )
    # Render text
    en_text_surface = font.render('Enemy: {}'.format(state['enemy_score']), True, (255, 255, 255))
    en_text_rect = en_text_surface.get_rect()
    en_text_rect.topleft = (PADDING, PADDING)
    screen.blit(en_text_surface, en_text_rect)
    player_text_surface = font.render('Player: {}'.format(state['player_score']), True, (255, 255, 255))
    player_text_rect = player_text_surface.get_rect()
    player_text_rect.topright = (WIDTH - PADDING, PADDING)
    screen.blit(player_text_surface, player_text_rect)
    pygame.display.flip()

def run_game():
    running = True
    reset()
    clock = pygame.time.Clock()
    while running:
        # Process events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_r:
                    reset()
        keys = pygame.key.get_pressed()
        if keys[K_UP]:
            player_move_up()
        elif keys[K_DOWN]:
            player_move_down()
        update()
        render(state)
        clock.tick(60)
    pygame.quit()

class Action(IntEnum):
    Up = 0
    Down = 1
    DoNothing = 2

class PongEnv(gym.Env):
    K = 5
    ACTION_TO_FUN_MAPPING = {
        Action.Up: player_move_up,
        Action.Down: player_move_down,
        Action.DoNothing: lambda: None,
    }

    def __init__(self):
        self.action_space = spaces.Discrete(len(list(PongEnv.ACTION_TO_FUN_MAPPING.keys())))
        # self.state = np.array(list(state.values()))

    @staticmethod
    def state_diff(new_state, old_state):
        return new_state['player_score'] - old_state['player_score']

    def get_state(self):
        global state
        return state

    @staticmethod
    def state_to_input_repr(state):
        s = np.array([
            state['enemy_pos'][0],
            state['enemy_pos'][1],
            state['player_pos'][0],
            state['player_pos'][1],
            state['ball_pos'][0],
            state['ball_pos'][1],
            state['ball_vel'][0],
            state['ball_vel'][1],
        ])
        return s

    def perform_action(self, action):
        global state
        PongEnv.ACTION_TO_FUN_MAPPING[action]()

    def reset(self):
        reset()
        state = self.get_state()
        return PongEnv.state_to_input_repr(state)

    def render(self, mode='human'):
        render(self.get_state())

    def step(self, action):
        # print("step(): {}".format(action))
        self.perform_action(action)
        # print("TAKING ACTION: ", action)
        update()
        state = self.get_state()
        reward = state['player_score'] - state['enemy_score']
        done = abs(state['player_score'] - state['enemy_score']) >= PongEnv.K
        return PongEnv.state_to_input_repr(state), reward, done, {}

    def clone_state(self):
        return copy.deepcopy(state)

    def restore_state(self, state_):
        global state
        state = copy.deepcopy(state_)

    def clone_full_state(self):
        return copy.deepcopy(state)

    def restore_full_state(self, state_):
        global state
        state = copy.deepcopy(state_)

# Model
def setup_model(env):
    model = Sequential()
    state_param_amount = 8
    actions_amount = env.action_space.n
    print(env.action_space.n)
    model.add(Flatten(input_shape=(1, state_param_amount)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions_amount, activation='linear'))
    return model

# Agent
def setup_agent(env, model):
    actions_amount = env.action_space.n
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy,
                    nb_actions=actions_amount, nb_steps_warmup=10, target_model_update=1e-2)
    dqn.compile(Adam(lr=1e-3), metrics=['mae'])
    return dqn

def train():
    env = PongEnv()
    model = setup_model(env)
    dqn = setup_agent(env, model)
    dqn.fit(env, nb_steps=50000, visualize=False, verbose=1)
    dqn.save_weights(MODEL_FILE, overwrite=True)

def run_model():
    env = PongEnv()
    model = setup_model(env)
    dqn = setup_agent(env, model)
    dqn.load_weights(MODEL_FILE)
    dqn.test(env, nb_episodes=50, visualize=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument('--play', action='store_true', default=False)
    args = parser.parse_args()
    if args.train:
        print("Training")
        train()
    elif args.play:
        run_game()
    else:
        print("Running saved model")
        run_model()
