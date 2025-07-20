# main.py
import sys
import pygame
import torch
from gomoku import GomokuEnv
from gomoku.actors import *
from gomoku.gomoku_net import GomokuNet

def has_quit_event(events):
    for event in events:
        if event.type == pygame.QUIT:
            return True
    return False


def get_human_action(events):
    for event in events:
        if event.type == pygame.MOUSEBUTTONDOWN:
            x, y = event.pos
            row, col = y // 30, x // 30
            return (row, col)
    return None


def get_machine_actor():
    if len(sys.argv) == 1:
        print("No machine actor selected. Defaulting to Random.")
        return RandomActor()
    elif sys.argv[1] == "random":
        print("Using Random Actor.")
        return RandomActor()
    elif sys.argv[1] == "ai":
        print("Using AI Actor.")
        return AIActor()

def run_game():
    env = GomokuEnv(board_size=15)
    env.reset()
    done = False

    machine_actor = get_machine_actor()

    while not done:
        env.render(mode="human")

        pygame.time.wait(100)
        events = pygame.event.get()
        if has_quit_event(events):
            done = True
            continue

        if env.game_over:
            continue

        action = get_human_action(events) if env.current_player == 1 else machine_actor(env)
        if action is None:
            continue

        env.step(action)

    pygame.quit()


if __name__ == "__main__":
    run_game()
