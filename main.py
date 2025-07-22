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

def render_top_moves(env, probs, top_n=10):
    # 获取概率最高的前 top_n 个落子位置及其概率
    import numpy as np
    board_size = env.board_size
    flat_probs = probs.flatten()
    top_indices = np.argpartition(-flat_probs, top_n)[:top_n]
    top_indices = top_indices[np.argsort(-flat_probs[top_indices])]
    for idx in top_indices:
        row, col = divmod(idx, board_size)
        prob = flat_probs[idx]
        # 在棋盘上标注概率（例如用红色数字显示）
        x, y = col * 30 + 15, row * 30 + 15
        pygame.draw.circle(env.screen, (255,0,0), (x, y), 10, 2)
        font = pygame.font.SysFont(None, 20)
        text = font.render(f"{prob:.2f}", True, (255,0,0))
        env.screen.blit(text, (x-10, y-10))

def run_game():
    env = GomokuEnv(board_size=15)
    env.reset()
    done = False

    mode = "battle"
    if len(sys.argv) > 1 and sys.argv[1] == "demo":
        mode = "demo"
        print("演示模式：人类交替落子")
    else:
        print("对战模式：人类 vs 机器")

    machine_actor = get_machine_actor() if mode == "battle" else None
    ai_actor = AIActor() if mode == "demo" else None

    while not done:
        if mode == "demo" and ai_actor is not None:
            with torch.no_grad():
                probs = ai_actor.get_action_probs(env)
                env.top_move_probs = probs  # 只设置，不再单独渲染
        else:
            env.top_move_probs = None  # 非demo模式不显示概率

        env.render(mode="human")

        pygame.time.wait(100)
        events = pygame.event.get()
        if has_quit_event(events):
            done = True
            continue

        if env.game_over:
            continue

        if mode == "battle":
            action = get_human_action(events) if env.current_player == 1 else machine_actor(env)
        else:  # demo模式，人类交替落子
            action = get_human_action(events)

        if action is None:
            continue

        env.step(action)

    pygame.quit()


if __name__ == "__main__":
    run_game()
