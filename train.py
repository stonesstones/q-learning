import numpy as np
from pathlib import Path
from dataclasses import dataclass
import os

from env import Env
from models import Qtable
from logger import Logger
from collections import OrderedDict
import time

@dataclass
class EnvConfig:
    domain: str = "pendulum"
    task: str = "swingup"
    num_digitized: int = 16
    num_action: int = 4
    state_size: int = num_digitized**2
    gamma: float = 0.99
    alpha: float = 0.5
    max_episode: int = int(10e7)
    episode_length: int = 200
    should_log_model: int = (10e4)
    should_log_scalar: int = int(10e2)
    should_log_video: int = int(5*10e2)
    restore: bool = False
    restore_file: str = "Qtable.npy"
    video_length: int = 200
    logdir: str = "./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "/"

class Agent():
    def __init__(self, config: EnvConfig) -> None:
        self._config = config
        self._build_model()

    def get_action(self, state, explore=True):
        return self._qtable.get_action(state, explore)

    def update_Qtable(self, state, action, reward, next_state):
        return self._qtable.update_Qtable(state, action, reward, next_state)

    def _build_model(self):
        self._qtable = Qtable(self._config)
    
def main():
    config = EnvConfig()
    env = Env(config)
    os.makedirs(config.logdir, exist_ok=True)
    logger = Logger(config.logdir)
    agent = Agent(config)

    # main training loop
    for episode in range(config.max_episode):
        state = env.reset()
        episode_reward = 0
        for step in range(config.episode_length):
            action = agent.get_action(state)
            next_state, reward, done, _ = env.step(action)
            qtable = agent.update_Qtable(state, action, reward, next_state)
            episode_reward += reward
            state = next_state
            if done:
                break
        
        if episode % config.should_log_scalar == 0:
            print(f"\nepisode: {episode}, episode_reward: {episode_reward}")
            logger.add_scalars(OrderedDict([
                ("episode_reward", episode_reward),
            ]))

        if episode % config.should_log_model == 0:
            save_file = config.logdir + f"qtable_{episode}.npy"
            np.save(save_file, qtable)
        
        if episode % config.should_log_video == 0:
            env.reset()
            eval_reward = 0
            img_arr = []
            for _ in range(config.video_length):
                img_arr.append(env._env.physics.render(height=480, width=640,camera_id=0))
                next_state, reward, done, _ = env.step(agent.get_action(state, explore=False))
                state = next_state
                eval_reward += reward
                if done:
                    break
            print(f"evaluate episode reward: {eval_reward}")
            logger.log_video(img_arr)
            logger.add_scalars(OrderedDict([("eval_reward", eval_reward)]))
        logger.step()

if __name__ == "__main__":
    main()

