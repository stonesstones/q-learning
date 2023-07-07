from dm_control import suite
import numpy as np
from sim.acrobot import Balance, swingup
class Env():
    def __init__(self, config):
        if config.domain == 'acrobot':
            self._env = swingup()
        else:
            self._env = suite.load(domain_name=config.domain, task_name=config.task)
        self._config = config
        self._digitized_action = np.linspace(self._env.action_spec().minimum[0], self._env.action_spec().maximum[0], config.num_action)
        self._arrange = np.zeros(config.num_digitized-1)
        self._arrange[config.num_digitized//2] = -0.20
        self._arrange[config.num_digitized//2 - 2] = 0.20
    
    @property
    def action_space(self):
        return self._config.num_action
    @property
    def obs_space(self):
        return self._config.state_size
    
    def step(self, action):
        action = self._digitized_action[action]
        obs = self._env.step(action).observation
        digitized_state, done = self._digitized_state(obs)
        reward = self._get_reward(obs, done)
        return digitized_state, reward, done, None

    def reset(self):
        obs = self._env.reset().observation
        digitized_state, _ = self._digitized_state(obs)
        return digitized_state

    def _get_reward(self, obs, done):
        if done:
            return -10
        d = self._config.num_digitized
        vec, vel = obs["orientation"], obs["velocity"]
        rad = np.arctan2(vec[1], vec[0])
        n_best = (d+1)/2 - 1
        n_rad = np.digitize(rad, np.linspace(-np.pi, np.pi, d+1)[1:-1] + self._arrange)
        n_vel = np.digitize(vel.clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])
        # return (np.abs(n_rad - n_best) + 1)**(-1)
        bonus = 0
        if np.abs(n_rad - n_best) < 1 and np.abs(n_vel - n_best) < 1:
            bonus = 5
        return -(((n_rad - n_best)/n_best)**2 + ((n_vel - n_best)/n_best)**2) + bonus

        

    def _digitized_state(self, obs):
        d = self._config.num_digitized
        vec, vel = obs["orientation"], obs["velocity"]
        rad = np.arctan2(vec[1], vec[0])
        # done = bool(rad < -np.pi/3 or rad > np.pi/3)
        done = False
        n_rad = np.digitize(rad, np.linspace(-np.pi, np.pi, d+1)[1:-1] + self._arrange)
        n_vl = np.digitize(vel.clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])
        return n_rad + n_vl*d, done

def main():
    from dataclasses import dataclass
    import time 
    from PIL import Image

    @dataclass
    class EnvConfig:
        domain: str = "acrobot"
        task: str = "swingup"
        num_digitized: int = 16
        num_action: int = 2
        state_size: int = num_digitized**2
        gamma: float = 0.99
        alpha: float = 0.5
        max_episode: int = int(10e3)
        episode_length: int = 400
        should_log_model: int = (10e3)
        should_log_scalar: int = int(10)
        should_log_video: int = int(50)
        restore: bool = False
        restore_file: str = "Qtable.npy"
        video_length: int = 400
        logdir: str = "./logs/" + str(time.strftime("%Y-%m-%d-%H-%M-%S")) + "/"
    env = Env(EnvConfig())._env
    # env = suite.load(domain_name="acrobot", task_name="swingup")
    env.reset()
    for i in range(100):
        img = Image.fromarray(env.physics.render(height=480, width=640,camera_id=0))
        img.save("./img.png")
        for i in range(10):
            env.step(3)

if __name__ == "__main__":
    main()