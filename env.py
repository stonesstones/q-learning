from dm_control import suite
import numpy as np
from sim.acrobot import swingup
from collections import OrderedDict

def make_env(config):
    if config.domain == 'double_pendulum':
        return Env_DOUBLE(config)
    elif config.domain == 'single_pendulum':
        return Env_SINGLE(config)
    else:
        raise NotImplementedError

class Env_DOUBLE():
    def __init__(self, config):
        self._config = config
        self._arrange = np.zeros(config.num_digitized-1)
        self._env = swingup()
        self._arrange[config.num_digitized//2] = -0.20
        self._arrange[config.num_digitized//2 - 2] = 0.20
        self._digitized_action = np.linspace(self._env.action_spec().minimum[0], self._env.action_spec().maximum[0], config.num_action)
    
    @property
    def action_space(self):
        return self._config.num_action
    @property
    def obs_space(self):
        return self._config.state_size
    
    def step(self, action):
        action = self._digitized_action[action]
        obs = self._env.step(action).observation
        obs = self._env.step(action).observation
        digitized_state, done, state_dict = self._digitized_state(obs)
        reward = self._get_reward(state_dict, done)
        return digitized_state, reward, done, None

    def reset(self):
        obs = self._env.reset().observation
        digitized_state, _, state_dict = self._digitized_state(obs)
        return digitized_state

    def _get_reward(self, state_dict, done):
        if done:
            return -10
        
        d = self._config.num_digitized
        n_elbow_rad, n_elbow_vel = state_dict["n_elbow_rad"], state_dict["n_elbow_vel"]
        n_best = (d-1)/2
        n_shoulder_rad, n_shoulder_vel = state_dict["n_shoulder_rad"], state_dict["n_shoulder_vel"]
        n_shoulder_best = (d//2 - 1) / 2
        # return (np.abs(n_rad - n_best) + 1)**(-1)
        bonus = 0
        if np.abs(n_elbow_rad - n_best) < 1 and np.abs(n_elbow_vel - n_best) < 1 and np.abs(n_shoulder_rad - n_shoulder_best) < 1:
            bonus = 3
        
        return -(((n_elbow_rad - n_best)/n_best)**2 + 0.3*((n_elbow_vel - n_best)/n_best)**2 + ((n_shoulder_rad - n_shoulder_best)/n_shoulder_best)**2) + bonus
        # if state_dict["elbow_rad"] < -np.pi/2 or state_dict["elbow_rad"] > np.pi/2:
        #     return -((1.1*(n_elbow_rad - n_best)/n_best)**2 + 0.3*((n_elbow_vel - n_best)/n_best)**2 + 0.9*((n_shoulder_rad - n_shoulder_best)/n_shoulder_best)**2) + bonus
        # else:
        #     return -((1.4*(n_elbow_rad - n_best)/n_best)**2 + 0.3*((n_elbow_vel - n_best)/n_best)**2 + 0.6*((n_shoulder_rad - n_shoulder_best)/n_shoulder_best)**2) + bonus

        

    def _digitized_state(self, obs):
        # 0 is positive z axis at doule pendulum elbow angle 
        state_dict = OrderedDict()
        d = self._config.num_digitized
        vec, vel = obs["orientations"], obs["velocity"]
        shoulder_vec, elbow_vec = vec[0:2], vec[2:4]
        shoulder_vel, elbow_vel = vel[0], vel[1]
        shoulder_rad = np.arctan2(shoulder_vec[1], shoulder_vec[0])
        elbow_rad = np.arctan2(elbow_vec[1], elbow_vec[0])
        n_shoulder_rad = np.digitize(shoulder_rad, np.linspace(-np.pi, np.pi, d//2 +1)[1:-1])
        n_shoulder_vel = np.digitize(shoulder_vel.clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])

        n_elbow_rad = np.digitize(elbow_rad, np.linspace(-np.pi, np.pi, d+1)[1:-1] + self._arrange)
        n_elbow_vel = np.digitize(elbow_vel.clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])

        state_dict["shoulder_rad"] = shoulder_rad
        state_dict["elbow_rad"] = elbow_rad
        state_dict["shoulder_vel"] = shoulder_vel
        state_dict["elbow_vel"] = elbow_vel
        state_dict["n_shoulder_rad"] = n_shoulder_rad
        state_dict["n_elbow_rad"] = n_elbow_rad
        state_dict["n_shoulder_vel"] = n_shoulder_vel
        state_dict["n_elbow_vel"] = n_elbow_vel
        state_dict["digitized_state"] = n_elbow_rad + n_elbow_vel*d + n_shoulder_vel*d**2 + n_shoulder_rad*d**3
        if n_shoulder_rad == 0 or n_shoulder_rad == d//2 - 1:
            done = True
        else:
            done = False
        return state_dict["digitized_state"], done, state_dict

class Env_SINGLE():
    def __init__(self, config):
        self._config = config
        self._arrange = np.zeros(config.num_digitized-1)
        self._env = suite.load(domain_name=config.domain, task_name=config.task)
        self._arrange[config.num_digitized//2] = -0.20
        self._arrange[config.num_digitized//2 - 2] = 0.20
        self._digitized_action = np.linspace(self._env.action_spec().minimum[0], self._env.action_spec().maximum[0], config.num_action)
    
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
        domain: str = "double_pendulum"
        task: str = "swingup"
        num_digitized: int = 16
        num_action: int = 2
        state_size: int = num_digitized**3
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
    env = Env_DOUBLE(EnvConfig())
    # env = suite.load(domain_name="acrobot", task_name="swingup")
    env.reset()
    for i in range(100):
        img = Image.fromarray(env._env.physics.render(height=480, width=640,camera_id=0))
        img.save("./img.png")
        for i in range(10):
            env.step(2)

if __name__ == "__main__":
    main()