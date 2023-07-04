from dm_control import suite
import numpy as np

class Env():
    def __init__(self, config):
        self._env = suite.load(domain_name=config.domain, task_name=config.task)
        self._config = config
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
        n_best = (d+1)/2
        n_rad = np.digitize(rad, np.linspace(-np.pi, np.pi, d+1)[1:-1])
        n_vel = np.digitize(vel.clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])
        # return (np.abs(n_rad - n_best) + 1)**(-1)
        return -(((n_rad - n_best)/n_best)**2 + ((n_vel - n_best)/n_best)**2)

        

    def _digitized_state(self, obs):
        d = self._config.num_digitized
        vec, vel = obs["orientation"], obs["velocity"]
        rad = np.arctan2(vec[1], vec[0])
        # done = bool(rad < -np.pi/3 or rad > np.pi/3)
        done = False
        n_rad = np.digitize(rad, np.linspace(-np.pi, np.pi, d+1)[1:-1])
        n_vl = np.digitize(vel.clip(-8, 8), np.linspace(-8, 8, d+1)[1:-1])
        return n_rad + n_vl*d, done