import numpy as np
from utils import softmax

class Qtable():
    def __init__(self, config) -> None:
        self._Qtable = np.random.uniform(low=-1, high=1, size=(config.state_size, config.num_action))
        self.gamma = config.gamma
        self.alpha = config.alpha
        self.action_arr = np.arange(config.num_action)
        if config.restore:
            restore_file_path = config.logdir + config.restore_file
            self._Qtable = np.load(restore_file_path)
    # 行動の選択
    def get_action(self, state, explore=True, global_step=None, method="softmax"):
        if method == "softmax":
            if explore:
                prob = softmax(self._Qtable[state])
                next_action = np.random.choice(self.action_arr, p=prob)
            else:
                max_action = np.where(self._Qtable[state] == np.max(self._Qtable[state]))[0]
                next_action = np.random.choice(max_action)
            return next_action
        elif method == "epsilon-greedy":
            if explore:
                pass
            else:
                pass
        elif method == "random":
            next_action = np.random.choice(self.action_arr)
            return next_action
    
    def update_Qtable(self, state, action, reward, next_state):
        next_maxQ = np.max(self._Qtable[next_state])
        self._Qtable[state, action] = (1 - self.alpha) * self._Qtable[state, action] \
                                    + self.alpha * (reward + self.gamma * next_maxQ)
        debug = False
        if debug:
            print(f"Qtable: {self._Qtable}")
        return self._Qtable
