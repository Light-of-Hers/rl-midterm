from pysc2.agents import base_agent
from ..utils import *
import os

INIT_NUM = 1


class FeatureUnitAgent(base_agent.BaseAgent):
    def __init__(self):
        super().__init__()
        self.last_reward = 0
        self.max_reward = 0
        self.reward_list = []

    def setup(self, obs_spec, action_spec):
        super().setup(obs_spec, action_spec)
        if "feature_units" not in obs_spec:
            raise Exception("This agent needs feature_units observation.")

    def reset(self):
        if self.episodes:
            self.max_reward = max(
                self.max_reward, self.reward - self.last_reward)
            self.last_reward = self.reward
            print("average reward: {}, max reward: {}".format(
                self.reward / self.episodes, self.max_reward))
            self.reward_list.append(
                (int(self.reward / self.episodes), int(self.max_reward)))
        super().reset()


class RLAgent(FeatureUnitAgent):
    def __init__(self):
        super().__init__()
        self.file_name = None
        self.actions = []
        self.curr_state_func = []
        self.num_state = 0
        self.num_action = 0
        self.curr_state = -1
        self.prev_state = -1
        self.action = 0
        self.arg_bin = []
        self.q_table = np.random.uniform(0, 1, size=(INIT_NUM, INIT_NUM))
        self.init_act_flag = [0, 0]

    def decide_action(self):
        epsilon = 0.5 * (1 / (self.steps + 1))
        if epsilon <= np.random.uniform(0, 1):
            action = np.argmax(self.q_table[self.curr_state][:])
        else:
            action = np.random.randint(0, self.num_action)
        return action

    def update_q_table(self, obs):
        max_q_next = max(self.q_table[self.curr_state][:])
        self.q_table[self.prev_state][self.action] = (
                0.5 * self.q_table[self.prev_state][self.action] + 0.5 * (
                obs.reward + 0.99 * max_q_next))
        return

    def reset(self):
        super().reset()
        self.steps = 0
        self.init_act_flag = [0, 0]
        if self.episodes == 40:
            np.set_printoptions(suppress=True)
            np.savetxt(self.file_name, self.q_table, fmt="%.2f")

    def digitize_state(self, obs):
        digitized = 0
        for x in range(self.num_state):
            digitized = digitized * self.arg_bin[x][2] + np.digitize(
                self.curr_state_func[x](obs), bins=np.linspace(
                    self.arg_bin[x][0], self.arg_bin[x][1],
                    self.arg_bin[x][2] + 1)[1: -1])
        return digitized

    def step(self, obs):
        super().step(obs)
        if not self.init_act_flag[0]:
            self.init_act_flag[0] = 1
            return FUNCTIONS.no_op()
        if not self.init_act_flag[1]:
            self.init_act_flag[1] = 1
            return FUNCTIONS.select_army("select")
        self.curr_state = self.digitize_state(obs)
        if self.prev_state != -1:
            self.update_q_table(obs)
        self.action = self.decide_action()
        self.prev_state = self.curr_state
        return self.actions[self.action](obs)
