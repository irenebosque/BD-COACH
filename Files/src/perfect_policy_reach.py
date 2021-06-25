import numpy as np

from metaworld.policies.action import Action
from metaworld.policies.policy import Policy, assert_fully_parsed, move

class PolicyReach:
    def __init__(self):
        self.data = []
    def parse_obs(self, obs):
        return {
            'hand_pos': obs[:3],
            'unused_1': obs[3],
            'puck_pos': obs[4:7],
            'unused_2': obs[7:-3],
            'goal_pos': obs[-3:],
        }


    def action_perfect_policy(self, obs):

        o_d = self.parse_obs(obs[0][0])

        action = Action({
            'delta_pos': np.arange(3),
            'grab_effort': 3
        })

        action['delta_pos'] = move(o_d['hand_pos'], to_xyz=o_d['goal_pos'], p=5.)
        action['grab_effort'] = 0.


        action = np.clip(action.array, -1, 1)

        return action
