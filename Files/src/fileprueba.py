from metaworld.envs import ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE
from metaworld.policies.sawyer_button_press_v2_policy import SawyerButtonPressV2Policy
policy_oracle = SawyerButtonPressV2Policy()


plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["button-press-v2-goal-observable"]
env = plate_slide_goal_observable_cls()

observation = env.reset()


for t in range(1000):
    print("t: ", t)
    env.render()  # Make the environment visible
    action = policy_oracle.get_action(observation)
    observation, reward, environment_done, info = env.step(action)
