# DCOACH with Human Model
This repository contains the code for my master thesis “Towards Off-Policy Corrective Imitation Learning”



### Random initialization of the task

To randomly initialize a task, it is necessary to modify the file `sawyer_xyz_env.py` from the metaworld package.

```python
self.random_init = True
```

Then and as it was stated in the Meta-World garage slack:


 > _Avnish Narayan: here is the long answer:random_init is a parameter that used to control the task initial configuration of an environment. **The task initial configuration of an environment is the initial object and goal positions of that environment.The task initial configuration of an environment is only changed after the user makes a call to set_task with any of the tasks** that have been sampled from MTXX/MLXX.train_tasks or ML.test_tasksrandom initialization (random_init=True) is only needed in the ML benchmarks, and so we’ve enabled this by default when you construct ML1 ML10 or ML45 (edited)_ 
 
Taking into account the previous comment, it is necessary to create again the environment after the beginning of each episode. Otherwise, even if you change the variable `random_init` to `True` it won't have any effect. Here is a simple example tht prints the initial position of the end-effector (the "hand"), the goal and the object at the beginning of each episode. Because `random_init` is set to `True`, different initial positions for the goal and end object will be printed.
📢 **The initial position of the end-effector will remain the same.**


```python
from metaworld.envs import (ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE)

for i_episode in range(20):
    plate_slide_goal_observable_cls = ALL_V2_ENVIRONMENTS_GOAL_OBSERVABLE["push-v2-goal-observable"]
    env = plate_slide_goal_observable_cls()
    observation = env.reset() # AL final de cada episodio se resetea
    print('env.reset')
    for t in range(100):
        if t == 0:
            print('Episode: ', i_episode, 'end-effector position: ', observation[:3],  'target position: ', observation[-3:], 'object (ball) position: ', observation[4:7])

        #env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
```

📢 Not all the environments have initial random positions for the object. For example in the case of the `plate-slide-v2`, both limits `obj_low` and
`obj_high` are the same and therefore, the disk, will always appear at the same initial position even if `random_init` is set to `True`.


```python
class SawyerPlateSlideEnvV2(SawyerXYZEnv):
    OBJ_RADIUS = 0.04

    def __init__(self):

        goal_low = (-0.1, 0.85, 0.)
        goal_high = (0.1, 0.9, 0.)
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (0., 0.6, 0.)
        obj_high = (0., 0.6, 0.)
```
On the other hand, tasks like `push-v2`, have different values for `obj_low` and `low_high`:
```python
class SawyerPushEnvV2(SawyerXYZEnv):
    TARGET_RADIUS=0.05

    def __init__(self):
        hand_low = (-0.5, 0.40, 0.05)
        hand_high = (0.5, 1, 0.5)
        obj_low = (-0.1, 0.6, 0.02)
        obj_high = (0.1, 0.7, 0.02)
        goal_low = (-0.1, 0.8, 0.01)
        goal_high = (0.1, 0.9, 0.02)
```

