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

## Timesteps
Wall clock time and simulated time are 2 different things. Simulation time is **not the real-time** that it takes to run a simulation (this is the wall-clock time); Simulated time is a variable maintained by the simulation program and is the one plotted in the x-axis of the report. 
When rendering the meta-world environment, we can see at the lower right corner information about the time. The `step` show there is computed in
`mjviewer.py`:

```python
step = round(self.sim.data.time / self.sim.model.opt.timestep)
```
Where `self.sim.model.opt.timestep` is the duration of a simulated timestep which in the case of this benchmark is 0.0025, and `self.sim.data.time` is the simulated time until that point.
It seems that the `step` value is updated every 5 steps. The simulation starts with a value of `step = 250` and ends when `step = 750` therefore there are 500 steps.

Indepently that we have delays due to our own computers or added sleep times, the simulation time should always be the same




```python
# timestep 98
self.sim.data.time 1.8499999999999717 
self.sim.model.opt.timestep 0.0025
# step = 1.8499999999999717 / 0.0025 = 740

# timestep 99
self.sim.data.time 1.8624999999999714
self.sim.model.opt.timestep 0.0025
# step = 1.8624999999999714 / 0.0025 = 745
env.reset
```

The mujoco/metaworld simulation states that every episode consists on 500 timesteps, but because it counts every 5 timesteps, it is necessary to multiply the provided timestep duration by 5 (We perform 100 calls to `env.step(action)`, not 500 calls:

1.86249999999997-1.84999999999997 = 0.0025
0.0025 * 5 = 0.0125 seconds/timestep

## Episode length

The episode length for metaworld v2 is 500 which translates in 100 calls to `env.step(action)` 

_"The max_path_length limit is intended to ensure that results generated with metaworld are reproducible across research papers. The performance of two algorithms can be drastically different using two different max_path_length numbers, given the same number of environment samples." 
[source](https://githubmemory.com/repo/rlworkgroup/metaworld/issues/157)_

I asked about this issue on the Meta-World Garage Slack:

>**Question:**
> 
>Irene Bosque  12:38 PM
Hi, I have a question regarding the max_path_length parameter.
In the file /home/irene/metaworld-task/lib/python3.6/site-packages/metaworld/envs/mujoco/mujoco_env.py appears as max_path_length = 150
But in metaworld/metaworld/envs/mujoco/sawyer_xyz/sawyer_xyz_env.py it is set to max_path_length = 500 and this is the value that seems to have effect. I would like to know why are those values different and on which depends to use one value or the other
Thanks for this awesome benchmark!
> 
> **Response:**
> Avnish Narayan
>Hey sorry for the late response!
>Mujoco env is the base class, and it used to be that the max path length of environments was 150. **We upped it to 500 in metaworld v2**.This is an artifact from old code. It being 150 shouldn’t have a bearing on any code you write, unless you’re accessing the super super class’s max path length of your environment. 


```python
max_episode_steps=env.max_path_length
```

# Meta-World observation

An observation is the sum of:
`obs = np.hstack((curr_obs, self._prev_obs, pos_goal))`
### Example observation
```
obs [-0.01423575  0.56735747  0.15540215  0.26406011  0.          0.6
  0.015       0.          0.          0.          1.          0.
  0.          0.          0.          0.          0.          0.
 -0.01246431  0.57015151  0.15802904  0.2835925   0.          0.6
  0.015       0.          0.          0.          1.          0.
  0.          0.          0.          0.          0.          0.
 -0.0017014   0.88267895  0.        ]
```
The first 3 elements are the position of the hand, and the fourth, the gripper. From 4 to 7, it is the position of the object.
```
curr_obs [-0.01423575  0.56735747  0.15540215  0.26406011  0.          0.6
  0.015       0.          0.          0.          1.          0.
  0.          0.          0.          0.          0.          0.        ]
```

```
self._prev_obs [-0.01246431  0.57015151  0.15802904  0.2835925   0.          0.6
  0.015       0.          0.          0.          1.          0.
  0.          0.          0.          0.          0.          0.        ]
```

The last 3 elements are the goal position
```
pos_goal [-0.0017014   0.88267895  0.        ]
```