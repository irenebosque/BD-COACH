
import time
import metaworld
import random
import numpy as np

print('# Check out the available environments: ', metaworld.ML1.ENV_NAMES)  # Check out the available environments
ml1 = metaworld.ML1('soccer-v2') # Construct the benchmark, sampling tasks
env = ml1.train_classes['soccer-v2']()  # Create an environment with task `pick_place`
task = random.choice(ml1.train_tasks)
env.set_task(task)  # Set task

for i_episode in range(20):
    observation = env.reset()  # Reset environment
    for t in range(1000):
        env.render() 
        #print('observation: ', observation)
        print('observation shape : ', observation.shape)
        a = env.action_space.sample()  # Sample an action
        a = [0.05, 0.05, 0.05, 0.1]
        #print('action: ', a)
        observation, reward, done, info = env.step(a)  # Step the environoment with the sampled random action
        print('observation: ', observation)
        hand_pos =  observation[:3]
        unused_1 =  observation[3]
        ball_pos =  observation[4:7]
        unused_2 =  observation[7:-3]
        goal_pos =  observation[-3:]

        print('hand_pos =  observation[:3]', hand_pos)
        print('unused_1 =  observation[3]', unused_1)
        print('ball_pos =  observation[4:7]', ball_pos)
        print('unused_2 =  observation[7:-3]', unused_2)
        print('goal_pos =  observation[-3:]', goal_pos)

        obs_soccer_v2 = np.hstack((hand_pos, ball_pos, goal_pos))
        print('obs_soccer_v2: ', obs_soccer_v2)


        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()
 


