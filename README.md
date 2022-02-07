# DCOACH with Human Model
This repository contains the code for my master thesis “Towards Robust Corrective Interactive Deep Learning in Data Intensive Environments”


### Video

[![Watch the video](https://img.youtube.com/vi/aWCvxShtHk4/default.jpg)](https://youtu.be/aWCvxShtHk4)

### Full pdf
[Link to the repository](https://repository.tudelft.nl/islandora/object/uuid%3A44ee5fb4-9828-47f1-9545-be88263a9d5a?collection=education)

### The algorithm in a nutshell

![image](https://user-images.githubusercontent.com/56187267/152881935-1a3de218-b2e6-4641-b76b-1b41c7c98ed0.png)


### Abstract


*Interactive imitation learning refers to learning methods where a human teacher interacts with an agent during the learning process providing feedback to improve its behaviour. This type of learning may be preferable with respect to reinforcement learning techniques when dealing with real-world problems. This fact is especially true in the case of robotic applications where reinforcement learning may be unfeasible as there are long training times and reward functions can be hard to shape/compute.*

*The present thesis focuses on interactive learning with corrective feedback and, in particular, in the framework Deep Corrective Advice Communicated by Humans (D-COACH), which has successfully shown to be advantageous in terms of training time and data efficiency. D-COACH, a supervised learning method whose policy is represented by an artificial neural network, incorporates a replay buffer where samples of states and corresponding labels gathered by the agent's policy from human feedback are stored and replayed. However, this causes conflicts between the data in the buffer because samples collected by older versions of the policy may be contradictory and could deteriorate the performance of the current policy. In order to reduce this issue, the current implementation of D-COACH uses a first-in-first-out buffer with limited size, as the older the sample is, the more likely it is to deteriorate the performance of the learner. Nonetheless, this limitation propitiates catastrophic forgetting, an inherent tendency of neural networks to forget what they have already learnt, and that can be mitigated by replaying information gathered during all the stages of the problem. Therefore, D-COACH suffers from a trade-off between reducing conflicting data and avoiding catastrophic forgetting. The fact that D-COACH limits the size of its buffer automatically restricts the types of problems that it can solve, given that, if the problem is too complex (i.e. it requires large amounts of data), it simply will not be able to remember everything.*

*If we want to utilise a buffer to train data intensive tasks with corrective feedback, a new method is needed to solve the problem of using information gathered by older versions of the policy. We propose an improved version of D-COACH, which we call Batch Deep COACH (BD-COACH, pronounced “be the coach”). BD-COACH incorporates a human model module that learns the feedback from the teacher and that can be employed to make corrections gathered by older versions of the policy still useful for batch updating the current version of the policy.*

*To compare the performance of BD-COACH with respect to D-COACH, three simulated experiments were done using the open-source Meta-World benchmark, which is based on MuJoCo and OpenAI gym. Moreover, to validate the proposed method in a real setup, two planar manipulation tasks were solved using a seven degrees of freedom KUKA robot arm.*

*Furthermore, we present an analysis between on-policy and off-policy methods both in the fields of reinforcement learning and in imitation learning. We believe there is an interesting simile between this classification and the problem of correctly implementing a replay buffer when learning from corrective feedback.*



