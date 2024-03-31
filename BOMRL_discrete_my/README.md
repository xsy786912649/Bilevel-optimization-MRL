# A General Bilevel Optimization Framework for Meta-Reinforcement Learning: Provable Optimality and Generalization for One-Step Policy Adaptation (BOMRL)

#### Discrete state action space

#### Requirements
 - Python 3.10.9
 - PyTorch 1.13.1
 - Gym 0.26.2
 - Mujoco is included in Gym 0.26.2

## Usage

#### Training
You can use the [`train.py`](train.py) script in order to run reinforcement learning experiments.
```
python train.py 
```

#### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test.py
```

