# A General Bilevel Optimization Framework for Meta-Reinforcement Learning: Provable Optimality and Generalization for One-Step Policy Adaptation (BOMRL)


#### Requirements
 - Python 3.8.16
 - PyTorch 1.7.1
 - Gym 0.15
 - Mujoco 150

## Usage

#### Training
You can use the [`train.py`](train.py) script in order to run reinforcement learning experiments with BOMRL. Note that by default, logs are available in [`train.py`](train.py) but **are not** saved (eg. the returns during meta-training). For example, to run the script on HalfCheetah-Vel:
```
python train.py --config configs/maml/halfcheetah-vel.yaml --output-folder bomrl-halfcheetah-vel --seed 1 --num-workers 8
```

#### Testing
Once you have meta-trained the policy, you can test it on the same environment using [`test.py`](test.py):
```
python test.py --config bomrl-halfcheetah-vel/config.json --policy bomrl-halfcheetah-vel/policy.th --output bomrl-halfcheetah-vel/results.npz --meta-batch-size 20 --num-batches 10  --num-workers 8
```

## References

This code is build based on an implementation code of MAML:
```
@misc{deleu2018mamlrl,
  author = {Tristan Deleu},
  title  = {{Model-Agnostic Meta-Learning for Reinforcement Learning in PyTorch}},
  note   = {Available at: https://github.com/tristandeleu/pytorch-maml-rl},
  year   = {2018}
}
```
Thanks for the implementation by the author.