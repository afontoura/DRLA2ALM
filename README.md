A Deep Reinforcement Learning Approach to Asset-Liability Management 
====================================================================

This is an educational resource produced by Alan Fontoura, as part of his thesis on CEFET-RJ's computer science's Masters program.

This code has a fully functional reinforcement learning environment base on OpenAi's gym package. To install it, just clone this repo, navigate to its main folder and run 'pip install -e .'

To use the code, in a pyhton window, run:

```python
import gym
import envs
env = gym.make('ALM-v0', T = 30, rate = .06)
```
