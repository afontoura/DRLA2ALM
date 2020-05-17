A Deep Reinforcement Learning Approach to Asset-Liability Management 
====================================================================

This is an educational resource produced by Alan Fontoura, as part of his thesis on CEFET-RJ's computer science's Masters program.

This code has a fully functional reinforcement learning environment based on OpenAi's gym package. To install it, just clone this repo, navigate to its main folder and run 'pip install -e .'

To use the code, in a pyhton window, run:

```python
import gym
import envs
env = gym.make('ALM-v0')
```
`T` is the total time horizon, in years, which will be used. `rate` is the discount rate which equalizes initial assets and liability's present value. The higher the rate, the higher the liabilities. These parameters have to be manually changed in lines 26 and 27 of envs/custom_ALM/ALM_env.py