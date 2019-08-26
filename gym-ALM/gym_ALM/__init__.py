from gym.envs.registration import register

register(id = 'ALM-v0',
         entry_point = 'gym_ALM.envs:ALMEnv')
