from gym.envs.registration import register

register(
    id = 'ALM-v0',
    entry_point='envs.custom_ALM:ALM',
    kwargs = {'T': 30, 'rate': .06, 'hist_returns': False}
)
