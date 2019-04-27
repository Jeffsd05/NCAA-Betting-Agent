from gym.envs.registration import register

register(
    id='Bet-v0',
    entry_point='gym_bet.envs:BetEnv',
    kwargs={'datadir': 'data'}
)
