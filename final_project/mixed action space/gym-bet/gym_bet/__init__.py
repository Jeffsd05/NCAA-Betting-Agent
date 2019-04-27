from gym.envs.registration import register

register(
    id='Bet-v0',
    entry_point='gym_bet.envs:BetEnv',
    kwargs={'datadir': 'final_project_part_2/gym-bet/data'}
)
