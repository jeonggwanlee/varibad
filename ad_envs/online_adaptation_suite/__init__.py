from gym.envs.registration import register
from ad_envs.online_adaptation_suite.mj_env import MujocoEnv


register(
    id='half-cheetah-block-v0',
    entry_point='ad_envs.online_adaptation_suite:HalfCheetahBlocksEnv',
    max_episode_steps=200,
)
from ad_envs.online_adaptation_suite.half_cheetah_blocks_env import HalfCheetahBlocksEnv

register(
    id='half-cheetah-hfield-v0',
    entry_point='ad_envs.online_adaptation_suite:HalfCheetahHFieldEnv',
    max_episode_steps=200,
)
from ad_envs.online_adaptation_suite.half_cheetah_hfield_env import HalfCheetahHFieldEnv

register(
    id='half-cheetah-joint-v0',
    entry_point='ad_envs.online_adaptation_suite:HalfCheetahJointEnv',
    max_episode_steps=200,
)
from ad_envs.online_adaptation_suite.half_cheetah_joint_env import HalfCheetahJointEnv