from gym.envs.registration import register
from ad_envs.online_adaptation_suite.half_cheetah_joint_env import HalfCheetahJointEnv
register(
    id='half-cheetah-block-v0',
    entry_point='ad_envs.online_adaptation_suite:HalfCheetahBlockEnv',
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
    id='half-cheetah-join-v0',
    entry_point='ad_env.online_adaptation_suite:HalfCheetahJointEnv',
    max_episode_steps=200,
)
from ad_envs.online_adaptation_suite.half_cheetah_joint_env import HalfCheetahJointEnv