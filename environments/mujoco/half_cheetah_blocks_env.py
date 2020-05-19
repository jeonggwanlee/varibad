import numpy as np
# from learning_to_adapt.utils.serializable import Serializable
# from learning_to_adapt.envs.mujoco_env import MujocoEnv
# from learning_to_adapt.logger import logger
import os

from environments.mujoco.mj_env import MujocoEnv


#class HalfCheetahBlocksEnv(MujocoEnv, Serializable):
class HalfCheetahBlocksEnv(MujocoEnv):


    def __init__(self, task='damping', max_episode_steps=200, reset_every_episode=False):
        #Serializable.quick_init(self, locals())

        self.reset_every_episode = reset_every_episode
        self.first = True
        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                              "assets", "half_cheetah_blocks.xml"))
        task = None if task == 'None' else task

        self.cripple_mask = np.ones(self.action_space.shape)

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        assert task in [None, 'damping']

        self.task = task

        self._max_episode_steps = max_episode_steps
        self.visualise_behaviour = True

    def get_current_obs(self):
        return np.concatenate([
            self.data.qpos.flatten()[9:],
            self.data.qvel.flat[8:],
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.data.subtree_com[idx]

    def get_task(self):
        return 1  # dummy

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.forward_dynamics(action)
        xy_position_after = self.get_body_com("torso")[:2].copy()
        next_obs = self.get_current_obs()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        #forward_reward = x_velocity
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False

        info = {
            'reward_forward': forward_reward,
            'reward_ctrl': -ctrl_cost,

            'x_postiion': xy_position_after[0],
            'y_position': xy_position_after[1],
            'distance_from_origin': np.linalg.norm(xy_position_after, ord=2),

            'x_velocity': x_velocity,
            'y_velocity': y_velocity,

            'forward_reward': forward_reward,
            'task': 1, ## dummy ## TODO
        }
        return next_obs, reward, done, info

    def reward(self, obs, action, next_obs):
        assert obs.ndim == 2
        assert obs.shape == next_obs.shape
        assert obs.shape[0] == action.shape[0]
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action), axis=1)
        forward_reward = (next_obs[:, -3] - obs[:, -3])/self.dt
        reward = forward_reward - ctrl_cost
        return reward

    def reset_mujoco(self, init_state=None):
        super(HalfCheetahBlocksEnv, self).reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()
        if self.first:
            self.first = False

    def reset_task(self, value=None):
        if self.task == 'damping':
            damping = self.model.dof_damping.copy()
            damping[:8] = value if value is not None else np.random.uniform(0, 10, size=8)

            for idx in range(8):
                self.model.dof_damping[idx] = damping[idx]

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.sim.forward()

    # def log_diagnostics(self, paths):
    #     progs = [
    #         path["observations"][-1][-3] - path["observations"][0][-3]
    #         for path in paths
    #         ]
    #     logger.logkv('AverageForwardProgress', np.mean(progs))
    #     logger.logkv('MaxForwardProgress', np.max(progs))
    #     logger.logkv('MinForwardProgress', np.min(progs))
    #     logger.logkv('StdForwardProgress', np.std(progs))


if __name__ == '__main__':
    env = HalfCheetahBlocksEnv(task='damping')
    while True:
        env.reset()
        env.reset_task()
        for _ in range(1000):
            env.step(env.action_space.sample())
            env.render()


