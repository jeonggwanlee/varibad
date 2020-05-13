import numpy as np
from learning_to_adapt.utils.serializable import Serializable
#from utils.serializable import Serializable
from learning_to_adapt.envs.mujoco_env import MujocoEnv
#from gym.envs.mujoco.mujoco_env import MujocoEnv
from learning_to_adapt.logger import logger

from learning_to_adapt import spaces

import os


class HalfCheetahHFieldEnv(MujocoEnv, Serializable):
    def __init__(self, task='hfield', reset_every_episode=False, reward=True, *args, **kwargs):
        Serializable.quick_init(self, locals())

        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self.first = True

        self.action_noise = 0.0

        MujocoEnv.__init__(self, os.path.join(os.path.abspath(os.path.dirname(__file__)), "assets", "half_cheetah_hfield.xml"),
                           5)  # frame skip

        task = None if task == 'None' else task

        self._init_geom_rgba = self.model.geom_rgba.copy()
        self._init_geom_contype = self.model.geom_contype.copy()
        self._init_geom_size = self.model.geom_size.copy()
        self._init_geom_pos = self.model.geom_pos.copy()
        self.dt = self.model.opt.timestep

        assert task in [None, 'hfield', 'hill', 'basin', 'steep', 'gentle']

        self.task = task
        self.x_walls = np.array([250, 260, 261, 270, 280, 285])
        self.height_walls = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.height = 0.8
        self.width = 15


    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def get_body_xmat(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.xmat[idx].reshape((3, 3))

    def get_body_com(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.data.com_subtree[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))
        forward_reward = self.get_body_comvel("torso")[0]
        reward = forward_reward - ctrl_cost
        done = False
        info = {}
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
        #super(HalfCheetahHFieldEnv, self).reset_mujoco(init_state=init_state)
        self.reset_mujoco(init_state=init_state)
        if self.reset_every_episode and not self.first:
            self.reset_task()

        if self.first:
            self.first = False

    def reset_task(self, value=None):
        if self.task == 'hfield':
            height = np.random.uniform(0.2, 1)
            width = 10
            n_walls = 6
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            x_walls = np.random.choice(np.arange(255, 310, width), replace=False, size=n_walls)
            x_walls.sort()
            sign = np.random.choice([1, -1], size=n_walls)
            sign[:2] = 1
            height_walls = np.random.uniform(0.2, 0.6, n_walls) * sign
            row = np.zeros((500,))
            for i, x in enumerate(x_walls):
                terrain = np.cumsum([height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x+width:] = row[x+width - 1]
            row = (row - np.min(row))/(np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'basin':
            self.height_walls = np.array([-1, 1, 0., 0., 0., 0.])
            self.height = 0.55
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'hill':
            self.height_walls = np.array([1, -1, 0, 0., 0, 0])
            self.height = 0.6
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'gentle':
            self.height_walls = np.array([1, 1, 1, 1, 1, 1])
            self.height = 1
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task == 'steep':
            self.height_walls = np.array([1, 1, 1, 1, 1, 1])
            self.height = 4
            height = self.height
            width = self.width
            self.x_walls = np.array([255, 270, 285, 300, 315, 330]) - 5
            self.model.hfield_size = np.array([50, 5, height, 0.1])
            row = np.zeros((500,))
            for i, x in enumerate(self.x_walls):
                terrain = np.cumsum([self.height_walls[i]] * width)
                row[x:x + width] += terrain
                row[x + width:] = row[x + width - 1]
            row = (row - np.min(row)) / (np.max(row) - np.min(row))
            hfield = np.tile(row.reshape(-1, 1), (1, 528)).T.reshape(-1, 1)
            self.model.hfield_data = hfield

        elif self.task is None:
            pass

        else:
            raise NotImplementedError

        self.model.forward()

    #@property
    #def action_space(self):
    #    bounds = self.model.actuator_ctrlrange
    #    lb = bounds[:, 0]
    #    ub = bounds[:, 1]
    #    return spaces.Box(lb, ub)

    @property
    def action_bounds(self):
        bounds = self.model.actuator_ctrlrange
        return (bounds[:, 0], bounds[:, 1])

    def inject_action_noise(self, action):
        # generate action noise
        noise = self.action_noise * \
                np.random.normal(size=action.shape)
        # rescale the noise to make it proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def forward_dynamics(self, action):
        self.data.ctrl = self.inject_action_noise(action)
        for _ in range(self.frame_skip):
            self.model.step()
        self.model.forward()
        new_com = self.model.data.com_subtree[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

    def get_body_comvel(self, body_name):
        idx = self.model.body_names.index(body_name)
        return self.model.body_comvels[idx]

    def reset_mujoco(self, init_state=None):
        if init_state is None:
            if self.random_init_state:
                self.model.data.qpos = self.init_qpos + \
                    np.random.normal(size=self.init_qpos.shape) * 0.01
                self.model.data.qvel = self.init_qvel + \
                    np.random.normal(size=self.init_qvel.shape) * 0.1
            else:
                self.model.data.qpos = self.init_qpos
                self.model.data.qvel = self.init_qvel

            self.model.data.qacc = self.init_qacc
            self.model.data.ctrl = self.init_ctrl
        else:
            start = 0
            for datum_name in ["qpos", "qvel", "qacc", "ctrl"]:
                datum = getattr(self.model.data, datum_name)
                datum_dim = datum.shape[0]
                datum = init_state[start: start + datum_dim]
                setattr(self.model.data, datum_name, datum)
                start += datum_dim


    def log_diagnostics(self, paths, prefix):
        progs = [
            path["observations"][-1][-3] - path["observations"][0][-3]
            for path in paths
            ]
        logger.logkv(prefix + 'AverageForwardProgress', np.mean(progs))
        logger.logkv(prefix + 'MaxForwardProgress', np.max(progs))
        logger.logkv(prefix + 'MinForwardProgress', np.min(progs))
        logger.logkv(prefix + 'StdForwardProgress', np.std(progs))


if __name__ == '__main__':
    env = HalfCheetahHFieldEnv(task='hfield')
    while True:
        env.reset()
        env.reset_task()
        for _ in range(100):
            env.render()
            env.step(env.action_space.sample())
        env.stop_viewer()





