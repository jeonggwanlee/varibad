import numpy as np
import os
from learning_to_adapt.utils.serializable import Serializable
from gym.envs.mujoco.mujoco_env import MujocoEnv

from .mjlib import mjlib

# class MujocoEnv(gym.Env):
#
#     def __init__(self, model_path, frame_skip):
#         if model_path.startswith("/"):
#             fullpath = model_path
#         else:
#             fullpath = os.path.join(os.path.dirname(__file__), "assets", model_path)
#         if not path.exists(fullpath):
#             raise IOError("File %s does not exist" % fullpath)
#         self.frame_skip = frame_skip
#         self.model = mujoco_py.load_model_from_path(fullpath)
#         self.sim = mujoco_py.MjSim(self.model)
#         self.data = self.sim.data
#         # MjSim = MjModel + MjData``
#

class HalfCheetahHFieldEnv(MujocoEnv, Serializable):
    def __init__(self, task='hfield', reset_every_episode=False, reward=True, *args, **kwargs):
        Serializable.quick_init(self, locals())

        self.cripple_mask = None
        self.reset_every_episode = reset_every_episode
        self.first = True

        MujocoEnv.__init__(self,
                           os.path.join(os.path.abspath(os.path.dirname(__file__)),
                                        "assets",
                                        "half_cheetha_hfield.xml"),
                           frame_skip=5)

        task = None if task == 'None' else task

        # rgba when material is omitted (ngeom x 4)
        self._init_geom_rgba = self.model.geom_rgba.copy()
        # geom_contype : geom contact type (ngeom x 1)
        self._init_geom_contype = self.model.geom_contype.copy()
        # geom-specific size parameters (ngeom x 3)
        self._init_geom_size = self.model.geom_size.copy()
        # local position offset rel. to body
        self.init_geom_pos = self.model.geom_pos.copy()
        # Opt : options for mj_setLengthRange
        # timestep : simulation timestep; 0: use mjOption.timestep
        ## self.dt = self.model.opt.timestep

        assert task in [None, 'hfield', 'hill', 'basin', 'steep', 'gentle']

        self.task = task
        self.x_walls = np.array([250, 260, 261, 270, 280, 285])
        self.height_walls = np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2])
        self.height = 0.8
        self.width = 15

        # action_noise
        self.action_noise = 0.0

        self._body_comvels = None

    @property
    def action_bounds(self):
        return self.action_space.bounds

    def inject_action_noise(self, action):  # l2a mujoco_env
        # generate action noise
        noise = self.action_noise * np.random.normal(size=action.shape)
        # rescale the noise to make it  proportional to the action bounds
        lb, ub = self.action_bounds
        noise = 0.5 * (ub - lb) * noise
        return action + noise

    def forward_dynamics(self, action):   # l2a mujoco env
        ctrl = self.inject_action_noise(action)
        self.do_simulation(ctrl, self.frame_skip)
        self.model.forward()  #TODO why this part is needed?
        # subtree_com : center of mass of each subtree (nbody x 3)
        # Therefore, new_com is torco's com
        new_com = self.model.data.com_subtree[0]
        self.dcom = new_com - self.current_com
        self.current_com = new_com

    def get_current_obs(self):
        return np.concatenate([
            self.model.data.qpos.flatten()[1:],
            self.model.data.qvel.flat,
            self.get_body_com("torso").flat,
        ])

    def _compute_subtree(self):
        body_vels = np.zeros((self.model.nbody, 6))
        #
        mass = self.body_mass.flatten()
        for i in range(self.model.nbody):
            # body velocity
            mjlib.mj_objectVelocity(
                self.
            )

    @property
    def body_comvels(self):
        if self._body_comvels is None:
            self._body_comvels = self._compute_subtree()
        return self._body_comvels

    def get_body_comvel(self, body_name):  # l2a mujoco env
        idx = self.model.body_names.index(body_name)
        return self.model.data.body_comvels[idx]

    def step(self, action):
        self.forward_dynamics(action)
        next_obs = self.get_current_obs()
        ctrl_cost = 1e-1 * 0.5 * np.sum(np.square(action))  # 1/20 * sum of square
        # comVel // Compute cvel, cdof_dot
        # cvel // com-based velocity [3D rot; 3D tran] (nbody x 6)
        # cdof_dot // time-derivative of cdof (com-based motion axis of each dof)
        forward_reward = self.get_body_comvel("torso")[0]  # x axis c