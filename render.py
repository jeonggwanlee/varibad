

from metalearner import MetaLearner
import argparse
from config.mujoco import args_mujoco_cheetah_joint_varibad, args_mujoco_cheetah_hfield_varibad, \
    args_mujoco_cheetah_blocks_varibad
import matplotlib
matplotlib.use('Agg')

parser = argparse.ArgumentParser()
parser.add_argument('--env-type', default='gridworld_varibad')
args, rest_args = parser.parse_known_args()
env = args.env_type

#args = args_mujoco_cheetah_joint_varibad.get_args(rest_args)
#args = args_mujoco_cheetah_hfield_varibad.get_args(rest_args)
args = args_mujoco_cheetah_blocks_varibad.get_args(rest_args)

metalearner = MetaLearner(args)

metalearner.load_and_render(load_iter=4000)
#metalearner.load(load_iter=3500)