import gym
import numpy as np
from envs.wrappers.time_limit import TimeLimit
from envs.wrappers.drS_reward import DrsRewardWrapper

import mani_skill2.envs
import envs.tasks.envs_with_stage_indicators


MANISKILL_TASKS = {
	'lift-cube': dict(
		env='LiftCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'pick-cube': dict(
		env='PickCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'stack-cube': dict(
		env='StackCube-v0',
		control_mode='pd_ee_delta_pos',
	),
	'pick-ycb': dict(
		env='PickSingleYCB-v0',
		control_mode='pd_ee_delta_pose',
	),
	'turn-faucet': dict(
		env='TurnFaucet-v0',
		control_mode='pd_ee_delta_pose',
	),
	'pick-place': dict(
		env='PickAndPlace_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='dense',
	),
	## Semi-sparse reward tasks with stage-indicators
	'pick-place-semi': dict (
		env='PickAndPlace_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'turn-faucet-semi': dict (
		env='TurnFaucet_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='semi_sparse', 
	),
	'pick-place-drS': dict (
		env='PickAndPlace_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
	'turn-faucet-drS': dict (
		env='TurnFaucet_DrS_reuse-v0',
		control_mode='pd_ee_delta_pose',
		reward_mode='drS', 
	),
}


def make_env(cfg):
	"""
	Make ManiSkill2 environment.
	"""
	if cfg.task not in MANISKILL_TASKS:
		raise ValueError('Unknown task:', cfg.task)
	assert cfg.obs == 'state', 'This task only supports state observations.'
	task_cfg = MANISKILL_TASKS[cfg.task]
	env = gym.make(
		task_cfg['env'],
		obs_mode='state',
		control_mode=task_cfg['control_mode'],
		render_camera_cfgs=dict(width=384, height=384),
		reward_mode=task_cfg.get("reward_mode", None),
	)
	
	# DrS Reward Wrapper
	if task_cfg.get("reward_mode", None) == "drS":
		env = DrsRewardWrapper(env, cfg.drS_ckpt)
	
	env = TimeLimit(env, max_episode_steps=200)
	env.max_episode_steps = env._max_episode_steps
	return env
