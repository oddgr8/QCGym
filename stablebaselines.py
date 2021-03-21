from stable_baselines import DQN, PPO2, A2C, ACKTR, TRPO
from stable_baselines.common.cmd_util import make_vec_env

from QCGym.environments.generic_env_cont_reward import GenericEnv
from QCGym.hamiltonians.cross_resonance_bb import CrossResonance

env = GenericEnv()
env = make_vec_env(lambda: env, n_envs=1)

from stable_baselines.common.evaluation import evaluate_policy
from stable_baselines.common.policies import MlpPolicy

# Random Agent, before training

model = PPO2(MlpPolicy, env, verbose=0)
model.learn(total_timesteps=450000)
env.close()
model.save("ppo_cartpole")
