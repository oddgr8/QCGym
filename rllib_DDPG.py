import ray
from ray import tune
from ray.rllib.agents.registry import get_agent_class
from ray.rllib.agents import with_common_config
from ray.rllib.models import ModelCatalog
from ray.tune import run_experiments
from ray.tune.registry import register_env
from ray.rllib.agents.ppo import PPOTrainer
from QCGym.environments.generic_env import GenericEnv

DEFAULT_CONFIG = with_common_config({
    # === Twin Delayed DDPG (TD3) and Soft Actor-Critic (SAC) tricks ===
    # TD3: https://spinningup.openai.com/en/latest/algorithms/td3.html
    # In addition to settings below, you can use "exploration_noise_type" and
    # "exploration_gauss_act_noise" to get IID Gaussian exploration noise
    # instead of OU exploration noise.
    # twin Q-net
    "twin_q": False,
    # delayed policy update
    "policy_delay": 1,
    # target policy smoothing
    # (this also replaces OU exploration noise with IID Gaussian exploration
    # noise, for now)
    "smooth_target_policy": False,
    # gaussian stddev of target action noise for smoothing
    "target_noise": 0.2,
    # target noise limit (bound)
    "target_noise_clip": 0.5,

    # === Evaluation ===
    # Evaluate with epsilon=0 every `evaluation_interval` training iterations.
    # The evaluation stats will be reported under the "evaluation" metric key.
    # Note that evaluation is currently not parallelized, and that for Ape-X
    # metrics are already only reported for the lowest epsilon workers.
    "evaluation_interval": None,
    # Number of episodes to run per evaluation period.
    "evaluation_num_episodes": 10,

    # === Model ===
    # Apply a state preprocessor with spec given by the "model" config option
    # (like other RL algorithms). This is mostly useful if you have a weird
    # observation shape, like an image. Disabled by default.
    "use_state_preprocessor": False,
    # Postprocess the policy network model output with these hidden layers. If
    # use_state_preprocessor is False, then these will be the *only* hidden
    # layers in the network.
    "actor_hiddens": [400, 300],
    # Hidden layers activation of the postprocessing stage of the policy
    # network
    "actor_hidden_activation": "relu",
    # Postprocess the critic network model output with these hidden layers;
    # again, if use_state_preprocessor is True, then the state will be
    # preprocessed by the model specified with the "model" config option first.
    "critic_hiddens": [400, 300],
    # Hidden layers activation of the postprocessing state of the critic.
    "critic_hidden_activation": "relu",
    # N-step Q learning
    "n_step": 1,

    # === Exploration ===
    "exploration_config": {
        # DDPG uses OrnsteinUhlenbeck (stateful) noise to be added to NN-output
        # actions (after a possible pure random phase of n timesteps).
        "type": "OrnsteinUhlenbeckNoise",
        # For how many timesteps should we return completely random actions,
        # before we start adding (scaled) noise?
        "random_timesteps": 1000,
        # The OU-base scaling factor to always apply to action-added noise.
        "ou_base_scale": 0.1,
        # The OU theta param.
        "ou_theta": 0.15,
        # The OU sigma param.
        "ou_sigma": 0.2,
        # The initial noise scaling factor.
        "initial_scale": 1.0,
        # The final noise scaling factor.
        "final_scale": 1.0,
        # Timesteps over which to anneal scale (from initial to final values).
        "scale_timesteps": 10000,
    },
    # Number of env steps to optimize for before returning
    "timesteps_per_iteration": 1000,
    # Extra configuration that disables exploration.
    "evaluation_config": {
        "explore": False
    },
    # === Replay buffer ===
    # Size of the replay buffer. Note that if async_updates is set, then
    # each worker will have a replay buffer of this size.
    "buffer_size": 50000,
    # If True prioritized replay buffer will be used.
    "prioritized_replay": True,
    # Alpha parameter for prioritized replay buffer.
    "prioritized_replay_alpha": 0.6,
    # Beta parameter for sampling from prioritized replay buffer.
    "prioritized_replay_beta": 0.4,
    # Time steps over which the beta parameter is annealed.
    "prioritized_replay_beta_annealing_timesteps": 20000,
    # Final value of beta
    "final_prioritized_replay_beta": 0.4,
    # Epsilon to add to the TD errors when updating priorities.
    "prioritized_replay_eps": 1e-6,
    # Whether to LZ4 compress observations
    "compress_observations": False,
    # If set, this will fix the ratio of replayed from a buffer and learned on
    # timesteps to sampled from an environment and stored in the replay buffer
    # timesteps. Otherwise, the replay will proceed at the native ratio
    # determined by (train_batch_size / rollout_fragment_length).
    "training_intensity": None,

    # === Optimization ===
    # Learning rate for the critic (Q-function) optimizer.
    "critic_lr": 1e-3,
    # Learning rate for the actor (policy) optimizer.
    "actor_lr": 1e-3,
    # Update the target network every `target_network_update_freq` steps.
    "target_network_update_freq": 0,
    # Update the target by \tau * policy + (1-\tau) * target_policy
    "tau": 0.002,
    # If True, use huber loss instead of squared loss for critic network
    # Conventionally, no need to clip gradients if using a huber loss
    "use_huber": False,
    # Threshold of a huber loss
    "huber_threshold": 1.0,
    # Weights for L2 regularization
    "l2_reg": 1e-6,
    # If not None, clip gradients during optimization at this value
    "grad_clip": None,
    # How many steps of the model to sample before learning starts.
    "learning_starts": 1500,
    # Update the replay buffer with this many samples at once. Note that this
    # setting applies per-worker if num_workers > 1.
    "rollout_fragment_length": 1,
    # Size of a batched sampled from replay buffer for training. Note that
    # if async_updates is set, then each worker returns gradients for a
    # batch of this size.
    "train_batch_size": 256,

    # === Parallelism ===
    # Number of workers for collecting samples with. This only makes sense
    # to increase if your environment is particularly slow to sample, or if
    # you're using the Async or Ape-X optimizers.
    "num_workers": 0,
    # Whether to compute priorities on workers.
    "worker_side_prioritization": False,
    # Prevent iterations from going lower than this time span
    "min_iter_time_s": 1,
    "env": "cross_resonance"
})

# Driver code for training
def setup_and_train():

    # Create a single environment and register it
    def env_creator(_):
        return GenericEnv()
    single_env = GenericEnv(max_timesteps=100)
    env_name = "cross_resonance"
    register_env(env_name, env_creator)

    # # Get environment obs, action spaces and number of agents
    # obs_space = single_env.observation_space
    # act_space = single_env.action_space
    # num_agents = single_env.num_agents

    # # Create a policy mapping
    # def gen_policy():
    #     return (None, obs_space, act_space, {})

    # policy_graphs = {}
    # for i in range(num_agents):
    #     policy_graphs['agent-' + str(i)] = gen_policy()

    # def policy_mapping_fn(agent_id):
    #     return 'agent-' + str(agent_id)

    # Define configuration with hyperparam and training details
    config = {
        "log_level": "WARN",
        "num_workers": 3,
        "num_cpus_for_driver": 1,
        "num_cpus_per_worker": 1,
        # "num_gpus_per_worker": 0.33,
        "num_sgd_iter": 10,
        "train_batch_size": 128,
        "lr": 5e-3,
        "model": {"fcnet_hiddens": [300, 300, 300]},
        # "multiagent": {
        #     "policies": policy_graphs,
        #     "policy_mapping_fn": policy_mapping_fn,
        # },
        "env": "cross_resonance"}

    # Define experiment details
    exp_name = 'cross_res'
    exp_dict = {
        'name': exp_name,
        'run_or_experiment': 'DDPG',
        "stop": {
            "training_iteration": 1000
        },
        'checkpoint_freq': 20,
        "config": DEFAULT_CONFIG,
    }

    # Initialize ray and run
    ray.init()
    tune.run(**exp_dict)


if __name__ == '__main__':
    setup_and_train()

