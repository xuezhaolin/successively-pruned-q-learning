"""
DQN in async mode with CPU parallel sampler.
"""

from rlpyt.utils.launching.affinity import make_affinity
from rlpyt.samplers.async_.cpu_sampler import AsyncCpuSampler
from rlpyt.envs.atari.atari_env import AtariEnv, AtariTrajInfo
from rlpyt.algos.dqn.dqn import DQN
from rlpyt.agents.dqn.atari.atari_dqn_agent import AtariDqnAgent
from rlpyt.runners.async_rl import AsyncRlEval
from rlpyt.utils.logging.context import logger_context
import torch

def build_and_train(game="pong",
                    run_ID=0,
                    double_dqn=False,
                    clipped_dqn_sy=False,
                    target_update_k_interval=312,
                    using_target=None,
                    K=1):
    # Change these inputs to match local machine and desired parallelism.
    affinity=make_affinity(
        run_slot=0,
        n_cpu_core=8,  # Use 16 cores across all experiments.
        n_gpu=1,  # Use 8 gpus across all experiments.
        sample_gpu_per_run=0,
        async_sample=True,
        # hyperthread_offset=8,  # If machine has 24 cores.
        # n_socket=1,  # Presume CPU socket affinity to lower/upper half GPUs.
        # gpu_per_run=1,  # How many GPUs to parallelize one run across.
        # cpu_per_run=1,
    )

    sampler=AsyncCpuSampler(
        EnvCls=AtariEnv,
        TrajInfoCls=AtariTrajInfo,
        env_kwargs=dict(game=game),
        batch_T=4,
        batch_B=4,
        max_decorrelation_steps=100,
        eval_env_kwargs=dict(game=game),
        eval_n_envs=4,
        eval_max_steps=int(2e5),
        eval_max_trajectories=2,
    )
    if clipped_dqn_sy:
        type='Sy'
    algo = DQN(
        replay_ratio=8,
        min_steps_learn=1e4,
        replay_size=int(1e5),
        double_dqn=double_dqn,
        target_update_k_interval=target_update_k_interval,
        type=type,  # type Sy or De
        K=K,
        using_target=using_target
    )
    agent=AtariDqnAgent(number_K=K)

    runner=AsyncRlEval(
        algo=algo,
        agent=agent,
        sampler=sampler,
        n_steps=20e6,
        log_interval_steps=1e4,
        affinity=affinity,
    )

    config=dict(game=game)

    if double_dqn:
        log_dir="async_ddqn"
    elif clipped_dqn_sy:
        log_dir='async_cdqn_sy_t'
    else:
        log_dir='async_dqn'
    name = log_dir+'_' + game
    with logger_context(log_dir,run_ID,name,config):
        runner.train()


if __name__ == "__main__":
    torch.backends.cudnn.enabled = False
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--game', help='Atari game', default='pong')
    parser.add_argument('--run_ID', help='run identifier (logging)', type=int, default=0)
    args = parser.parse_args()
    for ii in range(1):
        build_and_train(game=args.game,run_ID=args.run_ID,double_dqn=False,clipped_dqn_sy=True,using_target=True,K=2)

