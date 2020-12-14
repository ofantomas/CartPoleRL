import os
import json
import inspect
import click
import torch
from torch.optim import lr_scheduler as lrs
# Core setup and run for CCA tabular experiments
#from git import Repo

from agents import ReinforceAgent, RandomAgent, \
                   ValueBaselineAgent,OptimalStateBaselineAgent,\
                   ActionStateBaselineAgent,TrajectoryCVAgent, \
                   DynamicsTrajCVAgent, ModelFreeDynamicsTrajCVAgent
from envs import get_sticky_actions_gym_env
from logger import LoggingManager, WandbLogger, LocalLogger
from train import train
from utils import plot_policy


#def get_current_sha():
#    repo = Repo('.')
#    # TODO maybe too harsh here, maybe need to add debugger check
#    if len(repo.index.diff(None)) > 0:
#        raise Exception("There are uncommited changes in repo!")
#    return repo.head.commit.hexsha


# ENV_CONSTRUCTORS = {
#     "test": lambda: TestEnv(),
#     "small_grid": lambda: SmallGridEnv(),
#     "small_grid_extra_actions": lambda: SmallGridExtraActionsEnv(),
#     "small_grid_no_no_op": lambda: SmallGridNoNoOpEnv(),
#     "small_grid_not_done": lambda: SmallGridNotDoneEnv(),
#     "shortcut_hca": lambda: ShortcutEnv(),
#     "delayed_hca": lambda: DelayedEffectEnv(),
#     "delayed_hca_noisy": lambda: DelayedEffectEnv(3, 1),
#     "delayed_10step": lambda: DelayedEffectEnv(8),
#     "delayed_100step": lambda: DelayedEffectEnv(98),
#     "delayed_100step_noisy": lambda: DelayedEffectEnv(98, 0.1),
#     "ambiguous_hca": lambda: AmbiguousBanditEnv(),
#     "frozenlake": lambda slip_rate: FrozenLakeEnv(slip_rate),
#     "counterexample_bandit": lambda: CounterexampleBanditEnv(),
#     "counterexample_bandit_2": lambda: CounterexampleBandit2Env()
# }


def ignore_extra_args(func, **kwargs):
    signature = inspect.signature(func)
    expected_keys = [p.name for p in signature.parameters.values()
                        if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
    filtered_kwargs = {k: kwargs[k] for k in kwargs if k in expected_keys}
    return filtered_kwargs


AGENT_CONSTRUCTORS = {
    "reinforce": ReinforceAgent,
    "value_baseline": ValueBaselineAgent,
    "optimal_state_baseline" : OptimalStateBaselineAgent,
    "state_action_baseline": ActionStateBaselineAgent,
    "traj_cv": TrajectoryCVAgent,
    "dynamics_traj_cv": DynamicsTrajCVAgent,
    "model_free_dynamics_traj_cv": ModelFreeDynamicsTrajCVAgent,
    "random": RandomAgent
}


LR_SCHEDULERS = {
    "OneCycle": lrs.OneCycleLR,
    "MultiStep": lrs.MultiStepLR,
    "MultiplicativeLR": lrs.MultiplicativeLR
}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--device", type=click.Choice(['cpu', 'cuda']), default='cpu')
@click.option("--model_type", type=click.Choice(AGENT_CONSTRUCTORS.keys()), required=True)
@click.option("--env_type", type=str, default='CartPole-v0')
@click.option("--env_sticky_prob", type=float, default=0.25)
@click.option("--episodes", type=int, required=True)
@click.option("--epi_length", type=int, required=True)
@click.option("--eps_per_train", type=int, default=1)
@click.option("--lr_scheduler", type=click.Choice(LR_SCHEDULERS.keys()), default=None)
@click.option("--lr_scheduler_kwargs", type=str, default='{}')
@click.option("--alpha", type=float, default=0.1)
@click.option("--beta", type=float, default=None)
@click.option("--delta", type=float, default=None)
@click.option("--gamma_env", type=float, default=0.99)
@click.option("--run_n_times", type=int, default=1)
@click.option("--log_freq", type=int, default=1)
@click.option("--project", type=str, default=None)
@click.option("--log_folder", type=str, default=None)
@click.option("--exp_name", type=str, default='experiment')
@click.option("--show_policy", type=bool, default=False)
@click.option("--estimate_variance", type=bool, default=False)
def run(
    device,
    model_type,
    env_type,
    env_sticky_prob,
    episodes,
    epi_length,
    eps_per_train,
    lr_scheduler,
    lr_scheduler_kwargs,
    alpha,
    beta,
    delta,
    gamma_env,
    run_n_times,
    log_freq,
    project,
    log_folder,
    exp_name,
    show_policy,
    estimate_variance
):
    try:
        lr_scheduler_kwargs = json.loads(lr_scheduler_kwargs)
    except json.decoder.JSONDecodeError as exception:
        print("Unable to parse lr_scheduler_kwargs. JSON error: " + str(exception) + ".")

    device_torch = torch.device(device)

    for _ in range(run_n_times):
        config = {
            'model_type': model_type,
            'env_type': env_type,
            'episodes': episodes,
            'epi_length': epi_length,
            'alpha': alpha,
            'beta': beta,
            'delta': delta,
            'gamma_env': gamma_env,
            'eps_per_train': eps_per_train,
            #'git_sha': get_current_sha(),
            'exp_name': exp_name,
        }
        logger_manager = LoggingManager()
        if project is not None:
            logger_manager.add_logger(WandbLogger(project, exp_name))
        if log_folder is not None:
            local_logger = LocalLogger(log_folder, exp_name)
            current_run_folder = local_logger.current_run_folder
            logger_manager.add_logger(local_logger)
        logger_manager.log_config(config)
        
        env = get_sticky_actions_gym_env(env_type, env_sticky_prob)
        state_space_shape = env.observation_space.shape
        n_actions = env.action_space.n
        
        if model_type not in AGENT_CONSTRUCTORS:
            raise Exception("AGENT TYPE NOT IMPLEMENTED")
        else:
            scheduler = None
            if lr_scheduler is not None:
                scheduler = LR_SCHEDULERS[lr_scheduler]
            agent = AGENT_CONSTRUCTORS[model_type]
            agent_kwargs = ignore_extra_args(agent,
                                             env_shape=state_space_shape,
                                             n_actions=n_actions,
                                             alpha=alpha,
                                             beta=beta,
                                             delta=delta,
                                             gamma_env=gamma_env,
                                             device=device_torch,
                                             episode_length=epi_length,
                                             lr_scheduler=scheduler)
            agent = agent(**{**agent_kwargs, **lr_scheduler_kwargs})

        train(agent, env, episodes, epi_length, eps_per_train, log_freq=log_freq,
              logger=logger_manager, estimate_variance=estimate_variance)

        if show_policy is True:
            if log_folder is not None:
                output_path = os.path.join(current_run_folder, 'policy.pdf')
                plot_policy(env, agent.pi.detach().numpy(), epi_length, output_path)
            else:
                print('Log folder must not be None in order to save policy plot.')

if __name__ == "__main__":
    cli()
