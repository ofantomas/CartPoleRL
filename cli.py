import os
import inspect
import click
from torch.optim import lr_scheduler as lrs
# Core setup and run for CCA tabular experiments
#from git import Repo

from agents import ReinforceAgent, RandomAgent, \
                   ValueBaselineAgent, PerfectValueBaselineAgent,\
                   OptimalStateBaselineAgent, ActionStateBaselineAgent,\
                   PerfectActionStateBaselineAgent, TrajectoryCVAgent,\
                   PerfectTrajectoryCVAgent, PerfectDynamicsTrajCVAgent,\
                   DynamicsTrajCVAgent, PerfectDynamicsEstQVTrajCVAgent,\
                   ModelFreeDynamicsTrajCVAgent, PerfectModelFreeDynamicsTrajCVAgent

from envs import TestEnv, SmallGridEnv, SmallGridExtraActionsEnv, SmallGridNoNoOpEnv,\
    SmallGridNotDoneEnv, ShortcutEnv, DelayedEffectEnv, AmbiguousBanditEnv, FrozenLakeEnv,\
    CounterexampleBanditEnv, CounterexampleBandit2Env
from logger import LoggingManager, WandbLogger, LocalLogger
from train import train
from utils import plot_policy


#def get_current_sha():
#    repo = Repo('.')
#    # TODO maybe too harsh here, maybe need to add debugger check
#    if len(repo.index.diff(None)) > 0:
#        raise Exception("There are uncommited changes in repo!")
#    return repo.head.commit.hexsha


ENV_CONSTRUCTORS = {
    "test": lambda: TestEnv(),
    "small_grid": lambda: SmallGridEnv(),
    "small_grid_extra_actions": lambda: SmallGridExtraActionsEnv(),
    "small_grid_no_no_op": lambda: SmallGridNoNoOpEnv(),
    "small_grid_not_done": lambda: SmallGridNotDoneEnv(),
    "shortcut_hca": lambda: ShortcutEnv(),
    "delayed_hca": lambda: DelayedEffectEnv(),
    "delayed_hca_noisy": lambda: DelayedEffectEnv(3, 1),
    "delayed_10step": lambda: DelayedEffectEnv(8),
    "delayed_100step": lambda: DelayedEffectEnv(98),
    "delayed_100step_noisy": lambda: DelayedEffectEnv(98, 0.1),
    "ambiguous_hca": lambda: AmbiguousBanditEnv(),
    "frozenlake": lambda slip_rate: FrozenLakeEnv(slip_rate),
    "counterexample_bandit": lambda: CounterexampleBanditEnv(),
    "counterexample_bandit_2": lambda: CounterexampleBandit2Env()
}


def ignore_extra_args(func):
    def indifferent_func(**kwargs):
        signature = inspect.signature(func)
        expected_keys = [p.name for p in signature.parameters.values()
                         if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        filtered_kwargs = {k: kwargs[k] for k in kwargs if k in expected_keys}
        return func(**filtered_kwargs)
    return indifferent_func


AGENT_CONSTRUCTORS = {
    "reinforce": ignore_extra_args(ReinforceAgent),
    "value_baseline": ignore_extra_args(ValueBaselineAgent),
    "perfect_value_baseline": ignore_extra_args(PerfectValueBaselineAgent),
    "optimal_state_baseline" : ignore_extra_args(OptimalStateBaselineAgent),
    "state_action_baseline": ignore_extra_args(ActionStateBaselineAgent),
    "perfect_state_action_baseline": ignore_extra_args(PerfectActionStateBaselineAgent),
    "traj_cv": ignore_extra_args(TrajectoryCVAgent),
    "perfect_traj_cv": ignore_extra_args(PerfectTrajectoryCVAgent),
    "dynamics_traj_cv": ignore_extra_args(DynamicsTrajCVAgent),
    "perfect_dynamics_traj_cv": ignore_extra_args(PerfectDynamicsTrajCVAgent),
    "perfect_dynamics_est_QV_traj_cv": ignore_extra_args(PerfectDynamicsEstQVTrajCVAgent),
    "model_free_dynamics_traj_cv": ignore_extra_args(ModelFreeDynamicsTrajCVAgent),
    "perfect_model_free_dynamics_traj_cv": ignore_extra_args(PerfectModelFreeDynamicsTrajCVAgent),
    "random": ignore_extra_args(RandomAgent)
}

LR_SCHEDULERS = {
    "OneCycle": lrs.OneCycleLR
}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model_type", type=click.Choice(AGENT_CONSTRUCTORS.keys()), required=True)
@click.option("--env_type", type=click.Choice(ENV_CONSTRUCTORS.keys()), required=True)
@click.option("--env_slip_rate", type=float, default=0.1)
@click.option("--episodes", type=int, required=True)
@click.option("--epi_length", type=int, required=True)
@click.option("--eps_per_train", type=int, default=1)
@click.option("--lr_scheduler", type=click.Choice(LR_SCHEDULERS.keys()), default=None)
@click.option("--alpha", type=float, default=0.1)
@click.option("--beta", type=float, default=None)
@click.option("--delta", type=float, default=None)
@click.option("--gamma", type=float, default=0.99)
@click.option("--run_n_times", type=int, default=1)
@click.option("--log_freq", type=int, default=1)
@click.option("--project", type=str, default=None)
@click.option("--log_folder", type=str, default=None)
@click.option("--exp_name", type=str, default='experiment')
@click.option("--analytical", type=bool, default=False)
@click.option("--estimate_policy", type=bool, default=False)
@click.option("--show_policy", type=bool, default=False)
@click.option("--estimate_variance", type=bool, default=False)
def run(
    model_type,
    env_type,
    env_slip_rate,
    episodes,
    epi_length,
    eps_per_train,
    lr_scheduler,
    alpha,
    beta,
    delta,
    gamma,
    run_n_times,
    log_freq,
    project,
    log_folder,
    exp_name,
    analytical,
    estimate_policy,
    show_policy,
    estimate_variance
):
    for _ in range(run_n_times):
        config = {
            'model_type': model_type,
            'env_type': env_type,
            'episodes': episodes,
            'epi_length': epi_length,
            'alpha': alpha,
            'beta': beta,
            'delta': delta,
            'gamma': gamma,
            'eps_per_train': eps_per_train,
            #'git_sha': get_current_sha(),
            'analytical': analytical,
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

        if env_type not in ENV_CONSTRUCTORS:
            raise Exception("ENV NOT IMPLEMENTED")
        else:
            if env_type == "frozenlake":
                env = ENV_CONSTRUCTORS[env_type](env_slip_rate)
            else:
                env = ENV_CONSTRUCTORS[env_type]()

        env_shape = env.transitions.shape

        if model_type not in AGENT_CONSTRUCTORS:
            raise Exception("AGENT TYPE NOT IMPLEMENTED")
        else:
            if lr_scheduler is not None:
                scheduler = LR_SCHEDULERS[lr_scheduler]
            else:
                scheduler = None
            agent = AGENT_CONSTRUCTORS[model_type](env=env,
                                                   env_shape=env_shape,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   delta=delta,
                                                   gamma=gamma,
                                                   episode_length=epi_length,
                                                   analytical=analytical,
                                                   total_steps=episodes,
                                                   lr_scheduler=scheduler)

        train(agent, env, episodes, epi_length, eps_per_train, log_freq=log_freq,
              logger=logger_manager, estimate_policy=estimate_policy,
              analytical=analytical, estimate_variance=estimate_variance)

        if show_policy is True:
            if log_folder is not None:
                output_path = os.path.join(current_run_folder, 'policy.pdf')
                plot_policy(env, agent.pi.detach().numpy(), epi_length, output_path)
            else:
                print('Log folder must not be None in order to save policy plot.')

if __name__ == "__main__":
    cli()
