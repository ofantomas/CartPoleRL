from functools import partial
import inspect

import click
# Core setup and run for CCA tabular experiments
#from git import Repo

from agents import ReinforceAgent, RandomAgent, \
                   ValueBaselineAgent, PerfectValueBaselineAgent, \
                   OptimalStateBaselineAgent, ActionStateBaselineAgent,\
                   PerfectActionStateBaselineAgent    
    
from envs import TestEnv, SmallGridEnv, SmallGridExtraActionsEnv, SmallGridNoNoOpEnv, \
    SmallGridNotDoneEnv, ShortcutEnv, DelayedEffectEnv, AmbiguousBanditEnv, FrozenLakeEnv, CounterexampleBanditEnv, \
    CounterexampleBandit2Env
from logger import LoggingManager, WandbLogger, LocalLogger
from train import train


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
    "frozenlake": lambda: FrozenLakeEnv(),
    "counterexample_bandit": lambda: CounterexampleBanditEnv(),
    "counterexample_bandit_2": lambda: CounterexampleBandit2Env()
}


def ignore_extra_args(foo):
    def indifferent_foo(**kwargs):
        signature = inspect.signature(foo)
        expected_keys = [p.name for p in signature.parameters.values()
                         if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD]
        filtered_kwargs = {k: kwargs[k] for k in kwargs if k in expected_keys}
        return foo(**filtered_kwargs)
    return indifferent_foo


AGENT_CONSTRUCTORS = {
    "reinforce": ignore_extra_args(ReinforceAgent),
    "value_baseline": ignore_extra_args(ValueBaselineAgent),
    "perfect_value_baseline": ignore_extra_args(PerfectValueBaselineAgent),
    "optimal_state_baseline" : ignore_extra_args(OptimalStateBaselineAgent),
    "state_action_baseline": ignore_extra_args(ActionStateBaselineAgent),
    "perfect_state_action_baseline": ignore_extra_args(PerfectActionStateBaselineAgent),
    "random": ignore_extra_args(RandomAgent)
}


@click.group()
def cli():
    pass


@cli.command()
@click.option("--model_type", type=click.Choice(AGENT_CONSTRUCTORS.keys()), required=True)
@click.option("--env_type", type=click.Choice(ENV_CONSTRUCTORS.keys()), required=True)
@click.option("--episodes", type=int, required=True)
@click.option("--epi_length", type=int, required=True)
@click.option("--eps_per_train", type=int, default=1)
@click.option("--alpha", type=float, default=0.1)
@click.option("--beta", type=float, default=None)
@click.option("--gamma", type=float, default=0.99)
@click.option("--run_n_times", type=int, default=1)
@click.option("--log_freq", type=int, default=1)
@click.option("--project", type=str, default=None)
@click.option("--log_folder", type=str, default=None)
@click.option("--exp_name", type=str, default='experiment')
@click.option("--h_analytical", type=bool, default=False)
def run(
    model_type,
    env_type,
    episodes,
    epi_length,
    eps_per_train,
    alpha,
    beta,
    gamma,
    run_n_times,
    log_freq,
    project,
    log_folder,
    exp_name,
    h_analytical,
):
    for _ in range(run_n_times):
        config = {
            'model_type': model_type,
            'env_type': env_type,
            'episodes': episodes,
            'epi_length': epi_length,
            'alpha': alpha,
            'beta': beta,
            'gamma': gamma,
            'eps_per_train': eps_per_train,
            #'git_sha': get_current_sha(),
            'h_analytical': h_analytical,
            'exp_name': exp_name,
        }
        logger_manager = LoggingManager()
        if project is not None:
            logger_manager.add_logger(WandbLogger(project, exp_name))
        if log_folder is not None:
            logger_manager.add_logger(LocalLogger(log_folder, exp_name))
        logger_manager.log_config(config)

        if env_type not in ENV_CONSTRUCTORS:
            raise Exception("ENV NOT IMPLEMENTED")
        else:
            env = ENV_CONSTRUCTORS[env_type]()

        env_shape = env.transitions.shape

        if model_type not in AGENT_CONSTRUCTORS:
            raise Exception("AGENT TYPE NOT IMPLEMENTED")
        else:
            agent = AGENT_CONSTRUCTORS[model_type](env=env,
                                                   env_shape=env_shape,
                                                   alpha=alpha,
                                                   beta=beta,
                                                   gamma=gamma,
                                                   possible_rs=env.possible_r_values,
                                                   episode_length=epi_length,
                                                   h_analytical=h_analytical)

        train(agent, env, episodes, epi_length, eps_per_train, log_freq=log_freq, logger=logger_manager,)


if __name__ == "__main__":
    cli()
