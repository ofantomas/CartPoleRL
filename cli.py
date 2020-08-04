# Core setup and run for CCA tabular experiments

import os
import pathlib
import time

import click
import matplotlib

from .display import plot_training_runs
from .agents import ReinforceAgent, RandomAgent, \
    CreditBaselineAgent, ValueBaselineAgent, OneStepCreditWithValueAgent, QTargetAgent, QAdvantageAgent, \
    HCAStateConditionalQAgent, MultiStepCreditAgent, RandomMultWithValueAgent, CreditBaselineMixtureAgent, MICAAgent, \
    MICAValueAgent, ValueModAgent, MICAMixtureAgent, CreditBaselineMixtureCounterfactualAgent, \
    CreditBaselineCounterfactualAgent, MICACounterfactualAgent, MICAMixtureCounterfactualAgent
from .envs import TestEnv, SmallGridEnv, SmallGridExtraActionsEnv, SmallGridNoNoOpEnv, \
    SmallGridNotDoneEnv, ShortcutEnv, DelayedEffectEnv, AmbiguousBanditEnv, FrozenLakeEnv, CounterexampleBanditEnv, \
    CounterexampleBandit2Env
from .train import train

MODEL_OPTIONS = {
    "reinforce",
    "value_baseline",
    "value_baseline_+1",
    "value_baseline_-2",
    "value_baseline_+2",
    "value_baseline_+3",
    "value_baseline_+5",
    "value_baseline_+10",
    "value_baseline_0",
    "q_target",
    "q_value_advantage",
    "credit_baseline",
    "credit_baseline_cf",
    "credit_baseline_mixture_50",
    "credit_baseline_mixture_25",
    "credit_baseline_mixture_75",
    "credit_baseline_mixture_50_cf",
    "mica",
    "mica_cf",
    "mica_mixture_50",
    "mica_mixture_50_cf",
    "mica_value",
    "mica_multistep",
    "credit_baseline_multistep",
    "credit_random_value",
    "hca_q_states",
    "credit_baseline_old"
}

ENV_OPTIONS = {
    "test",
    "small_grid",
    "small_grid_extra_actions",
    "small_grid_no_no_op",
    "small_grid_not_done",
    "shortcut_hca",
    "delayed_hca",
    "delayed_hca_noisy",
    "delayed_10step",
    "delayed_100step",
    "delayed_100step_noisy",
    "ambiguous_hca",
    "frozenlake",
    "counterexample_bandit",
    "counterexample_bandit_2"
}

@click.group()
def cli():
    pass

@cli.command()
@click.option("--model_type", type=click.Choice(MODEL_OPTIONS), required=True)
@click.option("--env_type", type=click.Choice(ENV_OPTIONS), required=True)
@click.option("--episodes", type=int, required=True)
@click.option("--epi_length", type=int, required=True)
@click.option("--eps_per_train", type=int, default=1)
@click.option("--alpha", type=float, default=0.1)
@click.option("--gamma", type=float, default=0.99)
@click.option("--run_n_times", type=int, default=1)
@click.option("--log_freq", type=int, default=1)
@click.option("--project", type=str, default=None)
def run(
    model_type,
    env_type,
    episodes,
    epi_length,
    eps_per_train,
    alpha,
    gamma,
    run_n_times,
    log_freq,
    project,
):
    for run_n in range(run_n_times):

        if project is not None:
            import wandb
            wandb.init(project=project, reinit=True)
            wandb_handle = wandb.log

            setattr(wandb.config, 'model_type', model_type)
            setattr(wandb.config, 'env_type', env_type)
            setattr(wandb.config, 'episodes', episodes)
            setattr(wandb.config, 'epi_length', epi_length)
            setattr(wandb.config, 'alpha', alpha)
            setattr(wandb.config, 'gamma', gamma)
            setattr(wandb.config, 'eps_per_train', eps_per_train)
        else:
            wandb_handle = print

        # Will this help?
        # matplotlib.use('Qt5Agg')

        # Create an env.
        if env_type == "test":
            env = TestEnv()
        elif env_type == "small_grid":
            env = SmallGridEnv()
        elif env_type == "small_grid_extra_actions":
            env = SmallGridExtraActionsEnv()
        elif env_type == "small_grid_no_no_op":
            env = SmallGridNoNoOpEnv()
        elif env_type == "small_grid_not_done":
            env = SmallGridNotDoneEnv()
        elif env_type == "shortcut_hca":
            env = ShortcutEnv()
        elif env_type == "delayed_hca":
            env = DelayedEffectEnv()
        elif env_type == "delayed_hca_noisy":
            env = DelayedEffectEnv(3, 1)
        elif env_type == "delayed_10step":
            env = DelayedEffectEnv(8)
        elif env_type == "delayed_100step":
            env = DelayedEffectEnv(98)
        elif env_type == "delayed_100step_noisy":
            env = DelayedEffectEnv(98,0.1)
        elif env_type == "ambiguous_hca":
            env = AmbiguousBanditEnv()
        elif env_type == "frozenlake":
            env = FrozenLakeEnv()
        elif env_type == "counterexample_bandit":
            env = CounterexampleBanditEnv()
        elif env_type == "counterexample_bandit_2":
            env = CounterexampleBandit2Env()
        else:
            print("ENV NOT IMPLEMENTED")
            exit()

        env_shape = env.transitions

        if model_type == "reinforce":
            agent = ReinforceAgent(env_shape, alpha, gamma)
        elif model_type == "value_baseline":
                agent = ValueBaselineAgent(env_shape, alpha, gamma)
        # Value baseline ablations, multiply the value baseline by some amount. These should be bad.
        elif model_type == "value_baseline_+1":
                agent = ValueModAgent(env_shape, alpha, gamma, value_mult=-1)
        elif model_type == "value_baseline_+2":
                agent = ValueModAgent(env_shape, alpha, gamma, value_mult=-2)
        elif model_type == "value_baseline_+3":
                agent = ValueModAgent(env_shape, alpha, gamma, value_mult=-3)
        elif model_type == "value_baseline_+5":
                agent = ValueModAgent(env_shape, alpha, gamma, value_mult=-5)
        elif model_type == "value_baseline_+10":
                agent = ValueModAgent(env_shape, alpha, gamma, value_mult=-10)
        elif model_type == "value_baseline_-2":
                agent = ValueModAgent(env_shape, alpha, gamma, value_mult=2)
        elif model_type == "value_baseline_0":
                agent = ValueModAgent(env_shape, alpha, gamma, value_mult=0)

        elif model_type == "q_target":
                agent = QTargetAgent(env_shape, alpha, gamma)
        elif model_type == "q_value_advantage":
                agent = QAdvantageAgent(env_shape, alpha, gamma)

        elif model_type == "credit_baseline":
            agent = CreditBaselineAgent(env_shape, alpha, gamma, env.possible_r_values)
        elif model_type == "credit_baseline_cf":
            agent = CreditBaselineCounterfactualAgent(env_shape, alpha, gamma, env.possible_r_values)
        elif model_type == "credit_baseline_mixture_50":
            agent = CreditBaselineMixtureAgent(env_shape, alpha, gamma, env.possible_r_values, mix_ratio=0.5)
        elif model_type == "credit_baseline_mixture_25":
            agent = CreditBaselineMixtureAgent(env_shape, alpha, gamma, env.possible_r_values, mix_ratio=0.25)
        elif model_type == "credit_baseline_mixture_75":
            agent = CreditBaselineMixtureAgent(env_shape, alpha, gamma, env.possible_r_values, mix_ratio=0.75)
        elif model_type == "credit_baseline_mixture_50_cf":
            agent = CreditBaselineMixtureCounterfactualAgent(env_shape, alpha, gamma, env.possible_r_values, mix_ratio=0.5)

        elif model_type == "mica":
            agent = MICAAgent(env_shape, alpha, gamma, env.possible_r_values)
        elif model_type == "mica_cf":
            agent = MICACounterfactualAgent(env_shape, alpha, gamma, env.possible_r_values)
        elif model_type == "mica_mixture_50":
            agent = MICAMixtureAgent(env_shape, alpha, gamma, env.possible_r_values, mix_ratio=0.5)
        elif model_type == "mica_mixture_50_cf":
            agent = MICAMixtureCounterfactualAgent(env_shape, alpha, gamma, env.possible_r_values, mix_ratio=0.5)
        elif model_type == "mica_value":
            agent = MICAValueAgent(env_shape, alpha, gamma, env.possible_r_values)


        elif model_type == "credit_baseline_old":
            agent = OneStepCreditWithValueAgent(env_shape, alpha, gamma, env.possible_r_values, credit_type="baseline")
        elif model_type == "credit_random_value":
                agent = RandomMultWithValueAgent(env_shape, alpha, gamma, env.possible_r_values)
        elif model_type == "mica_multistep":
                agent = MultiStepCreditAgent(env_shape, alpha, gamma, env.possible_r_values, epi_length, credit_type="mica")
        elif model_type == "credit_baseline_multistep":
                agent = MultiStepCreditAgent(env_shape, alpha, gamma, env.possible_r_values, epi_length, credit_type="baseline")
        elif model_type == "hca_q_states":
                agent = HCAStateConditionalQAgent(env_shape, alpha, gamma, epi_length)
        elif model_type == "random":
            agent = RandomAgent(env_shape)
        else:
            print("AGENT TYPE NOT IMPLEMENTED")
            exit()

        train_rs, val_rs, visited, losses, loss_vars = train(
            agent, env, episodes, epi_length, eps_per_train, log_freq=log_freq, wandb_handle=wandb_handle,
        )


if __name__ == "__main__":
    cli()
