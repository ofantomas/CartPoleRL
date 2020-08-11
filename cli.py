import click
from agents import ReinforceAgent, RandomAgent, \
    CreditBaselineAgent, ValueBaselineAgent, OneStepCreditWithValueAgent, QTargetAgent, QAdvantageAgent, \
    HCAStateConditionalQAgent, MultiStepCreditAgent, RandomMultWithValueAgent, CreditBaselineMixtureAgent, MICAAgent, \
    MICAValueAgent, ValueModAgent, MICAMixtureAgent, CreditBaselineMixtureCounterfactualAgent, \
    CreditBaselineCounterfactualAgent, MICACounterfactualAgent, MICAMixtureCounterfactualAgent, \
    MixtureVBaselineAgent
from envs import TestEnv, SmallGridEnv, SmallGridExtraActionsEnv, SmallGridNoNoOpEnv, \
    SmallGridNotDoneEnv, ShortcutEnv, DelayedEffectEnv, AmbiguousBanditEnv, FrozenLakeEnv, CounterexampleBanditEnv, \
    CounterexampleBandit2Env
from train import train

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


AGENT_CONSTRUCTORS = {
    "reinforce": lambda env_shape, alpha, gamma, possible_r, epi_length:
        ReinforceAgent(env_shape, alpha, gamma),
    "value_baseline": lambda env_shape, alpha, gamma, possible_r, epi_length:
        ValueBaselineAgent(env_shape, alpha, gamma),
    # "value_baseline_+1": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     ValueModAgent(env_shape, alpha, gamma, value_mult=-1),
    # "value_baseline_+2": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     ValueModAgent(env_shape, alpha, gamma, value_mult=-2),
    # "value_baseline_+3": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     ValueModAgent(env_shape, alpha, gamma, value_mult=-3),
    # "value_baseline_+5": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     ValueModAgent(env_shape, alpha, gamma, value_mult=-5),
    # "value_baseline_+10": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     ValueModAgent(env_shape, alpha, gamma, value_mult=-10),
    # "value_baseline_-2": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     ValueModAgent(env_shape, alpha, gamma, value_mult=2),
    # "value_baseline_0": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     ValueModAgent(env_shape, alpha, gamma, value_mult=0),
    # "q_target": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     QTargetAgent(env_shape, alpha, gamma),
    # "q_value_advantage": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     QAdvantageAgent(env_shape, alpha, gamma),
    "credit_baseline": lambda env_shape, alpha, gamma, possible_r, epi_length:
        CreditBaselineAgent(env_shape, alpha, gamma, possible_r),
    # "credit_baseline_cf": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     CreditBaselineCounterfactualAgent(env_shape, alpha, gamma, possible_r),
    # "credit_baseline_mixture_50": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     CreditBaselineMixtureAgent(env_shape, alpha, gamma, possible_r, mix_ratio=0.5),
    # "credit_baseline_mixture_25": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     CreditBaselineMixtureAgent(env_shape, alpha, gamma, possible_r, mix_ratio=0.25),
    # "credit_baseline_mixture_75": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     CreditBaselineMixtureAgent(env_shape, alpha, gamma, possible_r, mix_ratio=0.75),
    # "credit_baseline_mixture_50_cf": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     CreditBaselineMixtureCounterfactualAgent(env_shape, alpha, gamma, possible_r, mix_ratio=0.5),
    # 'mixture_v': lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MixtureVBaselineAgent(env_shape, alpha, gamma, possible_r, mix_ratio=0.5),
    # "mica": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MICAAgent(env_shape, alpha, gamma, possible_r),
    # "mica_cf": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MICACounterfactualAgent(env_shape, alpha, gamma, possible_r),
    # "mica_mixture_50": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MICAMixtureAgent(env_shape, alpha, gamma, possible_r, mix_ratio=0.5),
    # "mica_mixture_50_cf": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MICAMixtureCounterfactualAgent(env_shape, alpha, gamma, possible_r, mix_ratio=0.5),
    # "mica_value": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MICAValueAgent(env_shape, alpha, gamma, possible_r),
    # "credit_baseline_old": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     OneStepCreditWithValueAgent(env_shape, alpha, gamma, possible_r, credit_type="baseline"),
    # "credit_random_value": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     RandomMultWithValueAgent(env_shape, alpha, gamma, possible_r),
    # "mica_multistep": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MultiStepCreditAgent(env_shape, alpha, gamma, possible_r, epi_length, credit_type="mica"),
    # "credit_baseline_multistep": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     MultiStepCreditAgent(env_shape, alpha, gamma, possible_r, epi_length, credit_type="baseline"),
    # "hca_q_states": lambda env_shape, alpha, gamma, possible_r, epi_length:
    #     HCAStateConditionalQAgent(env_shape, alpha, gamma, epi_length),
    "random": lambda env_shape, alpha, gamma, possible_r, epi_length:
        RandomAgent(env_shape)
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
@click.option("--beta", type=float, default=0.1)
@click.option("--gamma", type=float, default=0.99)
@click.option("--run_n_times", type=int, default=1)
@click.option("--log_freq", type=int, default=1)
@click.option("--project", type=str, default=None)
@click.option("--exp_name", type=str, default=None)
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
    exp_name
):
    for run_n in range(run_n_times):

        if project is not None:
            import wandb
            wandb.init(project=project, reinit=True, name=exp_name)
            wandb_handle = wandb.log

            setattr(wandb.config, 'model_type', model_type)
            setattr(wandb.config, 'env_type', env_type)
            setattr(wandb.config, 'episodes', episodes)
            setattr(wandb.config, 'epi_length', epi_length)
            setattr(wandb.config, 'alpha', alpha)
            setattr(wandb.config, 'beta', beta)
            setattr(wandb.config, 'gamma', gamma)
            setattr(wandb.config, 'eps_per_train', eps_per_train)
        else:
            wandb_handle = print

        if env_type not in ENV_CONSTRUCTORS:
            raise Exception("ENV NOT IMPLEMENTED")
        else:
            env = ENV_CONSTRUCTORS[env_type]()

        env_shape = env.transitions.shape

        if model_type not in AGENT_CONSTRUCTORS:
            raise Exception("AGENT TYPE NOT IMPLEMENTED")
        else:
            agent = AGENT_CONSTRUCTORS[model_type](env_shape, alpha, gamma, env.possible_r_values, epi_length)

        train_rs, val_rs, visited, losses, loss_vars = train(
            agent, env, episodes, epi_length, eps_per_train, log_freq=log_freq, wandb_handle=wandb_handle,
        )


if __name__ == "__main__":
    cli()
