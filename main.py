from typing import Optional
from pathlib import Path
from os import path
from dotmap import DotMap
from helpers.board import TensorboardLogger

import typer
import yaml
import coloredlogs
import logging
import pathlib
import os
import time

logger = logging.getLogger(__name__)
coloredlogs.install(level="DEBUG")

default_output_dir = path.expanduser("~/.MARLIO-runner/")
if not os.path.exists(default_output_dir):
    os.makedirs(default_output_dir)
sample_conf = "./config.yml"


def main(
    config: Path = typer.Option(
        sample_conf,
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        resolve_path=True,
    ),
    verbose: bool = typer.Option(False, "--verbose/"),
    gui: bool = typer.Option(False, "--gui/--no-gui"),
    tensorboard: bool = typer.Option(False, "--tensorboard"),
    output_dir: Path = typer.Option(
        default_output_dir,
        exists=True,
        file_okay=False,
        dir_okay=True,
        writable=True,
        resolve_path=True,
    ),
    expname: str = typer.Option(None, prompt=True),
    train: bool = typer.Option(None, "--train/--test", prompt=True),
):

    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf = DotMap(conf)
    logger.info("Config Loaded")

    if conf.game_config.agents == 1:
        output_dir = os.path.join(output_dir, expname)
        os.makedirs(output_dir)
        single_agent(conf, verbose, gui, tensorboard, output_dir, train)
    elif conf.game_config.agents == 2:
        multi_agent(conf, verbose, gui, tensorboard, output_dir, train)
    else:
        typer.secho(
            "No. of agents should be either 1 or 2",
            fg=typer.colors.WHITE,
            bg=typer.colors.RED,
            err=True,
        )
        raise typer.Abort()

    return


def single_agent(config, verbose, gui, tensorboard, output_dir, train):
    if type(config.agents_config.agents) != str:
        typer.secho(
            "Agent definition error",
            fg=typer.colors.WHITE,
            bg=typer.colors.RED,
            err=True,
        )
        raise typer.Abort()

    if not path.isfile(f"./agents/agent_{config.agents_config.agents}.py"):
        typer.secho(
            f"Agent strategy file './agents/agent_{config.agents_config.agents}.py' not found",
            fg=typer.colors.WHITE,
            bg=typer.colors.RED,
            err=True,
        )
        raise typer.Abort()

    # Import Agent Strategy
    import importlib

    strategy = importlib.import_module(
        f"agents.agent_{config.agents_config.agents}")
    strategy = strategy.Strategy

    # Initialize Gym
    from helpers.utils import game_config_json
    config_json = game_config_json(config)

    import gym

    x = pathlib.Path(__file__).parent.absolute()
    x = str(x).replace(" ", "\\ ")
    env = gym.make("codeside:codeside-v0", config=f"{x}/config.json")
    time.sleep(2)

    # Initialize Agent Strategy
    agents_config = config.agents_config
    agent = strategy(env, agents_config, logger)

    # Initialize TensorBoard Logger
    tensorboard = TensorboardLogger(config, output_dir)
    replays = os.path.join(output_dir, "replays")
    results = os.path.join(output_dir, "results")
    models = os.path.join(output_dir, "models")
    os.makedirs(replays)
    os.makedirs(results)
    os.makedirs(models)
    # Start
    for episode in range(agents_config.episodes):
        # Spawn Our Player
        replay = os.path.join(replays, f"ep_{episode}")
        result = os.path.join(results, f"ep_{episode}")
        _ = env.reset(replay, result, config.game_config.batch_mode)
        player = env.create_player(port=config.game_config.port)
        cur_state = env.get_state(player)
        step = 0
        tot_reward = 0
        while True:
            action, discrete_action = agent.act(cur_state)
            logger.debug(action)
            logger.debug(discrete_action)
            new_state, reward, done, _ = env.step(player, action)
            if done:
                break

            agent.custom_logic(cur_state, discrete_action, reward,
                               new_state, done, step)
            tensorboard.log_step(episode, step, action, reward, new_state)

            cur_state = new_state
            tot_reward += reward
            step += 1
        tensorboard.log_episode(
            episode, step, tot_reward, done)  # add win state
        if episode % agents_config.save_every == 0:
            model = os.path.join(models, f"ep_{episode}.model")
            agent.save_model(model)

    return


def multi_agent(config, verbose, gui, tensorboard, output_dir, train):
    raise NotImplementedError


if __name__ == "__main__":
    typer.run(main)
