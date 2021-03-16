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

logger = logging.getLogger(__name__)
coloredlogs.install(level='DEBUG', logger=logger)

default_output_dir = path.expanduser("~/.MARLIO-runner/")
sample_conf = "./config.yml"


def main(config: Path = typer.Option(sample_conf, exists=True, file_okay=True, dir_okay=False, readable=True, resolve_path=True),
         verbose: bool = typer.Option(False, "--verbose/"),
         gui: bool = typer.Option(False, "--gui/--no-gui"),
         tensorboard: bool = typer.Option(False, "--tensorboard"),
         output_dir: Path = typer.Option(default_output_dir,
                                         exists=True, file_okay=False, dir_okay=True, writable=True, resolve_path=True),
         train: bool = typer.Option(None, "--train/--test", prompt=True)):

    with open(config) as f:
        conf = yaml.load(f, Loader=yaml.FullLoader)
        conf = DotMap(conf)
    logger.info("Config Loaded")

    if conf.game_config.agents == 1:
        single_agent(conf, verbose, gui, tensorboard, output_dir, train)
    elif conf.game_config.agents == 2:
        multi_agent(conf, verbose, gui, tensorboard, output_dir, train)
    else:
        typer.secho("No. of agents should be either 1 or 2",
                    fg=typer.colors.WHITE, bg=typer.colors.RED, err=True)
        raise typer.Abort()

    return


def single_agent(config, verbose, gui, tensorboard, output_dir, train):
    if type(config.agents_config.agents) != str:
        typer.secho("Agent definition error",
                    fg=typer.colors.WHITE, bg=typer.colors.RED, err=True)
        raise typer.Abort()

    if not path.isfile(f'./agents/agent_{config.agents_config.agents}.py'):
        typer.secho(f"Agent strategy file './agents/agent_{config.agents_config.agents}.py' not found",
                    fg=typer.colors.WHITE, bg=typer.colors.RED, err=True)
        raise typer.Abort()

    # Import Agent Strategy
    import importlib
    strategy = importlib.import_module(
        f'agents.agent_{config.agents_config.agents}')
    strategy = strategy.Strategy

    # Initialize Gym
    from helpers.utils import game_config_json
    config_json = game_config_json(config)

    import gym
    import time
    x = pathlib.Path(__file__).parent.absolute()
    x = str(x).replace(" ", "\\ ")
    env = gym.make("codeside:codeside-v0", config=f"{x}/config.json")
    time.sleep(2)

    # Spawn Our Player
    player = env.create_player(port=config.game_config.port)

    # Initialize Agent Strategy
    agents_config = config.agents_config
    agent = strategy(env, agents_config)

    # Initialize TensorBoard Logger
    tensorboard = TensorboardLogger(config)

    # Start
    for episode in range(agents_config.episodes):
        cur_state = env.reset()
        step = 0
        tot_reward = 0
        while True:
            action = agent.act(cur_state)
            new_state, reward, done, _ = env.step(action)

            agent.custom_logic(cur_state, action, reward,
                               new_state, done, episode)
            tensorboard.log_step(episode, step, action, reward, new_state)

            cur_state = new_state

            tot_reward += reward
            step += 1
            if done:
                break
        tensorboard.log_episode(
            episode, step, tot_reward, done)  # add win state
        if episode % agents_config.save_every == 0:
            agent.save_model(f"{agents_config.agents}_{episode}.model")

    return


def multi_agent(config, verbose, gui, tensorboard, output_dir, train):
    raise NotImplementedError


if __name__ == "__main__":
    typer.run(main)
