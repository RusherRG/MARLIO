import os
import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .logger import get_logger
from .agent.agent import Agent
from .agent.debug import Debug
from .agent.model import PlayerMessageGame, ServerMessageGame, Versioned
import threading


class CodeSideEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, config="config_test.json"):
        """
        Initialize a thread for the game environment
        """
        self.game = threading.Thread(target=self.reset, args=[config])
        self.game.setDaemon = True
        self.game.start()
        self.logger = get_logger(__name__)

    def step(self, agent, actions):
        """
        Perform a step for given actions and return the current state
        observations of the game environment
        """
        # Perform the action
        PlayerMessageGame.ActionMessage(
            Versioned(actions)).write_to(agent.writer)
        agent.writer.flush()

        # Get state observations
        return self.get_state(agent)

    def get_state(self, agent):
        """
        Fetch the state observations from the environment
        """
        message = ServerMessageGame.read_from(agent.reader)
        if message.player_view is None:
            return
        player_view = message.player_view
        return player_view

    def reset(self, config, player1_name="Agent", player2_name="Computer"):
        """
        Reset the environment with a particular configuration
        """
        try:
            cmd = "../bin/aicup2019 --config ./levels/{} ".format(config) + \
                "--player-names {} {}".format(player1_name, player2_name)
            os.system(cmd)
            self.logger.info("Environment successfully setup!")
        except Exception as err:
            self.logger.error(err)
        return

    def create_agent(self, host="127.0.0.1", port=31000):
        """
        Create an agent object that connects to the server using sockets
        """
        agent = Agent(host, port)
        return agent

    def close(self):
        """
        Terminate the game thread
        """
        ...
