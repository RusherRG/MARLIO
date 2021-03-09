import os
import copy
import time
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .utils import get_logger, class_to_dict, calc_distance
from .agent.agent import Agent
from .agent.debug import Debug
from .agent.model import PlayerMessageGame, ServerMessageGame, \
    Versioned, UnitAction, Vec2Double
import threading
import tensorflow as tf


class CodeSideEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(self, name="Helen", config="config_simple.json"):
        """
        Initialize a thread for the game environment
        """
        self.game = threading.Thread(target=self.reset, args=[config])
        self.game.setDaemon = True
        self.game.start()
        self.logger = get_logger(__name__)
        self.prev_state = None
        self.prev_calc = {
            "distance": 10**9+7
        }
        self.steps = 0
        self.total_reward = 0
        self.file_writers = {}

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

    def get_state(self, agent):
        """
        Fetch the state observations from the environment
        """
        message = ServerMessageGame.read_from(agent.reader)
        if message.player_view is None:
            return
        player_view = message.player_view
        state = dict(copy.deepcopy(player_view.__dict__))
        state = class_to_dict(state)
        # state = player_view
        return state, player_view

    def create_action(self, velocity, jump, jump_down, aim_x, aim_y, shoot, reload,
                      swap_weapon, plant_mine):

        self.tensorboard_log("Actions", shoot, "Shoot")
        self.tensorboard_log("Actions", swap_weapon, "Swap_Weapon")
        self.tensorboard_log("Actions", plant_mine, "Plant_Mine")

        return UnitAction(
            velocity=velocity,
            jump=jump,
            jump_down=jump_down,
            aim=Vec2Double(aim_x, aim_y),
            shoot=shoot,
            reload=reload,
            swap_weapon=swap_weapon,
            plant_mine=plant_mine
        )

    def get_reward(self, state, prev_state):
        """
        Calculate reward for the given state
        """
        reward = 0
        if prev_state is None:
            return reward
        for unit in state.game.units:
            if unit.player_id == state.my_id:
                for prev_state_unit in prev_state.game.units:
                    if prev_state_unit.id == unit.id:
                        # damage taken
                        if prev_state_unit.health > unit.health:
                            reward = -(prev_state_unit.health -
                                       unit.health)//10
                        # picked a new weapon
                        if prev_state_unit.weapon is None and \
                                unit.weapon is not None:
                            reward = 5
                        # picked a mine
                        if prev_state_unit.mines == 0 and unit.mines > 0:
                            reward = 5
                        break
            else:
                # moving towards the opponent
                new_distance = calc_distance(unit.position,
                                             prev_state_unit.position)
                if (unit.position.x > prev_state_unit.position.x) ^ \
                        unit.walked_right:
                    if self.prev_calc["distance"] - new_distance > 0:
                        reward = 1
                self.prev_calc["distance"] = new_distance

        self.total_reward += reward
        self.tensorboard_log("Total_Reward", self.total_reward, "total_reward")
        self.tensorboard_log("Reward", reward, 'reward')

        return reward

    def tensorboard_log(self, name, value, logdir):
        if self.file_writers.get(logdir, 0):
            summary_writer = self.file_writers[logdir]
        else:
            summary_writer = tf.summary.create_file_writer('logs/'+logdir)
            self.file_writers[logdir] = summary_writer
        with summary_writer.as_default():
            tf.summary.scalar(name, value, step=self.steps)

    def step(self, agent, actions):
        """
        Perform a step for given actions and return the current state
        observations of the game environment
        """
        # Convert actions to UnitAction objects
        # for unit_id in actions:
        #     actions[unit_id] = self.create_action(**actions[unit_id])

        # Perform the action
        PlayerMessageGame.ActionMessage(
            Versioned(actions)).write_to(agent.writer)
        agent.writer.flush()

        # Get state observations
        state, raw_state = self.get_state(agent)
        reward = self.get_reward(raw_state, self.prev_state)
        self.logger.info("Reward: {}".format(reward))
        self.prev_state = raw_state
        self.steps += 1
        return state, raw_state, reward

    def close(self):
        """
        Terminate the game thread
        """
        ...
