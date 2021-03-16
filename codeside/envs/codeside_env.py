import os
import copy
import math
import time
import json
import random
import signal
from pathlib import Path
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from .utils import get_logger, class_to_dict, calc_distance
from .agent.agent import Agent
from .agent.debug import Debug
from .agent.model import PlayerMessageGame, ServerMessageGame, \
    Versioned, UnitAction, Vec2Double
import threading


class CodeSideEnv(gym.Env):
    def __init__(self, config="./levels/config_simple.json", player1_name="Player1", player2_name="Player2"):
        self.logger = get_logger(__name__)
        self.prev_state = None
        self.prev_calc = {
            "distance": 10**9+7
        }
        self.steps = 0
        self.total_reward = 0
        self.game = None
        self.config = config
        self.player1_name = player1_name
        self.player2_name = player2_name
        self.reset()

    def reset(self):
        """
        Reset the environment with a particular configuration and
        initialize a thread which runs the game environment
        """
        self.close()
        self.logger.debug("Setting up environment")
        try:
            bin_path = Path(__file__).parent.absolute().as_posix()
            cmd = f"{bin_path}/../../bin/aicup2019 --config {self.config} " + \
                f"--player-names {self.player1_name} {self.player2_name}"
            self.logger.debug(cmd)
            self.game = threading.Thread(target=os.system, args=[cmd])
            self.game.setDaemon(True)
            self.game.start()
            self.logger.info("Environment successfully setup!")
        except Exception as err:
            self.logger.error(err)
        return

    def create_player(self, host="127.0.0.1", port=31000):
        """
        Create an agent object that connects to the server using sockets
        """
        agent = Agent(host, port)
        self.logger.info(f"Agent connected at port {port}")
        return agent

    def state_reducer(self, state):
        """
        Reduce the state dictionary
        """
        if "game" not in state:
            return state
        reduced_state = {
            "player_id": state.get("my_id", 0),
            "units": [],
            "opp_units": [],
            "mines": [],
            "bullets": [],
            "loot_boxes": [],
        }
        for player in state["game"].get("players", []):
            if player["id"] == reduced_state["player_id"]:
                reduced_state["player_score"] = player["score"]
            else:
                reduced_state["opp_score"] = player["score"]

        tile_range = 3
        level_size_x = len(state["game"]["level"]["tiles"][0])
        level_size_y = len(state["game"]["level"]["tiles"])

        for unit in state["game"].get("units", []):
            if unit.get("weapon") is None:
                del unit["weapon"]
            observation = [
                unit["health"],
                unit["position"]["x"],
                unit["position"]["y"],
                int(unit["jump_state"]["can_jump"]),
                int(unit["walked_right"]),
                int(unit["stand"]),
                int(unit["on_ground"]),
                int(unit["on_ladder"]),
                unit["mines"],
                unit.get("weapon", {}).get("typ", {}).get("value", -1),
                unit.get("weapon", {}).get("magazine", 0),
                int(unit.get("weapon", {}).get("was_shooting", False)),
                unit.get("weapon", {}).get("last_angle", 0),
            ]

            x = math.floor(observation[1])
            y = math.ceil(observation[2])
            tiles = []
            for yy in range(y-tile_range, y+tile_range+1):
                if yy < 0 or yy >= level_size_y:
                    tiles.append([0]*level_size_x)
                    continue
                tiles_x = []
                for xx in range(x-tile_range, x+tile_range+1):
                    if xx < 0 or xx >= level_size_x:
                        tiles_x.append(0)
                    else:
                        tiles_x.append(state["game"]["level"]
                                       ["tiles"][yy][xx]["value"])
                tiles.append(tiles_x)
            observation.append(tiles)

            if unit["player_id"] == reduced_state["player_id"]:
                reduced_state["units"].append(observation)
            else:
                reduced_state["opp_units"].append(observation)

        for loot in state["game"].get("loot_boxes", []):
            box = [
                loot["position"]["x"],
                loot["position"]["y"],
                0,
                0
            ]
            if loot["item"].get("weapon_type") is not None:
                box[2] = 1
                box[3] = loot["item"]["weapon_type"]["value"]
            elif loot["item"].get("health") is not None:
                box[2] = 2
                box[3] = loot["item"]["health"]
            reduced_state["loot_boxes"].append(box)

        for bullet in state["game"].get("bullets", []):
            bull = [
                bullet["position"]["x"],
                bullet["position"]["y"],
                bullet["velocity"]["x"],
                bullet["velocity"]["y"],
                bullet["damage"],
                bullet["size"]
            ]
            if bullet["explosion_params"] is None:
                bull.extend([0, 0])
            else:
                bull.extend([
                    bullet["explosion_params"].get("damage", 0),
                    bullet["explosion_params"].get("range", 0)
                ])
            reduced_state["bullets"].append(bull)

        for mine in state["game"].get("mines", []):
            minee = [
                mine["position"]["x"],
                mine["position"]["y"],
                mine["state"],
                mine["timer"],
                mine["trigger_radius"]
            ]
            if mine["explosion_params"] is None:
                minee.extend([0, 0])
            else:
                minee.extend([
                    mine["explosion_params"].get("damage", 0),
                    mine["explosion_params"].get("range", 0)
                ])
            reduced_state["mines"].append(minee)

        return reduced_state

    def get_state(self, agent):
        """
        Fetch the state observations from the environment
        """
        message = ServerMessageGame.read_from(agent.reader)
        if message.player_view is None:
            return None, None
        player_view = message.player_view
        state = dict(copy.deepcopy(player_view.__dict__))
        state = class_to_dict(state)
        reduced_state = self.state_reducer(state)
        return reduced_state, state, player_view

    def create_action(self, velocity, jump, jump_down, aim_x, aim_y, shoot, reload,
                      swap_weapon, plant_mine):
        """
        Returns a UnitAction object as required by the CodeSide env
        """
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
        return reward

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
        reduced_state, state, raw_state = self.get_state(agent)
        if state is not None:
            reward = self.get_reward(raw_state, self.prev_state)
            self.logger.debug(f"Reward: {reward}")
            self.prev_state = raw_state
            self.steps += 1
            return reduced_state, state, raw_state, reward, False
        return None, None, None, None, True

    def sample_space(self):
        """
        Return a sample observation space
        """
        with open("sample_space.json", "r") as f:
            space = json.load(f)
        return space

    def sample_action(self):
        """
        Return a sample action
        """
        action = {
            "aim": {
                "x": 37.0,
                "y": 0.0
            },
            "jump": false,
            "jump_down": true,
            "plant_mine": false,
            "reload": false,
            "shoot": true,
            "swap_weapon": false,
            "velocity": 37.0
        }
        return action

    def close(self):
        """
        Terminate the game thread
        """
        if self.game is not None and self.game.is_alive():
            self.logger.debug("Killing game thread")
            os.kill(self.game.native_id+2, signal.SIGKILL)
        return
