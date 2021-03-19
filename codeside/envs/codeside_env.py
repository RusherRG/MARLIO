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
        self.prev_prev_state = None
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

    def reset(self, replay_path=None, result_path=None, batch_mode=True):
        """
        Reset the environment with a particular configuration and
        initialize a thread which runs the game environment
        """
        if not batch_mode:
            self.close()
        self.logger.debug("Setting up environment")
        self.prev_state = None
        try:
            bin_path = Path(__file__).parent.absolute().as_posix()
            bin_path = bin_path.replace(" ", "\\ ")
            cmd = f"/home/rusherrg/Projects/raic-2019/app-src/target/debug/aicup2019 --config {self.config}" + \
                f" --player-names {self.player1_name} {self.player2_name}" + \
                f" --save-replay {replay_path}" + \
                f" --save-results {result_path}"
            if batch_mode:
                cmd += " --batch-mode"
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
        level_size_x = len(state["game"]["level"]["tiles"])
        level_size_y = len(state["game"]["level"]["tiles"][0])

        for unit in state["game"].get("units", []):
            if unit.get("weapon") is None:
                del unit["weapon"]
            observation = [
                unit["id"],
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

            x = math.floor(observation[2])
            y = math.ceil(observation[3])
            tiles = []
            for xx in range(x-tile_range, x+tile_range+1):
                if xx < 0 or xx >= level_size_x:
                    tiles.append([0]*(2*tile_range+1))
                    continue
                tiles_y = []
                for yy in range(y-tile_range, y+tile_range+1):
                    if yy < 0 or yy >= level_size_y:
                        tiles_y.append(0)
                    else:
                        tiles_y.append(state["game"]["level"]["tiles"]
                                       [xx][yy]["value"])
                tiles.append(tiles_y)
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
                mine["state"]["value"],
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
            return None, None, None
        player_view = message.player_view
        state = dict(copy.deepcopy(player_view.__dict__))
        state = class_to_dict(state)
        reduced_state = self.state_reducer(state)
        return [reduced_state, state, player_view]

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
            return 0

        # last_reward
        if state is None:
            my_score = 0
            opp_score = 0
            for player in prev_state.game.players:
                if player.id == prev_state.my_id:
                    my_score = player.score
                else:
                    opp_score = player.score

            score_reward = 100 if my_score > opp_score else -100

            for prev_state_unit in self.prev_prev_state.game.units:
                for unit in prev_state.game.units:
                    if prev_state_unit.id == unit.id:
                        break
                else:
                    if prev_state_unit.player_id == prev_state.my_id:
                        reward -= prev_state_unit.health
                    else:
                        reward += prev_state_unit.health

            return score_reward + reward

        # shooting in correct direction
        

        for unit in state.game.units:
            if unit.player_id == state.my_id:
                for prev_state_unit in prev_state.game.units:
                    if prev_state_unit.id == unit.id:
                        # damage taken
                        if prev_state_unit.health > unit.health:
                            reward += -(prev_state_unit.health -
                                        unit.health)
                        else:
                            reward += (unit.health - prev_state_unit.health)
                        # picked a new weapon
                        if prev_state_unit.weapon is None and \
                                unit.weapon is not None:
                            reward += 5
                        # picked a mine
                        if prev_state_unit.mines == 0 and unit.mines > 0:
                            reward += 5
                        break
            else:
                for prev_state_unit in prev_state.game.units:
                    if prev_state_unit.id == unit.id:
                        # damage done
                        if prev_state_unit.health > unit.health:
                            # damage by bullet
                            prev_count = 0
                            for prev_bullet in prev_state.game.bullets:
                                if prev_bullet.unit_id != unit.id:
                                    prev_count += 1
                            for bullet in state.game.bullets:
                                if bullet.unit_id != unit.id:
                                    prev_count -= 1
                            if prev_count > 0:
                                reward += (prev_state_unit.health -
                                           unit.health)
                            # damage by mine
                            prev_count = 0
                            for prev_mine in prev_state.game.mines:
                                if prev_mine.player_id != unit.player_id:
                                    prev_count += 1
                            for mine in state.game.mines:
                                if mine.player_id != unit.player_id:
                                    prev_count -= 1
                            if prev_count > 0:
                                reward += (prev_state_unit.health -
                                           unit.health)
        self.total_reward += reward
        return reward

    def step(self, agent, actions):
        """
        Perform a step for given actions and return the current state
        observations of the game environment
        """
        # Convert actions to UnitAction objects
        for unit_id in actions:
            actions[unit_id] = self.create_action(**actions[unit_id])

        # Perform the action
        PlayerMessageGame.ActionMessage(
            Versioned(actions)).write_to(agent.writer)
        agent.writer.flush()

        # Get state observations
        reduced_state, state, player_view = self.get_state(agent)
        reward = self.get_reward(player_view, self.prev_state)
        self.logger.debug(f"Reward: {reward}")
        self.prev_prev_state = self.prev_state
        self.prev_state = player_view
        done = state is None
        self.steps += 1
        return [reduced_state, state, player_view], reward, done, None

    def sample_space(self):
        """
        Return a sample observation space
        """
        with open("sample_space.json", "r") as f:
            space = json.load(f)
        return space

    def get_action(self, discrete_action):
        actions = {}
        aim_argmax = discrete_action[0]
        deg = ((aim_argmax*30)/180)*math.pi
        aim_x = math.cos(deg) * 30
        aim_y = math.sin(deg) * 30

        # get velocity
        velocity_argmax = discrete_action[1]
        velocity = (velocity_argmax - 2) * 25

        # get action
        action_argmax = discrete_action[2]
        shoot = action_argmax == 0
        reload = action_argmax == 1
        swap_weapon = action_argmax == 2
        plant_mine = action_argmax == 3

        # get jump
        jump = bool(discrete_action[3])

        # get jump_down
        jump_down = bool(discrete_action[4]) and not jump

        actions = {
            "aim_x": aim_x,
            "aim_y": aim_y,
            "velocity": velocity,
            "jump": jump,
            "jump_down": jump_down,
            "shoot": shoot,
            "reload": reload,
            "swap_weapon": swap_weapon,
            "plant_mine": plant_mine
        }
        return actions

    def sample_action(self):
        """
        Return a sample action
        """
        discrete_action = [
            random.randint(0, 11),
            random.randint(0, 4),
            random.randint(0, 3),
            random.randint(0, 1),
            random.randint(0, 1)
        ]
        action = self.get_action(discrete_action)
        return action, discrete_action

    def close(self):
        """
        Terminate the game thread
        """
        if self.game is not None and self.game.is_alive():
            self.logger.debug("Killing game thread")
            os.kill(self.game.native_id+2, signal.SIGKILL)
        return
