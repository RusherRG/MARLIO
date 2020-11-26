import gym
import time
from my_strategy import MyStrategy
from debug import Debug
from pprint import pprint
import json

env = gym.make("codeside:codeside-v0", config="config_simple.json")
time.sleep(2)
agent = env.create_agent(port=31000)
state, raw_state = env.get_state(agent)
strategy = MyStrategy()
debug = Debug(agent.writer)


while True:
    t = input()
    # print("\nState")
    # pprint(state)
    actions = {}
    for unit in raw_state.game.units:
        if unit.player_id == raw_state.my_id:
            actions[unit.id] = strategy.get_action(
                unit, raw_state.game, debug)
    # print("\nAction")
    # state["actions"] = {
    #     key: actions[key].__dict__ for key in actions
    # }
    state, raw_state, reward = env.step(agent, actions)
