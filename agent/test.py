import gym
import time
from my_strategy import MyStrategy
from debug import Debug
from pprint import pprint
import json
import pickle


env = gym.make("codeside:codeside-v0")
time.sleep(2)
agent = env.create_agent(port=31000)
state, raw_state = env.get_state(agent)
strategy = MyStrategy()
debug = Debug(agent.writer)


def class_to_dict(state):
    if isinstance(state, list):
        return [class_to_dict(s) for s in state]
    if isinstance(state, dict):
        if "__objclass__" in state:
            return {
                "name": state["_name_"],
                "value": state["_value_"]
            }
        for prop in state:
            state[prop] = class_to_dict(state[prop])
        return state
    try:
        return class_to_dict(state.__dict__)
    except Exception:
        return state


while True:
    time.sleep(0.1)
    # pprint(state)
    x = input()
    with open("state.json", 'w') as f:
        json.dump(state, f)
    actions = {}
    for unit in raw_state.game.units:
        if unit.player_id == raw_state.my_id:
            actions[unit.id] = strategy.get_action(
                unit, raw_state.game, debug)
        pprint(actions)
    state, raw_state, reward = env.step(agent, actions)
