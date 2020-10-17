import gym
import time
from my_strategy import MyStrategy
from debug import Debug

env = gym.make("codeside:codeside-v0", config="config_simple.json")
time.sleep(2)
agent = env.create_agent(port=31000)
state = env.get_state(agent)
strategy = MyStrategy()
debug = Debug(agent.writer)

while True:
    time.sleep(1)
    actions = {}
    for unit in state.game.units:
        if unit.player_id == state.my_id:
            actions[unit.id] = strategy.get_action(
                unit, state.game, debug)
    env.step(agent, actions)
