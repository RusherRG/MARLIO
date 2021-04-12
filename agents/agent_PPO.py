from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, \
    Reshape, Concatenate
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras.optimizers import Adam
import math
import random
import numpy as np
import tensorflow.keras.backend as K
from tensorflow.python.framework.ops import disable_eager_execution
disable_eager_execution()


clipping_val = 0.2
critic_discount = 0.5
entropy_beta = 0.001
gamma = 0.99
lmbda = 0.95
dummy_12 = np.zeros((1, 1, 12))
dummy_5 = np.zeros((1, 1, 5))
dummy_4 = np.zeros((1, 1, 4))
dummy_2 = np.zeros((1, 1, 2))
dummy_1 = np.zeros((1, 1, 1))


class Strategy:
    def __init__(self, env, config, logger, logdir):
        self.env = env
        self.config = config
        self.logger = logger

        self.frame_count = 0
        self.epoch_count = 0
        self.memory = deque(maxlen=10000)

        self.batch_size = self.config.minibatch_size
        self.train_every = self.config.train_every

        self.actor = self.create_actor()
        self.critic = self.create_critic()

    def create_actor(self):
        tiles = Input(shape=(7, 7), name="tiles")
        tiles_out = Flatten()(tiles)
        tiles_out = Embedding(5, 4)(tiles_out)
        tiles_out = Reshape(target_shape=(196, ))(tiles_out)

        state = Input(shape=(110,), name="state")

        # different inputs for each of the oldpolicy output
        oldpolicy_probs_aim = Input(shape=(1, 12,), name="oldpolicy_probs_aim")
        oldpolicy_probs_velocity = Input(
            shape=(1, 5,), name="oldpolicy_probs_velocity")
        oldpolicy_probs_action = Input(
            shape=(1, 4,), name="oldpolicy_probs_action")
        oldpolicy_probs_jump = Input(
            shape=(1, 2,), name="oldpolicy_probs_jump")
        oldpolicy_probs_jump_down = Input(
            shape=(1, 2,), name="oldpolicy_probs_jump_down")
        # needed for PPO loss calculation
        advantages = Input(shape=(1, 1,), name="advantages")
        rewards = Input(shape=(1, 1,), name="rewards")
        values = Input(shape=(1, 1,), name="values")

        concat = Concatenate()([state, tiles_out])

        x = Dense(256, activation="relu")(concat)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        aim = self.build_branch(x, 12, "aim", "softmax")
        velocity = self.build_branch(x, 5, "velocity", "softmax")
        action = self.build_branch(x, 4, "action", "softmax")
        jump = self.build_branch(x, 2, "jump", "softmax")
        jump_down = self.build_branch(x, 2, "jump_down", "softmax")

        model = Model(inputs=[state, tiles, oldpolicy_probs_aim,
                              oldpolicy_probs_velocity, oldpolicy_probs_action,
                              oldpolicy_probs_jump, oldpolicy_probs_jump_down,
                              advantages, rewards, values],
                      outputs=[aim, velocity, action, jump, jump_down])
        # different loss for each output
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss=[self.ppo_loss(
                          oldpolicy_probs=op,
                          advantages=advantages,
                          rewards=rewards,
                          values=values)
                          for op in [oldpolicy_probs_aim,
                                     oldpolicy_probs_velocity,
                                     oldpolicy_probs_action,
                                     oldpolicy_probs_jump,
                                     oldpolicy_probs_jump_down]])

        self.logger.info(model.summary())
        return model

    def create_critic(self):
        tiles = Input(shape=(7, 7), name="tiles")
        tiles_out = Flatten()(tiles)
        tiles_out = Embedding(5, 4)(tiles_out)
        tiles_out = Reshape(target_shape=(196, ))(tiles_out)

        state = Input(shape=(110,), name="state")

        concat = Concatenate()([state, tiles_out])

        x = Dense(256, activation="relu")(concat)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)
        x = Dense(32, activation="relu")(x)
        out_actions = Dense(1, activation="tanh")(x)

        model = Model(inputs=[state, tiles], outputs=[out_actions])
        model.compile(optimizer=Adam(learning_rate=1e-4),
                      loss='MSE', metrics=['accuracy'])

        self.logger.info(model.summary())
        return model

    def preprocess(self, state):
        state = state[0]
        input_state = [state["player_score"], state["opp_score"]]

        if len(state["units"]) == 0 or len(state["opp_units"]) == 0:
            return [np.array([0]*110), np.array([[0]*7]*7)]

        for unit in state["units"] + state["opp_units"]:
            for i in range(1, len(unit)-1):
                if i < 7:
                    input_state.append(unit[i])
                elif i == 7:
                    onehot = np.zeros(3)
                    onehot[unit[i]] = 1
                    input_state.extend(onehot.tolist())
                else:
                    onehot = np.zeros(2)
                    onehot[unit[i]] = 1
                    input_state.extend(onehot.tolist())

        unit_x = state["units"][0][1]
        unit_y = state["units"][0][2]

        closest_bullet = [0] * 8
        min_dist = 10**9 + 7
        for bullet in state["bullets"]:
            dist = (bullet[0] - unit_x)**2 + (bullet[1] - unit_y)**2
            if dist < min_dist:
                min_dist = dist
                closest_bullet = bullet
        input_state.extend(closest_bullet)

        closest_mine = [0] * 7
        min_dist = 10**9 + 7
        for mine in state["mines"]:
            dist = (mine[0] - unit_x)**2 + (mine[1] - unit_y)**2
            if dist < min_dist:
                min_dist = dist
                closest_mine = mine
        onehot = np.zeros(4)
        onehot[closest_mine[2]] = 1
        closest_mine = closest_mine[:2] + \
            onehot.tolist() + \
            closest_mine[3:]
        input_state.extend(closest_mine)

        for loot in state["loot_boxes"][:10]:
            onehot = np.zeros(3)
            onehot[loot[2]] = 1
            input_state.extend(loot[:2] +
                               onehot.tolist() +
                               loot[3:])
        prev_len = len(input_state)
        for i in range(prev_len, 110):
            input_state.append(0)

        for i in range(len(input_state)):
            if input_state[i] is None or type(input_state[i]) == dict:
                input_state[i] = 0

        return [np.array(input_state), np.array(state["units"][0][-1])]

    def build_branch(self, x, n_outputs, name, activation):
        x = Dense(32, activation="relu")(x)
        x = Dense(n_outputs, activation=activation, name=f"{name}_output")(x)
        return x

    def act(self, state):
        # self.logger.debug("State")
        # self.logger.debug(state[0])
        input_state = self.preprocess(state)
        # self.logger.debug(input_state)

        predicted_action = self.actor.predict([
            input_state[0].reshape(1, -1),
            input_state[1].reshape(1, 7, 7),
            dummy_12, dummy_5, dummy_4, dummy_2, dummy_2,
            dummy_1, dummy_1, dummy_1
        ])
        action_probs = [q[0] for q in predicted_action]

        values = self.critic.predict([
            input_state[0].reshape(1, -1),
            input_state[1].reshape(1, 7, 7)
        ])

        self.logger.debug("Action")
        self.logger.debug(predicted_action)

        # probabilisitc random choice for action
        discrete_action = [np.random.choice(len(q), p=q)
                           for q in action_probs]
        print(discrete_action)
        action = self.env.get_action(discrete_action)
        action = {3: action}
        return action, discrete_action, values, action_probs

    def ppo_loss(self, oldpolicy_probs, advantages, rewards, values):
        def loss(y_true, y_pred):
            total_loss = 0
            newpolicy_probs = y_pred
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) -
                          K.log(oldpolicy_probs + 1e-10))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - clipping_val,
                        max_value=1 + clipping_val) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            critic_loss = K.mean(K.square(rewards - values))
            total_loss += critic_discount * critic_loss + actor_loss * K.mean(
                -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))

            return total_loss

        return loss

    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)

    def remember(self, state, action, discrete_action, value, reward, done):
        self.memory.append(
            [state, action, discrete_action, value, reward, done])

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        # create batch
        states = []
        states_tiles = []
        actions_probs = [[] for i in range(5)]
        actions_onehot = [[] for i in range(5)]
        values = []
        rewards = []
        masks = []

        for _ in range(self.batch_size):
            state, action_prob, action, value, reward, done = self.memory.popleft()
            states.append(state[0])
            states_tiles.append(state[1])
            for i in range(len(action)):
                actions_probs[i].append(action_prob[i])
                onehot = np.zeros(len(action_prob[i]))
                onehot[action[i]] = 1
                actions_onehot[i].append(onehot.tolist())
            values.append(value)
            rewards.append(reward)
            masks.append(not done)

        # add one extra future value
        values.append(self.critic.predict([
            state[0].reshape(1, -1),
            state[1].reshape(1, 7, 7)
        ]))
        # np.array for each output
        for i in range(5):
            actions_onehot[i] = np.array(actions_onehot[i])

        # fit actor and critic models
        returns, advantages = self.get_advantages(values, masks, rewards)
        actor_input = {
            "state": np.asarray(states),
            "tiles": np.asarray(states_tiles),
            "oldpolicy_probs_aim": np.asarray(actions_probs[0]).reshape(-1, 1, 12),
            "oldpolicy_probs_velocity": np.asarray(actions_probs[1]).reshape(-1, 1, 5),
            "oldpolicy_probs_action": np.asarray(actions_probs[2]).reshape(-1, 1, 4),
            "oldpolicy_probs_jump": np.asarray(actions_probs[3]).reshape(-1, 1, 2),
            "oldpolicy_probs_jump_down": np.asarray(actions_probs[4]).reshape(-1, 1, 2),
            "advantages": np.asarray(advantages),
            "rewards": np.reshape(rewards, newshape=(-1, 1, 1)),
            "values": np.asarray(values[:-1])
        }

        self.actor.fit(actor_input,
                       actions_onehot,
                       verbose=True, shuffle=True,
                       epochs=self.epoch_count+self.config.epochs,
                       initial_epoch=self.epoch_count)
        self.critic.fit([np.array(states),
                         np.array(states_tiles)],
                        np.reshape(returns, (-1, 1)),
                        shuffle=True, verbose=True,
                        epochs=self.epoch_count+self.config.epochs,
                        initial_epoch=self.epoch_count)
        return

    def custom_logic(self, cur_state, action, reward, new_state, done, step,
                     discrete_action, value, action_prob):
        # update replay memory
        self.remember(self.preprocess(cur_state), action_prob,
                      discrete_action, value, reward, done)

        # train every x steps
        if self.frame_count % self.train_every == 0:
            self.logger.info("Training model")
            self.train()

        self.frame_count += 1
        return

    def save_model(self, fn):
        self.actor.save(fn)
        self.critic.save(f"{fn}_critic")
