from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, \
    Reshape, Concatenate
from tensorflow.keras.callbacks import TensorBoard
import math
import random
import numpy as np
import tensorflow.keras.backend as K

class Strategy:
    def __init__(self, env, config, logger, logdir):
        self.env = env
        self.config = config
        self.logger = logger

        self.frame_count = 0
        self.action_count = 0
        self.epoch_count = 0
        self.memory = deque(maxlen=10000)

        self.epsilon = self.config.epsilon
        self.batch_size = self.config.minibatch_size
        self.train_every = self.config.train_every
        self.update_target_every = self.config.update_every
        self.epsilon_decay_value = self.config.epsilon_decay

        self.actor = self.create_actor()
        self.critic = self.create_critic()

    def create_actor(self):
        tiles = Input(shape=(7, 7), name="tiles")
        tiles_out = Flatten()(tiles)
        tiles_out = Embedding(5, 4)(tiles_out)
        tiles_out = Reshape(target_shape=(196, ))(tiles_out)

        state = Input(shape=(110,), name="state")

        concat = Concatenate()([state, tiles_out])

        x = Dense(256, activation="relu")(concat)
        x = Dense(128, activation="relu")(x)
        x = Dense(64, activation="relu")(x)

        aim = self.build_branch(x, 12, "aim")
        velocity = self.build_branch(x, 5, "velocity")
        action = self.build_branch(x, 4, "action")
        jump = self.build_branch(x, 2, "jump")
        jump_down = self.build_branch(x, 2, "jump_down")

        model = Model(inputs=[state, tiles], outputs=[
                      aim, velocity, action, jump, jump_down])
        model.compile(optimizer='Adam',
                      loss='MSE', metrics=['accuracy'])

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

        aim = self.build_branch(x, 1, "aim")
        velocity = self.build_branch(x, 1, "velocity")
        action = self.build_branch(x, 1, "action")
        jump = self.build_branch(x, 1, "jump")
        jump_down = self.build_branch(x, 1, "jump_down")

        model = Model(inputs=[state, tiles], outputs=[
                      aim, velocity, action, jump, jump_down])
        model.compile(optimizer='Adam',
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
                    input_state.extend(
                        tf.one_hot(unit[i], 3,
                                   on_value=1.0, off_value=0.0).numpy().tolist())
                else:
                    input_state.extend(
                        tf.one_hot(unit[i], 2,
                                   on_value=1.0, off_value=0.0).numpy().tolist())

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
        closest_mine = closest_mine[:2] + \
            tf.one_hot(closest_mine[2], 4,
                       on_value=1.0, off_value=0.0).numpy().tolist() + \
            closest_mine[3:]
        input_state.extend(closest_mine)

        for loot in state["loot_boxes"][:10]:
            input_state.extend(loot[:2] +
                               tf.one_hot(loot[2], 3,
                                          on_value=1.0, off_value=0.0)
                               .numpy().tolist() +
                               loot[3:])
        prev_len = len(input_state)
        for i in range(prev_len, 110):
            input_state.append(0)

        for i in range(len(input_state)):
            if input_state[i] is None or type(input_state[i]) == dict:
                input_state[i] = 0

        return [np.array(input_state), np.array(state["units"][0][-1])]


    def build_branch(self, x, n_outputs, name):
        x = Dense(32, activation="relu")(x)
        x = Dense(n_outputs, name=f"{name}_output")(x)
        return x

    def act(self, state):
        # self.logger.debug("State")
        # self.logger.debug(state[0])
        input_state = self.preprocess(state)
        # self.logger.debug(input_state)

        current_q_action = self.actor.predict([
            input_state[0].reshape(1, -1),
            input_state[1].reshape(1, 7, 7)
        ])
        current_q_critic = self.critic.predict([
            input_state[0].reshape(1, -1),
            input_state[1].reshape(1, 7, 7)
        ])
        self.logger.debug("Action")
        self.logger.debug(current_q_action)
        # if np.random.random() > self.epsilon:
        self.action_count += 1
        discrete_action = [np.argmax(q_values[0])
                            for q_values in current_q_action]
        action = self.env.get_action(discrete_action)
        # else:
        #     action, discrete_action = self.env.sample_action()
        action = {3: action}
        return action, discrete_action


    def ppo_loss(self, oldpolicy_probs, advantages, rewards, values):
        def loss(y_true, y_pred):
            clipping_val= 0.2
            critic_discount = 0.95

            newpolicy_probs = y_pred
            ratio = K.exp(K.log(newpolicy_probs + 1e-10) - K.log(oldpolicy_probs + 1e-10))
            p1 = ratio * advantages
            p2 = K.clip(ratio, min_value=1 - clipping_val, max_value=1 + clipping_val) * advantages
            actor_loss = -K.mean(K.minimum(p1, p2))
            critic_loss = K.mean(K.square(rewards - values))
            total_loss = critic_discount * critic_loss + actor_loss  * K.mean(
                -(newpolicy_probs * K.log(newpolicy_probs + 1e-10)))
            return total_loss

        return loss

        
    def get_advantages(self, values, masks, rewards):
        returns = []
        gae = 0
        gamma = 0.99
        lmbda = 0.95
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + gamma * values[i + 1] * masks[i] - values[i]
            gae = delta + gamma * lmbda * masks[i] * gae
            returns.insert(0, gae + values[i])

        adv = np.array(returns) - values[:-1]
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-10)


    def custom_logic(self, cur_state, action, reward, new_state, done, episode):
        q_value = model_critic.predict(state_input, steps=1)
        values.append(q_value)
        returns, advantages = get_advantages(values, masks, rewards)
        actor_loss = model_actor.fit(
            [states, actions_probs, advantages, np.reshape(rewards, newshape=(-1, 1, 1)), values[:-1]],
            [(np.reshape(actions_onehot, newshape=(-1, n_actions)))], verbose=True, shuffle=True, epochs=8,
            callbacks=[tensor_board])
        critic_loss = model_critic.fit([states], [np.reshape(returns, newshape=(-1, 1))], shuffle=True, epochs=8,
                                    verbose=True, callbacks=[tensor_board])

            

    def save_model(self, fn):
        raise NotImplementedError
