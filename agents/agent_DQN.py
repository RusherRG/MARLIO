from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Embedding, Flatten, \
    Reshape, Concatenate
from tensorflow.keras.callbacks import TensorBoard
import math
import random
import numpy as np


DISCOUNT = 0.95


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
        self.discount = DISCOUNT
        self.batch_size = self.config.minibatch_size
        self.train_every = self.config.train_every
        self.update_target_every = self.config.update_every
        self.epsilon_decay_value = self.config.epsilon_decay

        if self.config.model_path is None:
            self.logger.info("Create model and target model")
            self.model = self.create_model()
            self.target_model = self.create_model()
            self.update_target_model()
        else:
            self.logger.info("Loading model and target model")
            self.model = self.load_model(self.config.model_path)
            self.target_model = self.load_model(self.config.model_path)

    def build_branch(self, x, n_outputs, name):
        x = Dense(32, activation="relu")(x)
        x = Dense(n_outputs, name=f"{name}_output")(x)
        return x

    def create_model(self):
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

    def load_model(self, model_path):
        model = load_model(model_path)
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

    def act(self, state):
        # self.logger.debug("State")
        # self.logger.debug(state[0])
        input_state = self.preprocess(state)
        # self.logger.debug(input_state)

        current_q_values = self.model.predict([
            input_state[0].reshape(1, -1),
            input_state[1].reshape(1, 7, 7)
        ])
        self.logger.debug("Action")
        self.logger.debug(current_q_values)
        if np.random.random() > self.epsilon:
            self.action_count += 1
            discrete_action = [np.argmax(q_values[0])
                               for q_values in current_q_values]
            action = self.env.get_action(discrete_action)
        else:
            action, discrete_action = self.env.sample_action()
        action = {3: action}
        return action, discrete_action

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        raise NotImplementedError

    def train(self):
        if len(self.memory) < self.batch_size:
            return

        minibatch = random.sample(self.memory, self.batch_size)
        X_state, X_tiles, Y = [], [], [[] for i in range(5)]

        for cur_state, action, reward, new_state, done in minibatch:
            if not done:
                # get the future q value
                future_q = self.target_model.predict([
                    new_state[0].reshape(1, -1),
                    new_state[1].reshape(1, 7, 7)
                ])
                new_q = [reward + self.discount*np.max(future_q[i][0])
                         for i in range(len(action))]
            else:
                new_q = [reward for i in range(len(action))]
            
            # get current q-value and update it with new q-value
            cur_q = self.model.predict([
                cur_state[0].reshape(1, -1),
                cur_state[1].reshape(1, 7, 7)
            ])
            for i in range(len(action)):
                cur_q[i] = cur_q[i][0]
            for i in range(len(action)):
                cur_q[i][action[i]] = new_q[i]
            
            X_state.append(cur_state[0])
            X_tiles.append(cur_state[1])
            for i in range(len(action)):
                Y[i].append(cur_q[i])

        for i in range(len(action)):
            Y[i] = np.array(Y[i])

        self.model.fit({"state": np.array(X_state),
                        "tiles": np.array(X_tiles)},
                       Y,
                       batch_size=self.batch_size,
                       epochs=self.epoch_count+self.config.epochs,
                       initial_epoch=self.epoch_count)
        self.epoch_count += self.config.epochs
        return

    def update_target_model(self):
        self.target_model.set_weights(self.model.get_weights())
        return

    def custom_logic(self, cur_state, action, reward, new_state, done, step):
        # update replay
        self.remember(self.preprocess(cur_state), action, reward,
                      self.preprocess(new_state), done)

        # target train every x steps
        if self.frame_count % self.train_every == 0:
            self.logger.info(f"#Predicted actions: {self.action_count}" +
                             f" | #epsteps: {step}")
            self.action_count = 0

            self.logger.info("Training model")
            self.train()

        if self.frame_count % self.update_target_every == 0:
            self.logger.info("Updating target model")
            self.update_target_model()

        # epsilon decay
        if self.epsilon > 0.1:
            self.epsilon -= self.epsilon_decay_value

        self.frame_count += 1
        return

    def save_model(self, fn):
        self.model.save(fn)
