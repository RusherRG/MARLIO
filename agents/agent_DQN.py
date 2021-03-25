from collections import deque
import tensorflow as tf
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.callbacks import TensorBoard
import math
import random
import numpy as np


TRAIN_EVERY = 16
MINIBATCH_SIZE = 32
DISCOUNT = 0.99


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
        self.batch_size = MINIBATCH_SIZE
        self.train_every = TRAIN_EVERY
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

    def build_velocity_branch(self, x):
        x = Dense(5, name="velocity_output")(x)
        return x

    def build_action_branch(self, x):
        x = Dense(4, name="action_output")(x)
        return x

    def build_jump_branch(self, x):
        x = Dense(2, name="jump_output")(x)
        return x

    def build_jump_down_branch(self, x):
        x = Dense(2, name="jump_down_output")(x)
        return x

    def create_model(self):
        input_shape = (1, 132)
        inputs = Input(shape=input_shape)
        x = Dense(256, activation="relu")(inputs)
        x = Dense(128, activation="relu")(inputs)
        x = Dense(64, activation="relu")(inputs)

        aim = self.build_branch(x, 12, "aim")
        velocity = self.build_branch(x, 5, "velocity")
        action = self.build_branch(x, 4, "action")
        jump = self.build_branch(x, 2, "jump")
        jump_down = self.build_branch(x, 2, "jump_down")

        model = Model(inputs=inputs, outputs=[
                      aim, velocity, action, jump, jump_down])
        model.compile(optimizer='Adam',
                      loss='MSE', metrics=['accuracy'])
        return model

    def load_model(self, model_path):
        model = load_model(model_path)
        return model

    def preprocess(self, state):
        state = state[0]
        input_state = [state["player_score"], state["opp_score"]]
        
        if len(state["units"]) == 0 or len(state["opp_units"]) == 0:
            input_state.extend([0]*130)
            return np.array(input_state).reshape(1, -1)

        input_state.extend(state["units"][0][1:-1])
        input_state.extend(state["opp_units"][0][1:-1])
        
        for tiles in state["units"][0][-1]:
            input_state.extend(tiles)
        
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
        input_state.extend(closest_mine)

        for loot in state["loot_boxes"][:10]:
            input_state.extend(loot)

        prev_len = len(input_state)
        for _ in range(prev_len, 132):
            input_state.append(0)
        input_state = input_state[:132]

        for i in range(len(input_state)):
            if input_state[i] is None or type(input_state[i]) == dict:
                input_state[i] = 0

        return np.array(input_state).reshape(1, -1)

    def act(self, state):
        self.logger.debug("State")
        self.logger.debug(state[0])
        input_state = self.preprocess(state)
        self.logger.debug(input_state)
        self.logger.debug(input_state.shape)

        current_q_values = self.model.predict(np.array([input_state]))
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
        X, Y = [], [[] for i in range(5)]

        for cur_state, action, reward, new_state, done in minibatch:
            if not done:
                # get the future q value
                future_q = self.target_model.predict(np.array([new_state]))
                new_q = [reward + self.discount*np.max(future_q[i][0])
                         for i in range(len(action))]
            else:
                new_q = [reward for i in range(len(action))]
            # get current q-value and update it with new q-value
            cur_q = self.model.predict(np.array([cur_state]))
            for i in range(len(action)):
                cur_q[i][0] = cur_q[i][0][0]
            for i in range(len(action)):
                cur_q[i][0][0][action[i]] = new_q[i]

            X.append(cur_state)
            for i in range(len(action)):
                Y[i].append(cur_q[i][0])
        for i in range(len(action)):
            Y[i] = np.array(Y[i])

        self.model.fit(np.array(X), Y,
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
