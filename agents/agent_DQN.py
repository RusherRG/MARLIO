from collections import deque


class Strategy:
    def __init__(self, env, config):
        self.env = env
        self.config = config

        self.memory = deque(maxlen=2000)
        self.model = self.create_model()
        self.target_model = self.create_model()

    def create_model(self):
        model = "Model"

        return model

    def act(self, state):

        return self.env.action_space.sample()

    def remember(self, state, action, reward, new_state, done):
        self.memory.append([state, action, reward, new_state, done])

    def replay(self):
        raise NotImplementedError

    def target_train(self):
        raise NotImplementedError

    def custom_logic(self, cur_state, action, reward, new_state, done, episode):
        raise NotImplementedError

    def save_model(self, fn):
        self.model.save(fn)
