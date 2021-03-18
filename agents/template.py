class Strategy:
    def __init__(self, env, config, logger):
        self.env = env
        self.config = config
        raise NotImplementedError

    def act(self, state):
        raise NotImplementedError

    def custom_logic(self, cur_state, action, reward, new_state, done, episode):
        raise NotImplementedError

    def save_model(self, fn):
        raise NotImplementedError
