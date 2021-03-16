class Strategy:
    def __init__(self, env, config):
        self.env = env
        self.config = config

    def act(self, state):
        return self.env.sample_action()

    def custom_logic(self, cur_state, action, reward, new_state, done, episode):
        print("No Custom Logic for random agent")
        return

    def save_model(self, fn):
        return
