import tensorflow as tf


class TensorboardLogger:
    def __init__(self, config):
        self.episode = 0
        self.ep_writer = tf.summary.create_file_writer(
            logdir=f"./logs/ep_{self.episode}"
        )
        self.glob_writer = tf.summary.create_file_writer(logdir="./logs/glob")
        return

    def log_step(self, episode, step, action, reward, new_state):
        with self.ep_writer.as_default():
            tf.summary.scalar(name="reward", data=reward, step=step)
        return

    def log_episode(self, episode, steps, tot_reward, done):
        self.episode = episode
        with self.glob_writer.as_default():
            tf.summary.scalar(name="ep_reward", data=tot_reward, step=episode)
        self.ep_writer = tf.summary.create_file_writer(
            logdir=f"./logs/ep_{self.episode}"
        )
        return
