import tensorflow as tf
import shutil


class TensorboardLogger:
    def __init__(self, config, output_dir):
        self.output_dir = output_dir
        self.ep_writer = tf.summary.create_file_writer(
            logdir=f"{output_dir}/logs/ep_0"
        )
        self.glob_writer = tf.summary.create_file_writer(
            logdir=f"{output_dir}/logs/glob"
        )
        return

    def log_step(self, episode, step, action, reward):
        with self.ep_writer.as_default():
            tf.summary.scalar(name="reward", data=reward, step=step)
        return

    def log_episode(self, episode, steps, tot_reward, done):
        with self.glob_writer.as_default():
            tf.summary.scalar(name="ep_reward", data=tot_reward, step=episode)
        episode = episode + 1
        self.ep_writer = tf.summary.create_file_writer(
            logdir=f"{self.output_dir}/logs/ep_{episode}"
        )
        return
