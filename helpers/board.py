import tensorflow as tf
import shutil


class TensorboardLogger:
    def __init__(self, config, output_dir):
        self.episode = 0
        self.output_dir = output_dir
        self.ep_writer = tf.summary.create_file_writer(
            logdir=f"{output_dir}/logs/ep_{self.episode}"
        )
        shutil.rmtree(f"{output_dir}/logs/")
        self.glob_writer = tf.summary.create_file_writer(
            logdir=f"{output_dir}/logs/glob"
        )
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
            logdir=f"{self.output_dir}/logs/ep_{self.episode}"
        )
        return
