import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import random


class Logger():
    """Logging in tensorboard without tensorflow ops."""

    def log_scalar(self):
        summary_writer = tf.summary.create_file_writer('logs/a')
        summary_writer2 = tf.summary.create_file_writer('logs/b')
        for steps in range(100):
            with summary_writer.as_default():
                tf.summary.scalar('value', random.randint(0, 1000), step=steps)
            with summary_writer2.as_default():
                tf.summary.scalar('value', random.randint(0, 1000), step=steps)
            # summary_writer.add_summary(, steps)


test = Logger()
test.log_scalar()
