import tensorflow as tf

from tf_agents.metrics import tf_metric
from tf_agents.metrics.tf_metrics import TFDeque
from tf_agents.metrics.tf_metrics import ChosenActionHistogram
from tf_agents.utils import common


class CustomTFDeque(TFDeque):
    """Adds the ability to compute the standard deviation of the data"""

    def __init__(self, max_len, dtype, shape=(), name='TFDeque'):
        super().__init__(max_len, dtype, shape, name)

    @common.function(autograph=True)
    def std(self):
        if tf.equal(self._head, 0):
            return tf.zeros(self._spec.shape, self._spec.dtype)
        return tf.math.reduce_std(self.data, axis=0)


class StdMetric(tf_metric.TFStepMetric):
    """Metric to compute the standard deviation of the returns."""

    def __init__(self,
                 name='ReturnStd',
                 prefix='Metrics',
                 dtype=tf.float32,
                 batch_size=1,
                 buffer_size=10):
        super(StdMetric, self).__init__(name=name, prefix=prefix)
        self._buffer = CustomTFDeque(buffer_size, dtype)
        self._dtype = dtype
        self._return_accumulator = common.create_variable(
            initial_value=0, dtype=dtype, shape=(batch_size,), name='Accumulator')

    @common.function(autograph=True)
    def call(self, trajectory):
        # Zero out batch indices where a new episode is starting.
        self._return_accumulator.assign(
            tf.where(trajectory.is_first(), tf.zeros_like(self._return_accumulator),
                     self._return_accumulator))

        # Update accumulator with received rewards.
        self._return_accumulator.assign_add(trajectory.reward)

        # Add final returns to buffer.
        last_episode_indices = tf.squeeze(tf.where(trajectory.is_last()), axis=-1)
        for indx in last_episode_indices:
            self._buffer.add(self._return_accumulator[indx])

        return trajectory

    def result(self):
        return self._buffer.std()

    @common.function
    def reset(self):
        self._buffer.clear()
        self._return_accumulator.assign(tf.zeros_like(self._return_accumulator))


class ChosenActionIDHistogram(ChosenActionHistogram):
    """Metric to compute the frequency of each action_id chosen."""

    def __init__(self, name='ChosenActionIDHistogram', dtype=tf.int32, buffer_size=1000):
        super().__init__(name, dtype, buffer_size)

    @common.function
    def call(self, trajectory):
        self._buffer.extend(trajectory.action[0])
        return trajectory
