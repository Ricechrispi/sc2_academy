import collections
import tensorflow as tf

from tf_agents.drivers import dynamic_episode_driver
from tf_agents.utils import common
from tf_agents.metrics import tf_metrics

from metrics.custom_metrics import StdMetric
from metrics.custom_metrics import ChosenActionIDHistogram


def get_metrics(FLAGS):
    """Create all the evaluation and training metrics"""

    env_steps_metric = tf_metrics.EnvironmentSteps()
    step_metrics = [
        tf_metrics.NumberOfEpisodes(),
        env_steps_metric,
    ]
    train_metrics = step_metrics + [
        tf_metrics.AverageReturnMetric(batch_size=FLAGS.num_parallel_envs),
        tf_metrics.AverageEpisodeLengthMetric(batch_size=FLAGS.num_parallel_envs),
    ]
    eval_metrics = [
        tf_metrics.AverageReturnMetric(buffer_size=FLAGS.num_eval_episodes),
        tf_metrics.AverageEpisodeLengthMetric(buffer_size=FLAGS.num_eval_episodes),
        tf_metrics.MaxReturnMetric(buffer_size=FLAGS.num_eval_episodes),
        tf_metrics.MinReturnMetric(buffer_size=FLAGS.num_eval_episodes),
        StdMetric(buffer_size=FLAGS.num_eval_episodes),
        ChosenActionIDHistogram(),
    ]
    return env_steps_metric, step_metrics, train_metrics, eval_metrics


def eager_compute(metrics,
                  environment,
                  policy,
                  num_episodes=1,
                  train_step=None,
                  summary_writer=None,
                  summary_prefix='',
                  use_function=True,
                  buckets=None):
    """Compute metrics using `policy` on the `environment`.

    *DISCLAIMER*: This is mostly a copy from tf_agents.eval.metric_utils. I had to adapt
    it to support my histograms.

    *NOTE*: Because placeholders are not compatible with Eager mode we can not use
    python policies. Because we use tf_policies we need the environment time_steps
    to be tensors making it easier to use a tf_env for evaluations. Otherwise this
    method mirrors `compute` directly.

    Args:
      metrics: List of metrics to compute.
      environment: tf_environment instance.
      policy: tf_policy instance used to step the environment.
      num_episodes: Number of episodes to compute the metrics over.
      train_step: An optional step to write summaries against.
      summary_writer: An optional writer for generating metric summaries.
      summary_prefix: An optional prefix scope for metric summaries.
      use_function: Option to enable use of `tf.function` when collecting the
        metrics.
      buckets: Number of buckets used for histogram output.
    Returns:
      A dictionary of results {metric_name: metric_value}
    """
    for metric in metrics:
        metric.reset()

    time_step = environment.reset()
    policy_state = policy.get_initial_state(environment.batch_size)

    driver = dynamic_episode_driver.DynamicEpisodeDriver(
        environment,
        policy,
        observers=metrics,
        num_episodes=num_episodes)
    if use_function:
        common.function(driver.run)(time_step, policy_state)
    else:
        driver.run(time_step, policy_state)

    results = [(metric.name, metric.result()) for metric in metrics]
    if train_step is not None and summary_writer:
        with summary_writer.as_default():
            for m in metrics:
                tag = common.join_scope(summary_prefix, m.name)
                if m.name == 'ChosenActionIDHistogram': # this is my change
                    tf.compat.v2.summary.histogram(name=tag, data=m.result(), step=train_step, buckets=buckets)
                else:
                    tf.compat.v2.summary.scalar(name=tag, data=m.result(), step=train_step)

    return collections.OrderedDict(results)
