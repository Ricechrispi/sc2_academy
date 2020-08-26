import os
import sys

import numpy as np
from absl import logging

from tf_agents.policies import random_tf_policy

from custom_tf_agents import create_ppo_agent
from ppo_trainer import create_envs
from ppo_trainer import create_checkpoints
from metrics import custom_metric_utils


def eval_only(FLAGS):

    tf_env, eval_env = None, None  # default so that we can close them in case of an exception
    try:
        tf_env, eval_env = create_envs(env_name=FLAGS.map_name,
                                       use_multiprocessing=FLAGS.use_multiprocessing,
                                       num_parallel_envs=FLAGS.num_parallel_envs,
                                       visualize_eval=FLAGS.visualize_eval,
                                       mock_train_envs=True)

        # we need to load this after creating the envs, otherwise we get infinite CUDA_ERROR_NOT_INITIALIZED errors
        import tensorflow as tf

        # setting up the folders to store the agent and logs in
        root_dir = "./results"
        root_dir = os.path.expanduser(root_dir)
        root_dir = os.path.join(root_dir, FLAGS.map_name)

        eval_dir = os.path.join(root_dir, 'eval')
        checkpoint_dir = os.path.join(root_dir, 'checkpoints')

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=FLAGS.summaries_flush_secs * 1000)
        eval_summary_writer.set_as_default()

        with tf.compat.v2.summary.record_if(True):

            # setting up the metrics
            env_steps_metric, step_metrics, train_metrics, eval_metrics = custom_metric_utils.get_metrics(FLAGS)

            global_step = tf.compat.v1.train.get_or_create_global_step()

            if FLAGS.just_eval_random:
                # setting up the random baseline policy
                eval_policy = random_tf_policy.RandomTFPolicy(time_step_spec=eval_env.time_step_spec(),
                                                              action_spec=eval_env.action_spec())
            else:
                # setting up the agent
                tf_agent = create_ppo_agent(env=tf_env, global_step=global_step, FLAGS=FLAGS)

                eval_policy = tf_agent.policy  # .policy exploits all the knowledge with no further exploration
                # eval_policy = tf_agent.collect_policy  # .collect_policy has exploration built in

                train_checkpointer, policy_checkpointer = create_checkpoints(agent=tf_agent,
                                                                             global_step=global_step,
                                                                             checkpoint_dir=checkpoint_dir,
                                                                             train_metrics=train_metrics,
                                                                             eval_metrics=eval_metrics)
                # this _should_ load if there is already a checkpoint
                train_checkpointer.initialize_or_restore()
                policy_checkpointer.initialize_or_restore()

            global_step = tf.compat.v1.train.get_global_step()

            # buckets are needed for histogram output.
            # buckets = max(action_id) + 1. Since action_ids are 0 indexed.
            buckets = np.int32(eval_env.action_spec()[0].maximum.max() + 1)
            results = custom_metric_utils.eager_compute(
                eval_metrics,
                eval_env,
                eval_policy,
                num_episodes=FLAGS.num_eval_episodes,
                train_step=global_step,
                summary_writer=eval_summary_writer,
                summary_prefix='Metrics',
                buckets=buckets
            )
            # This plots the metrics against the other 'steps'
            # -> NumberOfEpisodes or NumberOfEnvironmentSteps
            for eval_metric in eval_metrics:
                if eval_metric.name != 'ChosenActionIDHistogram':
                    eval_metric.tf_summaries(step_metrics=step_metrics)

            results_str = ''
            for k, v in results.items():
                results_str = ''.join([results_str, '\n', str(k), ': ', str(v.numpy())])
            tf.print(''.join(['results: ', results_str]), output_stream=sys.stdout)
            logging.info(results_str)

            tf_env.close()
            eval_env.close()

    except KeyboardInterrupt:
        if tf_env is not None:
            tf_env.close()
        if eval_env is not None:
            eval_env.close()
        pass
