import time
import os
import sys
import numpy as np

import gym

from tf_agents.replay_buffers import tf_uniform_replay_buffer
from tf_agents.utils import common
from tf_agents.drivers import dynamic_episode_driver
from tf_agents.environments import tf_py_environment
from tf_agents.environments import parallel_py_environment
from tf_agents.environments import batched_py_environment
from tf_agents.environments import gym_wrapper
from tf_agents.eval import metric_utils

from absl import logging

from envs.sc2_gym_env import SC2GymEnv
from custom_tf_agents import create_ppo_agent
from metrics import custom_metric_utils


def create_envs(env_name, use_multiprocessing, num_parallel_envs, visualize_eval=False, mock_train_envs=False):

    def env_load_fn(env_map_name, visualize=False, mock=False):
        env = gym_wrapper.GymWrapper(gym_env=SC2GymEnv(map_name=env_map_name, visualize=visualize, mock=mock),
                                     spec_dtype_map={gym.spaces.Box: np.float32,
                                                     gym.spaces.Discrete: np.int32,
                                                     gym.spaces.MultiBinary: np.float32},
                                     )
        return env

    if num_parallel_envs == 1:
        par_env = env_load_fn(env_map_name=env_name, mock=mock_train_envs)
    elif use_multiprocessing:
        par_env = parallel_py_environment.ParallelPyEnvironment(
            [lambda: env_load_fn(env_map_name=env_name, mock=mock_train_envs)] * num_parallel_envs, start_serially=False)
    else:
        par_env = batched_py_environment.BatchedPyEnvironment(
            envs=[env_load_fn(env_map_name=env_name, mock=mock_train_envs) for _ in range(num_parallel_envs)])
    tf_env = tf_py_environment.TFPyEnvironment(par_env)
    tf_env.reset()

    eval_env = env_load_fn(env_name, visualize=visualize_eval)
    eval_env = tf_py_environment.TFPyEnvironment(eval_env)
    eval_env.reset()

    return tf_env, eval_env


def create_replay_buffer(agent, num_parallel_envs, replay_buffer_max_length):
    return tf_uniform_replay_buffer.TFUniformReplayBuffer(
        data_spec=agent.collect_data_spec,
        batch_size=num_parallel_envs,
        max_length=replay_buffer_max_length)


def create_collect_driver(env, collect_policy, buffer, train_metrics, num_collect_episodes):
    return dynamic_episode_driver.DynamicEpisodeDriver(
        env,
        collect_policy,
        observers=[buffer.add_batch] + train_metrics,
        num_episodes=num_collect_episodes
    )


def create_checkpoints(agent, global_step, checkpoint_dir, train_metrics, eval_metrics, max_to_keep=2):
    train_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(checkpoint_dir, 'collect_policy'),
        max_to_keep=max_to_keep,
        agent=agent,
        policy=agent.collect_policy,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(train_metrics, 'train_metrics')
    )

    policy_checkpointer = common.Checkpointer(
        ckpt_dir=os.path.join(checkpoint_dir, 'policy'),
        max_to_keep=max_to_keep,
        agent=agent,
        policy=agent.policy,
        global_step=global_step,
        metrics=metric_utils.MetricsGroup(eval_metrics, 'eval_metrics')
    )
    return train_checkpointer, policy_checkpointer


def train_eval(FLAGS):
    tf_env, eval_env = None, None  # default so that we can close them in case of an exception
    try:
        tf_env, eval_env = create_envs(env_name=FLAGS.map_name,
                                       use_multiprocessing=FLAGS.use_multiprocessing,
                                       num_parallel_envs=FLAGS.num_parallel_envs)

        # we need to load this after creating the envs, otherwise we get infinite CUDA_ERROR_NOT_INITIALIZED errors
        import tensorflow as tf

        # setting up the folders to store the agent and logs in
        root_dir = "./results"
        root_dir = os.path.expanduser(root_dir)
        root_dir = os.path.join(root_dir, FLAGS.map_name)

        train_dir = os.path.join(root_dir, 'train')
        eval_dir = os.path.join(root_dir, 'eval')
        best_checkpoint_dir = os.path.join(root_dir, 'best')
        checkpoint_dir = os.path.join(root_dir, 'checkpoints')

        train_summary_writer = tf.compat.v2.summary.create_file_writer(
            train_dir, flush_millis=FLAGS.summaries_flush_secs * 1000)
        train_summary_writer.set_as_default()

        eval_summary_writer = tf.compat.v2.summary.create_file_writer(
            eval_dir, flush_millis=FLAGS.summaries_flush_secs * 1000)

        # 'only record every summery interval', improves run time if we do not log all the time
        with tf.compat.v2.summary.record_if(
                lambda: tf.math.equal(global_step % FLAGS.summary_interval, 0)):

            # setting up the metrics
            env_steps_metric, step_metrics, train_metrics, eval_metrics = custom_metric_utils.get_metrics(FLAGS)

            # setting up the agent
            global_step = tf.compat.v1.train.get_or_create_global_step()
            tf_agent = create_ppo_agent(env=tf_env, global_step=global_step, FLAGS=FLAGS)

            buffer = create_replay_buffer(agent=tf_agent, num_parallel_envs=FLAGS.num_parallel_envs,
                                          replay_buffer_max_length=FLAGS.replay_buffer_max_length)

            eval_policy = tf_agent.policy  # .policy exploits all the knowledge with no further exploration
            collect_policy = tf_agent.collect_policy  # .collect_policy has exploration built in

            # setting up the train/eval loop
            collect_driver = create_collect_driver(env=tf_env,
                                                   collect_policy=collect_policy,
                                                   buffer=buffer,
                                                   train_metrics=train_metrics,
                                                   num_collect_episodes=FLAGS.num_collect_episodes)

            def train_step():
                # one single training step. gather_all() is deprecated, but the replacement does not work.
                trajectories = buffer.gather_all()
                return tf_agent.train(experience=trajectories)

            # This should optimize these functions by inserting them into the TF compute graph
            tf_agent.train = common.function(tf_agent.train, autograph=False)
            collect_driver.run = common.function(collect_driver.run, autograph=False)
            train_step = common.function(train_step)

            # checkpoints for the training process
            train_checkpointer, policy_checkpointer = create_checkpoints(agent=tf_agent,
                                                                         global_step=global_step,
                                                                         checkpoint_dir=checkpoint_dir,
                                                                         train_metrics=train_metrics,
                                                                         eval_metrics=eval_metrics)

            # best policies found during evaluation are saved here
            best_train_checkpointer, best_policy_checkpointer = create_checkpoints(agent=tf_agent,
                                                                                   global_step=global_step,
                                                                                   checkpoint_dir=best_checkpoint_dir,
                                                                                   train_metrics=train_metrics,
                                                                                   eval_metrics=eval_metrics,
                                                                                   max_to_keep=3)

            # this _should_ load if there is already a checkpoint, otherwise create a new one
            logging.info('initializing/restoring checkpoints:')
            train_checkpointer.initialize_or_restore()
            policy_checkpointer.initialize_or_restore()
            best_train_checkpointer.initialize_or_restore()
            best_policy_checkpointer.initialize_or_restore()
            logging.info('initializing/restoring done.')

            # The 'high-score' is saved to a file, to determine when to save a new 'best' policy.
            # Using a .txt file is not ideal, but this is the easiest solution.
            best_avg_return_path = os.path.join(best_checkpoint_dir, 'best_avg_return.txt')
            if os.path.isfile(best_avg_return_path):
                logging.info('best avg return file exists, reading content:')
                with open(best_avg_return_path, mode='r') as file:
                    lines = file.readlines()
                    high_score = float(lines[0])
                    logging.info(''.join(['got ', str(high_score), ' as best avg. return.']))
                    # in a numpy array so that we can later expand to hold high-scores
                    best_avg_return = np.array([high_score])
            else:
                logging.info('no best avg return file exists, creating it.')
                high_score = 0.0
                with open(best_avg_return_path, mode='w') as file:
                    file.write(str(high_score))
                best_avg_return = np.array([high_score])

            global_step = tf.compat.v1.train.get_global_step()
            collect_time = 0
            train_time = 0
            timed_at_step = global_step.numpy()

            # buckets are needed for histogram output.
            # buckets = max(action_id) + 1. Since action_ids are 0 indexed.
            buckets = np.int32(eval_env.action_spec()[0].maximum.max() + 1)
            def eval_run():
                logging.info('starting eval run:')
                # This first collects the data and then plots them against the number of steps (=train epochs)
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
                with eval_summary_writer.as_default():
                    for eval_metric in eval_metrics:
                        if eval_metric.name != 'ChosenActionIDHistogram':
                            eval_metric.tf_summaries(step_metrics=step_metrics)

                # This saves the best polices based on the average return
                avg_return = float(results['AverageReturn'])
                if avg_return > best_avg_return[0]:
                    best_avg_return[0] = avg_return
                    with open(best_avg_return_path, mode='w') as f:
                        f.write(str(avg_return))  # save new high score to file
                    logging.info('found new best policy with avg. return: ' + str(avg_return))
                    logging.info('saving at step: ' + str(global_step.numpy()))
                    best_train_checkpointer.save(global_step.numpy())
                    best_policy_checkpointer.save(global_step.numpy())
                logging.info('eval run done.')

            # training starts here!
            while env_steps_metric.result() < FLAGS.num_environment_steps:
                # calculate how far along in training we are (in %) and log it
                cur_steps = env_steps_metric.result().numpy()
                progress_str = ''.join([str(cur_steps), ' of ', str(FLAGS.num_environment_steps), ' env_steps done. ',
                                        str(100 * cur_steps / FLAGS.num_environment_steps), '%'])
                logging.info(progress_str)  # this logs it to the logging stream, default stderr
                tf.print(progress_str,
                         output_stream=sys.stdout)  # this logs it to stdout, to see progress during training

                global_step_val = global_step.numpy()
                # do an evaluation of the current performance every eval_interval iterations
                if global_step_val % FLAGS.eval_interval == 0:
                   eval_run()

                logging.info('starting collect run:')
                start_time = time.time()
                collect_driver.run()
                collect_time += time.time() - start_time
                logging.info('collect done.')

                logging.info('starting train step:')
                start_time = time.time()
                total_loss, _ = train_step()
                buffer.clear() # PPO throws away the replay buffer, this is no accident
                train_time += time.time() - start_time
                logging.info('train step done.')

                # log the train_metrics of the collected episodes
                for train_metric in train_metrics:
                    train_metric.tf_summaries(train_step=global_step, step_metrics=step_metrics)

                if global_step_val % FLAGS.log_interval == 0:
                    # log this to stdout to see progress during training even without tensorboard
                    avg_collect_reward = train_metrics[2].result()
                    tf.print('avg. collect reward: ', avg_collect_reward, output_stream=sys.stdout)

                    logging.info('train epochs: step = %d, loss = %f', global_step_val, total_loss)
                    steps_per_sec = ((global_step_val - timed_at_step) / (collect_time + train_time))
                    logging.info('train epochs: %.3f steps/sec', steps_per_sec)
                    logging.info('collect_time = %.3f, train_time = %.3f', collect_time, train_time)
                    with tf.compat.v2.summary.record_if(True):
                        tf.compat.v2.summary.scalar(name='global_steps_per_sec', data=steps_per_sec, step=global_step)

                    timed_at_step = global_step_val
                    collect_time = 0
                    train_time = 0

                if global_step_val % FLAGS.checkpoint_interval == 0:
                    logging.info('saving checkpoints:')
                    train_checkpointer.save(global_step_val)
                    policy_checkpointer.save(global_step_val)
                    logging.info('saving done.')

            # final eval before exiting
            logging.info('training is done, evaluating one final time:')
            eval_run()
            logging.info('final eval done.')

            # final save before exiting
            logging.info('saving checkpoint before exit:')
            train_checkpointer.save(global_step.numpy())
            policy_checkpointer.save(global_step.numpy())
            logging.info('saving done.')

            tf_env.close()
            eval_env.close()

    except KeyboardInterrupt:
        if tf_env is not None:
            tf_env.close()
        if eval_env is not None:
            eval_env.close()
        pass
