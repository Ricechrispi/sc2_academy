from absl import logging
from absl import app
from absl import flags

from tf_agents.system import system_multiprocessing as multiprocessing

from ppo_trainer import train_eval
from ppo_eval import eval_only

FLAGS = flags.FLAGS

## SETUP-PARAMETERS ##

flags.DEFINE_string('map_name', 'MoveToBeacon', 'The name of the mini-game map to train on.')
flags.DEFINE_integer('num_parallel_envs', 4, 'How many parallel environments are used for collecting training data.')
flags.DEFINE_integer('num_eval_episodes', 10, 'Number of episodes to evaluate performance for. Final result is the average of these episodes.')
flags.DEFINE_bool('just_eval', False, 'If true, just evaluate the performance and no training is done. All training related arguments are ignored.')
flags.DEFINE_bool('just_eval_random', False, 'If true, just evaluate the performance of a random agent. All training related arguments are ignored.')
flags.DEFINE_bool('visualize_eval', False, 'If true, visualize the evaluation done by just_eval or just_eval_random')

# less important parameters, there is usually no need to change these
flags.DEFINE_integer('eval_interval', 66, 'After this many collect/train iterations the performance is evaluated')
flags.DEFINE_integer('summary_interval', 33,'After this many collect/train iterations, summaries of the training/eval is written to tensorboard event files.')
flags.DEFINE_integer('checkpoint_interval', 333, 'After this many collect/train iterations, the models are saved.')
flags.DEFINE_integer('log_interval', 33, 'After this many collect/train iterations, various stats about speed are logged.')
flags.DEFINE_integer('summaries_flush_secs', 1, 'The interval at which the summary outputs are flushed')
flags.DEFINE_bool('use_multiprocessing', True, 'Whether to use multiprocessing or not.')

## HYPER-PARAMETERS ##

# the steps the AGENT takes in the environment. This does not factor in the step_mul.
flags.DEFINE_integer('num_environment_steps', 2000000, 'Number of steps to train for in the environment.')
flags.DEFINE_integer('num_collect_episodes', 20, 'Number of episodes to collect data for per iteration.')

# 4000 is enough for everything using 4 parallel envs.
# EXCEPT for BuildMarines, that requires 10000. If using 10 envs, 4000 is enough again.
flags.DEFINE_integer('replay_buffer_max_length', 4000, 'The maximum length a replay buffer can be. Increases RAM requirements substantially.')

flags.DEFINE_integer('num_epochs', 3, 'Number of epochs to train the agent on the data. Recommend < 10.')
flags.DEFINE_float('learning_rate', 0.0005, 'Learning rate for the AdamOptimizer')
flags.DEFINE_bool('norm_rewards', True, 'Whether to normalize the rewards')
flags.DEFINE_bool('use_lstms', False, 'Whether to use an LSTM cell in the actor/value networks.')

def main(_):
    logging.set_verbosity(logging.INFO)

    # !! IMPORTANT: making sure these are a multiple of num_epochs of the PPO agent
    FLAGS.eval_interval = FLAGS.eval_interval * FLAGS.num_epochs
    FLAGS.summary_interval = FLAGS.summary_interval * FLAGS.num_epochs
    FLAGS.checkpoint_interval = FLAGS.checkpoint_interval * FLAGS.num_epochs
    FLAGS.log_interval = FLAGS.log_interval * FLAGS.num_epochs

    main_fn = train_eval
    if FLAGS.just_eval or FLAGS.just_eval_random:
        main_fn = eval_only

    if FLAGS.use_multiprocessing:
        multiprocessing.enable_interactive_mode()
        multiprocessing.handle_main(lambda: main_fn(FLAGS))
    else:
        main_fn(FLAGS)

if __name__ == '__main__':
    app.run(main)
