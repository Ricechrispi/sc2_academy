import tensorflow as tf

from tf_agents.networks import actor_distribution_network
from tf_agents.networks import actor_distribution_rnn_network
from tf_agents.networks import value_network
from tf_agents.networks import value_rnn_network

import ppo.my_ppo_agent as my_ppo_agent


def create_ppo_agent(env, global_step, FLAGS):

    actor_fc_layers = (512, 256)
    value_fc_layers = (512, 256)

    lstm_fc_input = (1024, 512)
    lstm_size = (256,)
    lstm_fc_output = (256, 256)

    minimap_preprocessing = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(filters=16,
                                kernel_size=(5, 5),
                                strides=(2, 2),
                                activation='relu'),
         tf.keras.layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                activation='relu'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(units=256, activation='relu')])

    screen_preprocessing = tf.keras.models.Sequential(
        [tf.keras.layers.Conv2D(filters=16,
                                kernel_size=(5, 5),
                                strides=(2, 2),
                                activation='relu'),
         tf.keras.layers.Conv2D(filters=32,
                                kernel_size=(3, 3),
                                strides=(2, 2),
                                activation='relu'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(units=256, activation='relu')])

    info_preprocessing = tf.keras.models.Sequential(
        [tf.keras.layers.Dense(units=128, activation='relu'),
         tf.keras.layers.Dense(units=128, activation='relu')])

    entities_preprocessing = tf.keras.models.Sequential(
        [tf.keras.layers.Conv1D(filters=4,
                                kernel_size=4,
                                activation='relu'),
         tf.keras.layers.Flatten(),
         tf.keras.layers.Dense(units=256, activation='relu')
         ]
    )

    actor_preprocessing_layers = {
        'minimap': minimap_preprocessing,
        'screen': screen_preprocessing,
        'info': info_preprocessing,
        'entities': entities_preprocessing,
    }
    actor_preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    if FLAGS.use_lstms:
        actor_net = actor_distribution_rnn_network.ActorDistributionRnnNetwork(
            env.observation_spec(),
            env.action_spec(),
            preprocessing_layers=actor_preprocessing_layers,
            preprocessing_combiner=actor_preprocessing_combiner,
            input_fc_layer_params=lstm_fc_input,
            output_fc_layer_params=lstm_fc_output,
            lstm_size=lstm_size)
    else:
        actor_net = actor_distribution_network.ActorDistributionNetwork(
            input_tensor_spec=env.observation_spec(),
            output_tensor_spec=env.action_spec(),
            preprocessing_layers=actor_preprocessing_layers,
            preprocessing_combiner=actor_preprocessing_combiner,
            fc_layer_params=actor_fc_layers,
            activation_fn=tf.keras.activations.tanh)

    value_preprocessing_layers = {
        'minimap': minimap_preprocessing,
        'screen': screen_preprocessing,
        'info': info_preprocessing,
        'entities': entities_preprocessing,
    }
    value_preprocessing_combiner = tf.keras.layers.Concatenate(axis=-1)

    if FLAGS.use_lstms:
        value_net = value_rnn_network.ValueRnnNetwork(
            env.observation_spec(),
            preprocessing_layers=value_preprocessing_layers,
            preprocessing_combiner=value_preprocessing_combiner,
            input_fc_layer_params=lstm_fc_input,
            output_fc_layer_params=lstm_fc_output,
            lstm_size=lstm_size)
    else:
        value_net = value_network.ValueNetwork(
            env.observation_spec(),
            preprocessing_layers=value_preprocessing_layers,
            preprocessing_combiner=value_preprocessing_combiner,
            fc_layer_params=value_fc_layers,
            activation_fn=tf.keras.activations.tanh)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    # commented out values are the defaults
    tf_agent = my_ppo_agent.PPOAgent(time_step_spec=env.time_step_spec(),
                                     action_spec=env.action_spec(),
                                     optimizer=optimizer,
                                     actor_net=actor_net,
                                     value_net=value_net,
                                     importance_ratio_clipping=0.1,
                                     # lambda_value=0.95,
                                     discount_factor=0.95,
                                     entropy_regularization=0.003,
                                     # policy_l2_reg=0.0,
                                     # value_function_l2_reg=0.0,
                                     # shared_vars_l2_reg=0.0,
                                     # value_pred_loss_coef=0.5,
                                     num_epochs=FLAGS.num_epochs,
                                     use_gae=True,
                                     use_td_lambda_return=True,
                                     normalize_rewards=FLAGS.norm_rewards,
                                     reward_norm_clipping=0.0,
                                     normalize_observations=True,
                                     # log_prob_clipping=0.0,
                                     # KL from here...
                                     # To disable the fixed KL cutoff penalty, set the kl_cutoff_factor parameter to 0.0
                                     kl_cutoff_factor=0.0,
                                     kl_cutoff_coef=0.0,
                                     # To disable the adaptive KL penalty, set the initial_adaptive_kl_beta parameter to 0.0
                                     initial_adaptive_kl_beta=0.0,
                                     adaptive_kl_target=0.00,
                                     adaptive_kl_tolerance=0.0,  # ...to here.
                                     # gradient_clipping=None,
                                     value_clipping=0.5,
                                     # check_numerics=False,
                                     # compute_value_and_advantage_in_train=True,
                                     # update_normalizers_in_train=True,
                                     # debug_summaries=False,
                                     # summarize_grads_and_vars=False,
                                     train_step_counter=global_step,
                                     # name='PPOClipAgent'
                                     )

    tf_agent.initialize()
    return tf_agent
