import tensorflow as tf
import numpy as np
from gym import spaces

from pysc2.env import mock_sc2_env


class ObservationConverter:

    def __init__(self, env_settings, action_converter):
        super().__init__()

        self._env_settings = env_settings
        self._action_converter = action_converter
        self._action_space = action_converter.make_action_space()
        self._complete_obs_space = None

        self.MAX_AVAILABLE_ACTIONS = 10  # may make larger
        self.MAX_ACTION_ID = 10  # may make larger
        self.MAX_ENTITIES = 30  # may make larger
        self._entity_len = 0

        # depth of one_hot encodings
        self._player_info_space = {
            'mineral_depth': 19,
            'vespene_depth': 26,
            'food_used_depth': 200,
            'food_cap_depth': 200
        }

        # caching, since tf.one_hot is relatively expensive
        self._cached_action_one_hots = []
        for action_id in range(self.MAX_ACTION_ID):
            self._cached_action_one_hots.append(tf.one_hot(action_id, depth=self.MAX_ACTION_ID, dtype=np.int32))

    def _make_obs_space(self):
        # Note: might be out of spec easily, e.g. high is set to 1000, I do not know what the actual high can be
        box_lower_bound = 0
        box_higher_bound = 1000

        # mocking the sc2_env since we are making the observation space before the env is even created,
        # -> therefore we mock it with the same settings
        mock_env = mock_sc2_env.SC2TestEnv(**self._env_settings)
        timesteps = mock_env.reset()
        ts = timesteps[0]

        general_info = self.encode_general_info(player_info=ts.observation['player'],
                                                available_actions=ts.observation['available_actions'],
                                                last_actions=ts.observation['last_actions'])
        general_space = spaces.MultiBinary(general_info.size)

        encoded_screen = self.encode_screen(ts.observation['feature_screen'])
        screen_space = spaces.Box(low=box_lower_bound, high=box_higher_bound, shape=encoded_screen.shape,
                                  dtype=np.float32)

        encoded_minimap = self.encode_minimap(ts.observation['feature_minimap'])
        minimap_space = spaces.Box(low=box_lower_bound, high=box_higher_bound, shape=encoded_minimap.shape,
                                   dtype=np.float32)

        encoded_entities = self.encode_feature_units(ts.observation['feature_units'])
        entities_space = spaces.Box(low=box_lower_bound, high=box_higher_bound, shape=encoded_entities.shape,
                                    dtype=np.float32)

        space = spaces.Dict(
            {'info': general_space,
             'screen': screen_space,
             'minimap': minimap_space,
             'entities': entities_space
             })

        mock_env.close()  # might do nothing
        return space

    def encode_ts(self, ts):

        general_info = self.encode_general_info(player_info=ts.observation['player'],
                                                available_actions=ts.observation['available_actions'],
                                                last_actions=ts.observation['last_actions'])
        encoded_screen = self.encode_screen(ts.observation['feature_screen'])
        encoded_minimap = self.encode_minimap(ts.observation['feature_minimap'])
        encoded_entities = self.encode_feature_units(ts.observation['feature_units'])

        encoded_obs = {
            'info': general_info,
            'screen': encoded_screen,
            'minimap': encoded_minimap,
            'entities': encoded_entities
        }

        reward = ts.reward
        return encoded_obs, reward, ts.last(), {}  # {} is the 'info'

    def get_obs_space(self):
        if self._complete_obs_space is None:
            self._complete_obs_space = self._make_obs_space()
        return self._complete_obs_space

    ############ GENERAL PLAYER INFO ########################################

    def encode_general_info(self, player_info, available_actions, last_actions):
        player_info = self.encode_player_info(player_info)
        misc_info = self.encode_misc_info(available_actions, last_actions)
        return np.concatenate([player_info, misc_info], axis=0)

    # Encodes: minerals, vespene, food_used and food_cap
    def encode_player_info(self, player_info):
        current_minerals = tf.one_hot(int(player_info['minerals'] / 100),
                                      depth=self._player_info_space['mineral_depth'], dtype=np.int32)
        current_vespene = tf.one_hot(int(player_info['vespene'] / 100), depth=self._player_info_space['vespene_depth'],
                                     dtype=np.int32)

        food_used = tf.one_hot(int(player_info['food_used']), depth=self._player_info_space['food_used_depth'],
                               dtype=np.int32)
        food_cap = tf.one_hot(int(player_info['food_cap']), depth=self._player_info_space['food_cap_depth'],
                              dtype=np.int32)

        used_info = [current_minerals, current_vespene, food_used, food_cap]
        info = np.concatenate(used_info, axis=0)
        return info

    # Encodes:  Available Actions: all the actions that are currently possible
    #           Last Actions: all the actions that were made successfully since the last observation
    def encode_misc_info(self, available_actions, last_actions):

        empty_action = np.zeros(self.MAX_ACTION_ID, dtype=np.int32)
        actions_encoded = []
        iter_len = min(len(available_actions), self.MAX_AVAILABLE_ACTIONS)
        for action_id in available_actions[0:iter_len]:
            if action_id < self.MAX_ACTION_ID:
                # using cached values since this is done A LOT and takes some time
                actions_encoded.append(self._cached_action_one_hots[action_id])
            else:
                actions_encoded.append(empty_action)

        while len(actions_encoded) < self.MAX_AVAILABLE_ACTIONS:
            actions_encoded.append(empty_action)  # fill up the remaining slots with empty actions

        if len(last_actions) > 0:
            # since we only ever send one action between observations,
            # we are only interested in the first item here
            last_action_encoded = tf.one_hot(last_actions[0], depth=self.MAX_ACTION_ID, dtype=np.int32)
        else:
            # no successful action was made in the last step
            last_action_encoded = empty_action

        actions_encoded.append(last_action_encoded)
        misc_info = np.concatenate(actions_encoded, axis=0)
        return misc_info

    ############### FEATURE UNITS ##########################

    # Encodes basic information of every visible unit
    def encode_feature_units(self, feature_units):

        entities = []
        iter_len = min(len(feature_units), self.MAX_ENTITIES)
        for unit in feature_units[:iter_len]:

            unit_type = unit['unit_type']
            x = unit['x']
            y = unit['y']
            is_selected = unit['is_selected']  # 0 for no, 1 for yes

            alliance = unit['alliance']  # Self = 1, Ally = 2, Neutral = 3, Enemy = 4, Unknown = 0
            health_ratio = unit['health_ratio']  # 0 for dead, 255 for full hit-points

            entity = [unit_type, x, y, is_selected, alliance, health_ratio]
            if self._entity_len == 0:
                self._entity_len = len(entity)

            entities.append(entity)

        empty_entity = np.zeros(self._entity_len)
        while len(entities) < self.MAX_ENTITIES:
            entities.append(empty_entity)  # filling up the empty slots

        return np.stack(entities, axis=1)

############################ SPATIAL ##########################################

    # Encodes screen features such as the type of the unit or whether the unit is selected.
    # The screen is more or less a picture, where each layer represents something different,
    # similar to RGB layers in a normal picture.
    def encode_screen(self, screen):
        chosen_features = ['player_relative', 'unit_type', 'selected', 'unit_hit_points_ratio',
                           'visibility_map', 'build_progress', 'buildable']

        spatial_features = []
        for feature in chosen_features:
            spatial_features.append(screen[feature])
        return self._stack_spatial(spatial_features)

    # Encodes minimap features, same idea as encode_screen, but there are other features available here.
    def encode_minimap(self, minimap):
        chosen_features = ['player_relative', 'selected', 'unit_type', 'camera', 'visibility_map']

        spatial_features = []
        for feature in chosen_features:
            spatial_features.append(minimap[feature])
        return self._stack_spatial(spatial_features)

    def _stack_spatial(self, spatial_features):
        # tf-agents prefers images to be stacked this way.
        # e.g. shape (32, 32, 5) becomes (5, 32, 32)
        return np.stack(spatial_features, axis=2)
