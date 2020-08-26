from pysc2.env import sc2_env
from pysc2.env import mock_sc2_env
from pysc2.lib import features
from pysc2.lib import actions

from absl import logging
from gym.core import Env

from envs.action_converter import ActionConverter
from envs.action_converter import ActionSet
from envs.observation_converter import ObservationConverter


class SpaceConverter:

    def __init__(self, action_set, action_spec, observation_spec, env_settings, screen_dim):

        self.action_spec = action_spec
        self.observation_spec = observation_spec
        self.action_space = None
        self.observation_space = None

        self._action_converter = ActionConverter(action_set=action_set,
                                                 action_spec=self.action_spec,
                                                 screen_dim=screen_dim)
        self._obs_converter = ObservationConverter(env_settings=env_settings, action_converter=self._action_converter)

    def get_action_space(self):
        if self.action_space is None:
            self.action_space = self._action_converter.make_action_space()
        return self.action_space

    def get_observation_space(self):
        if self.observation_space is None:
            self.observation_space = self._obs_converter.get_obs_space()
        return self.observation_space

    def action_space_to_fun(self, space):
        # converts given value in action space to usable function for PySC2
        return self._action_converter.space_to_function(space)

    def ts_to_obs_space(self, ts):
        # converts a TimeStep to an observation
        return self._obs_converter.encode_ts(ts)


class SC2GymEnv(Env):

    def __init__(self, map_name='MoveToBeacon', visualize=False, screen_dim=32, minimap_dim=32, mock=False):
        super().__init__()

        self.settings = {
            'map_name': map_name,
            'players': [sc2_env.Agent(sc2_env.Race.terran)],  # true for all mini-games
            'agent_interface_format': features.AgentInterfaceFormat(
                feature_dimensions=features.Dimensions(screen=screen_dim, minimap=minimap_dim),
                use_feature_units=True),
            'step_mul': 8,  # how many game steps pass between actions; default is 8, which is 300APM, 16 means 150APM
            'game_steps_per_episode': 0,  # the fixed length of a game, if 0: as long as needed
            'visualize': visualize,  # whether to draw the game
        }

        # see https://github.com/deepmind/pysc2/blob/master/docs/mini_games.md
        if map_name == "MoveToBeacon":
            # Fog of War disabled
            # No camera movement required (single-screen)
            action_set = ActionSet.Select_Army_Move_2D
        elif map_name == "CollectMineralShards":
            # Fog of War disabled
            # No camera movement required (single-screen)
            # action_set = ActionSet.Select_Army_Move_2D
            action_set = ActionSet.Select_Multi_Move_2D
        elif map_name == "FindAndDefeatZerglings":
            # Fog of War enabled
            # Camera movement required (map is larger than single-screen)
            # action_set = ActionSet.Select_Army_Attack_2D
            action_set = ActionSet.Attack_2D_Move_Camera
        elif map_name == "DefeatRoaches":
            # Fog of War disabled
            # No camera movement required (single-screen)
            # action_set = ActionSet.Select_Army_Attack_2D
            action_set = ActionSet.Select_Multi_Move_Attack_2D
        elif map_name == "DefeatZerglingsAndBanelings":
            # Fog of War disabled
            # No camera movement required (single-screen)
            # action_set = ActionSet.Select_Army_Attack_2D
            action_set = ActionSet.Select_Multi_Move_Attack_2D
        elif map_name == "CollectMineralsAndGas":
            # Fog of War disabled
            # No camera movement required (single-screen)
            action_set = ActionSet.Build_SCVs
        elif map_name == "BuildMarines":
            # Fog of War disabled
            # No camera movement required (single-screen)
            action_set = ActionSet.Build_Marines
        else:
            raise ValueError("map is not supported")

        self.mock = mock
        self._wrapped_env = self._init_env()

        self._space_converter = SpaceConverter(action_set=action_set,
                                               action_spec=self._wrapped_env.action_spec(),
                                               observation_spec=self._wrapped_env.observation_spec(),
                                               env_settings=self.settings,
                                               screen_dim=screen_dim)

        self.action_space = self._space_converter.get_action_space()
        self.observation_space = self._space_converter.get_observation_space()

        self._cur_timestep = None

    def _init_env(self):
        if self.mock:
            env = mock_sc2_env.SC2TestEnv(**self.settings)
        else:
            env = sc2_env.SC2Env(**self.settings)
        self._wrapped_env = env
        return env

    def _can_do(self, ts, action):
        # True if action can be performed in the sc2_env
        return ts is not None \
               and action.function in [actions.FUNCTIONS[f].id for f in ts.observation.available_actions]

    def step(self, action, step_mul=None):
        # action is in action_space, sc2_action is in action_spec
        sc2_action = self._space_converter.action_space_to_fun(action)

        if not self._can_do(self._cur_timestep, sc2_action):
            logging.debug('Got action: ' + str(action) + '. Had to reset to noop!')
            sc2_action = actions.FUNCTIONS.no_op()

        # this returns a tuple of TimeSteps, one per agent
        timesteps = self._wrapped_env.step([sc2_action], step_mul)
        ts = timesteps[0]
        self._cur_timestep = ts

        obs, reward, done, info = self._space_converter.ts_to_obs_space(ts)
        return obs, reward, done, info

    def reset(self):
        if self._wrapped_env is None:
            self._init_env()
        timesteps = self._wrapped_env.reset()
        ts = timesteps[0]
        self._cur_timestep = ts
        obs, reward, done, info = self._space_converter.ts_to_obs_space(ts)
        return obs

    def render(self, mode='human'):
        # NOTE: does not do anything
        pass

    def close(self):
        if self._wrapped_env is not None:
            self._wrapped_env.close()
        super().close()

    def seed(self, seed=None):
        # NOTE: does not do anything
        return super().seed(seed)
