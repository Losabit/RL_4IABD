from do_not_touch.SingleAgentEnvStateData import *

from do_not_touch.bytes_wrapper import *
from do_not_touch import get_dll
from do_not_touch.contracts import SingleAgentEnv


class SecretSingleAgentEnv(SingleAgentEnv):
    def __init__(self, env):
        self.env = env
        self.data_ptr = get_dll().get_single_agent_env_state_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = SingleAgentEnvStateData.GetRootAsSingleAgentEnvStateData(self.data_bytes, 0)

    def state_id(self) -> int:
        return self.data.StateId()

    def is_game_over(self) -> bool:
        return self.data.IsGameOver()

    def act_with_action_id(self, action_id: int):
        get_dll().act_on_single_agent_env(self.env, action_id)
        get_dll().free_wrapped_bytes(self.data_ptr)
        self.data_ptr = get_dll().get_single_agent_env_state_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = SingleAgentEnvStateData.GetRootAsSingleAgentEnvStateData(self.data_bytes, 0)

    def score(self) -> float:
        return self.data.Score()

    def available_actions_ids(self) -> np.ndarray:
        return self.data.AvailableActionsIdsAsNumpy()

    def reset(self):
        get_dll().reset_single_agent_env(self.env)
        get_dll().free_wrapped_bytes(self.data_ptr)
        self.data_ptr = get_dll().get_single_agent_env_state_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = SingleAgentEnvStateData.GetRootAsSingleAgentEnvStateData(self.data_bytes, 0)

    def view(self):
        print("It's secret !")

    def reset_random(self):
        get_dll().reset_random_single_agent_env(self.env)
        get_dll().free_wrapped_bytes(self.data_ptr)
        self.data_ptr = get_dll().get_single_agent_env_state_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = SingleAgentEnvStateData.GetRootAsSingleAgentEnvStateData(self.data_bytes, 0)

    def __del__(self):
        get_dll().free_wrapped_bytes(self.data_ptr)
        get_dll().delete_single_agent_env(self.env)


class Env2(SecretSingleAgentEnv):
    def __init__(self):
        super().__init__(get_dll().create_secret_env2())


class Env3(SecretSingleAgentEnv):
    def __init__(self):
        super().__init__(get_dll().create_secret_env3())
