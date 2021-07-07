from do_not_touch.MDPEnvData import *
from do_not_touch import get_dll
from do_not_touch.bytes_wrapper import get_bytes
from do_not_touch.contracts import MDPEnv


class SecretMDPEnv(MDPEnv):
    def __init__(self, env):
        self.env = env
        self.data_ptr = get_dll().get_mdp_env_data(self.env)
        self.data_bytes = get_bytes(self.data_ptr)
        self.data = MDPEnvData.GetRootAsMDPEnvData(self.data_bytes, 0)

    def states(self) -> np.ndarray:
        return self.data.StatesAsNumpy()

    def actions(self) -> np.ndarray:
        return self.data.ActionsAsNumpy()

    def rewards(self) -> np.ndarray:
        return self.data.RewardsAsNumpy()

    def is_state_terminal(self, s: int) -> bool:
        return get_dll().mdp_env_is_state_terminal(self.env, s)

    def transition_probability(self, s: int, a: int, s_p: int, r: float) -> float:
        return get_dll().mdp_env_transition_probability(self.env, s, a, s_p, r)

    def view_state(self, s: int):
        print("It's secret !")

    def __del__(self):
        get_dll().free_wrapped_bytes(self.data_ptr)
        get_dll().delete_mdp_env(self.env)


class Env1(SecretMDPEnv):
    def __init__(self):
        super().__init__(get_dll().create_secret_env1())
