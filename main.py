import os
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = "hide"
os.add_dll_directory("D:\\Program Files (x86)\\Nvidia\\bin")
import to_do.dynamic_programming as dynamic_programming
import to_do.monte_carlo_methods as monte_carlo_methods
import to_do.deep_reinforcement_learning as deep_reinforcement_learning
import to_do.temporal_difference_learning as temporal_difference_learning

if __name__ == "__main__":
    #dynamic_programming.demo()
    monte_carlo_methods.demo()
    #temporal_difference_learning.demo()
    #deep_reinforcement_learning.demo()
