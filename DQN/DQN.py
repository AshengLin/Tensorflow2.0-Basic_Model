import tensorflow as tf
import numpy as np
from tensorflow.keras import layers
from maze_env import Maze


class DQN:
    def __init__(self):
        self.params = {}