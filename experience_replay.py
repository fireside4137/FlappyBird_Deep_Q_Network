# For the memory of experiences, deque(double ended queue) is used

from collections import deque
import random

class ReplayMemory():
    # create FIFO queue
    def __init__(self, maxlen, seed = None):
        self.memory = deque([], maxlen = maxlen)

    def append(self, new_exp):
        # to add elements
        self.memory.append(new_exp)

    def sample(self, sample_size):
    # to extract random samples
        return random.sample(self.memory, sample_size)

    # current buffer size
    def __len__(self):
        return len(self.memory)