import random

import numpy as np
import tensorflow as tf


def set_seed(seed: int = 42):
    """Set all relevant random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
