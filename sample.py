import numpy as np


class Sample(np.ndarray):
    def __new__(cls, input_array, smp_id=None):
        obj = np.asarray(input_array).view(cls)
        obj.smp_id = smp_id
        return obj
