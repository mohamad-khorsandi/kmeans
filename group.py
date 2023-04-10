import numpy as np

from sample import Sample


class Group:
    def __init__(self, center):
        self.center = center
        self.members = []
        self.sz_hist = []

    def dis(self, vec: Sample) -> float:
        temp = vec - self.center
        sum_sq = np.dot(temp.T, temp)
        return np.sqrt(sum_sq)

    def add(self, vec):
        self.members.append(vec)

    def flush(self):
        self.members = []

    def __len__(self):
        return len(self.members)

    def get_sz(self):
        return len(self.members)

    def move_center(self):
        self.center = np.mean(self.members, axis=0)

    def update_sz_hist(self):
        self.sz_hist.append(len(self))

    def get_sz_change(self):
        if len(self.sz_hist) == 0:
            raise Exception("size history of this group is empty")
        elif len(self.sz_hist) == 1:
            return self.sz_hist[0]
        else:
            return abs(self.sz_hist[-1] - self.sz_hist[-2])
