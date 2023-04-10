import math
import os
import shutil
from matplotlib import pyplot as plt
import centroid_initialization as cent_init
from group import Group
from sample import Sample


class Kmeans:
    def __init__(self, k, data: list, iteration):
        assert isinstance(data[0], Sample)
        self.data = data
        self.k = k
        self.groups = [Group(cen) for cen in cent_init.random(data, k)]
        self.vec_len = len(data[0])
        self.iteration = iteration
        self.__rate_hist = []

    def __assign(self):
        for vec in self.data:
            vec: Sample
            tmp_ls = [(gp, gp.dis(vec)) for gp in self.groups]
            best_gp = min(tmp_ls, key=lambda tup: tup[1])[0]
            best_gp.add(vec)

    def __move(self):
        for g in self.groups:
            g.move_center()

    def train(self):
        for i in range(self.iteration):
            self.__flush_groups()
            self.__assign()
            self.__cal_rate()
            self.__move()

        return self.groups

    def __cal_rate(self):
        for gp in self.groups:
            gp.update_sz_hist()

        change_sum = sum([g.get_sz_change() for g in self.groups])
        self.__rate_hist.append(change_sum / 2)

    def __flush_groups(self):
        for g in self.groups:
            g.flush()

    def gp_sz_hist(self):
        clrs = ['r', 'b', 'g', 'm', 'k']
        clrs = clrs * math.ceil(self.k / (len(clrs)))
        for i, gp in enumerate(self.groups):
            plt.plot(gp.sz_hist, clrs[i])
        plt.pause(0)

    def last_ten_rate(self):
        print(self.__rate_hist[-10:])

    def save_as_img(self, img_row, smpl_count):
        res_dir = 'result'
        if os.path.isdir(res_dir):
            shutil.rmtree(res_dir)
        os.mkdir(res_dir)

        for i, gp in enumerate(self.groups):
            gp_dir = "{}/{} size:{}".format(res_dir, i, len(self.groups[i]))
            os.mkdir(gp_dir)
            plt.imsave("{}/center.jpeg".format(gp_dir), self.groups[i].center.reshape(img_row, -1), cmap='gray')
            for j in range(smpl_count):
                plt.imsave("{}/{}.jpeg".format(gp_dir, j), self.groups[i].members[j].reshape(img_row, -1), cmap='gray')

    def show_nearest(self, smp_count):
        for i, gp in enumerate(self.groups):
            print("{} size: {}".format(i, len(gp)))
            print(gp.center)
            gp.members.sort(key=lambda x: gp.dis(x))
            tmp_smp_count = min(gp.get_sz(), smp_count)
            for mem in gp.members[:tmp_smp_count]:
                print(mem.smp_id)

    def show_conv(self):
        plt.plot(self.__rate_hist)
        plt.pause(0)
