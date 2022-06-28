import numpy as np


class Arch_Nsas_Manager():
    def __init__(self, knn_num, select_num, recent_num, arch_list):
        self.knn_num = knn_num
        self.select_num = select_num
        self.recent_num = recent_num
        self.recent_arch = []
        self.select_arch = []
        self.arch_list = arch_list


    def cal_arch_distance(self, arch1, arch2):
        assert len(arch1) == 51 and len(arch2) == 51
        arch1_ = list(arch1)
        arch2_ = list(arch2)
        dis = 0

        for i, j in zip(arch1_[4:-1], arch2_[4:-1]):
            num_i = int(i)
            num_j = int(j)

            if num_i == 0:
                num_i = 9
            if num_j == 0:
                num_j = 9

            dis = dis + abs(num_i-num_j)

        dis = dis / 47.0
        return dis


    def find_similar_arch(self, arch):
        dis = np.zeros(len(self.select_arch))

        for i in range(len(self.select_arch)):
            dis[i] = self.cal_arch_distance(arch, self.select_arch[i])
        
        m = np.argsort(dis)
        index = m[0]
        return index
        #if len(arch1) != 51 or len(arch2) != 51: 
        #    assert()


    def cal_arch_score_knn(self, arch, arch_list):
        n = len(arch_list)
        dis = np.zeros(n)
        for i in range(n):
            dis[i] = self.cal_arch_distance(arch, arch_list[i])        

        sort_dis = np.sort(dis)

        diver_score = np.mean(sort_dis[0: self.knn_num])

        return diver_score


    def replace_recent_arch(self, index):
        arch_compar = self.select_arch[index]
        a = np.arange(0, index)
        b = np.arange(index+1, len(self.select_arch))
        index_remain = np.append(a, b)

        arch_archive_remain = [self.select_arch[j] for j in index_remain]

        ini_diver_score = self.cal_arch_score_knn(arch_compar, arch_archive_remain)

        #print('distance score {}!!!'.format(ini_diver_score))
        for i in range(len(self.recent_arch)):
            arch = self.recent_arch[i]
            diver_score = self.cal_arch_score_knn(arch, arch_archive_remain)
            if diver_score > ini_diver_score:
                arch_compar = arch
                ini_diver_score = diver_score

        return arch_compar


    def update_select_arch(self, arch):
        assert len(self.select_arch) == self.select_num

        index = self.find_similar_arch(arch)
        #print('update select_arch index {}!!!'.format(index))
        new_arch = self.replace_recent_arch(index)
        if new_arch != self.select_arch[index]:
            #print('update select_arch from recent_arch!!!')
            self.select_arch[index] = new_arch
            #print(self.select_arch)
            #print(arch_nsas.select_arch)


    def update_recent_arch(self, arch):
        assert len(self.recent_arch) <= self.recent_num

        if len(self.recent_arch) == self.recent_num:
            self.recent_arch[0:self.recent_num-1] = self.recent_arch[1:self.recent_num]
            self.recent_arch[self.recent_num-1] = arch
        elif len(self.recent_arch) < self.recent_num:
            self.recent_arch.append(arch)


    def set_select_arch(self, arch_list):
        #assert len(arch_list) >= self.select_num
        self.select_arch.clear()

        index_list = np.random.randint(low=0, high=len(arch_list), size=self.select_num)
        for j in index_list:
            self.select_arch.append(arch_list[j])

        assert len(arch_list) >= self.select_num