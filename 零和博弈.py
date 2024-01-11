# -*- encoding:utf-8 -*-

import numpy as np


class Player(object):
    def __init__(self, policy_len, utility, num):
        """
        :param policy_len: 策略个数
        :param utility:  收益矩阵
        :param num: 玩家0写0 玩家1写1
        """
        self.utility = utility  # 设置自己收益矩阵
        self.policy_len = policy_len  # 设置策略个数
        self.policy = np.random.random(self.policy_len)  # 设置初始策略
        self.history = np.zeros(self.policy_len)  # 设置历史策略（累计）
        self.num = num  # 设置玩家编号

    def change_policy(self, op_pro):
        """
        根据传入的对手历史策略，选择自己的最优策略，并改变自己的策略
        :param op_pro: 对手策略
        """
        earn = op_pro * self.utility  # 收益 = 对手策略 * 自己收益矩阵
        money_sum = np.sum(earn, axis=1 - self.num)  # axis = 0: 行；axis = 1 : 列
        best_choice = np.argmax(money_sum)  # argmax返回一个numpy数组中最大值的索引值
        self.history[best_choice] += 1
        self.policy = self.history / np.sum(self.history)

    def get_policy(self):
        """
        :return: 返回自己本轮策略
        """
        if self.num == 0:
            return np.reshape(self.policy, (self.policy_len, 1))
        else:
            return self.policy

    def exploitability(self, op_pro):
        """
        测试对手策略的可利用度（实质就是epsilon-纳什均衡的epsilon）
        :param op_pro: 对手策略
        """
        earn = op_pro * self.utility  # 收益 = 对手策略 * 自己收益矩阵
        money_sum = np.sum(earn, axis=1 - self.num)  # axis = 0: 行；axis = 1 : 列
        best_choice = np.argmax(money_sum)
        print('p' + str(1 - self.num) + ' exploitability:', money_sum[best_choice])


class Nash(object):
    def __init__(self, p0, p1):
        self.p0 = p0
        self.p1 = p1

    def get_nash_equilibrium(self, loop_time):
        """
        求解纳什均衡
        :param loop_time: 迭代次数
        """
        for i in range(loop_time):
            self.p0.change_policy(self.p1.get_policy())
            self.p1.change_policy(self.p0.get_policy())

    def show_result(self):
        """
        显示结果
        """
        print('p0', self.p0.get_policy())
        print('p1', self.p1.get_policy())

    def show_exploitability(self):
        """
        显示可利用度
        """
        p0.exploitability(self.p1.get_policy())
        p1.exploitability(self.p0.get_policy())


if __name__ == '__main__':

    # 以下例子求解如下纳什均衡（囚徒困境）
    #     P0╲P1    坦白    抵赖
    #      坦白   -4，-4   0，-5
    #      抵赖   -5， 0  -1，-1

    u0 = np.array(
        [[-4, 0],
         [-5, -1]]
    )
    u1 = np.array(
        [[-4, -5],
         [0, -1]]
    )
    p0 = Player(2, u0, 0)
    p1 = Player(2, u1, 1)

    # # 以下例子求解如下纳什均衡（石头剪刀布）
    # #     P0╲P1    石头    剪刀    布
    # #      石头    0, 0   1,-1  -1, 1
    # #      剪刀   -1, 1   0, 0   1,-1
    # #       布     1,-1  -1, 1   0, 0
    #
    # u0 = np.array(
    #     [[0, 1, -1],
    #      [-1, 0, 1],
    #      [1, -1, 0]]
    # )
    # u1 = np.array(
    #     [[0, -1, 1],
    #      [1, 0, -1],
    #      [-1, 1, 0]]
    # )
    # p0 = Player(3, u0, 0)
    # p1 = Player(3, u1, 1)

    nash = Nash(p0, p1)
    nash.get_nash_equilibrium(1000)
    nash.show_result()
    nash.show_exploitability()
