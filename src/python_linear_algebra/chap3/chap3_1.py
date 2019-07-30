# -*- coding: utf-8 -*-

import numpy as np
x = np.array([3, 0, -4])
n1 = np.linalg.norm(x, ord = 1)  # ord = 1表示向量的1-范数
n2 = np.linalg.norm(x, ord = 2)  # ord = 2表示向量的2-范数，系统默认
n3 = np.linalg.norm(x, ord = np.inf)  # ord = np.inf表示向量的∞-范数
n4 = np.linalg.norm(x, ord = 0)  # ord = 0表示向量的0-范数
n5 = np.linalg.norm(x, ord = -np.inf)  # ord = -np.inf表示向量的-∞-范数
print('norm_1   ', n1)
print('norm_2   ', n2)
print('norm_∞   ', n3)
print('norm_0   ', n4)
print('norm_-∞   ', n5)
