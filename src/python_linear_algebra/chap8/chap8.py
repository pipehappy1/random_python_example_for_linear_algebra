from scipy.optimize import linprog

# 需要优化的函数对应的参数list
c = [-4, -3]

# 不等式对应参数矩阵
A = [[2, 1], [1, 1], [0, 1]]

# 不等式对应的上界
b = [10, 8, 7]

#各参数的取值范围
x0_bounds = (0, None)
x1_bounds = (0, None)

res = linprog(c, A_ub=A, b_ub=b, bounds=(x0_bounds, x1_bounds),  options={"disp": True})
print(res)
