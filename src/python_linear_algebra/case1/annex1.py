from numpy import *

STATES = ['B', 'M', 'E', 'S']  # B:标记词的开头；E:标记词的结尾；M:标记词的中间部分：S:标记单字词
array_A = {}  # 状态转移概率矩阵
array_B = {}  # 发射概率矩阵
array_Pi = {}  # 初始状态分布
count_dic = {}  # ‘B,M,E,S’每个状态在训练集中出现的次数
line_num = 0  # 训练集语句数量


# 初始化所有概率矩阵和状态计数数组
def Init_Array():
    for state0 in STATES:
        array_A[state0] = {}
        for state1 in STATES:
            array_A[state0][state1] = 0.0
    for state in STATES:
        array_Pi[state] = 0.0
        array_B[state] = {}
        count_dic[state] = 0


"""
获取词语word的状态标签,eg:word="天安门",得到tag=[B M E]
输入：
    word:待标记词语:string
输出：
    tag:word对应的状态标记：[]
"""
def get_tag(word):
    tag = []
    if len(word) == 1:
        tag = ['S']
    elif len(word) == 2:
        tag = ['B', 'E']
    else:
        num = len(word) - 2
        tag.append('B')
        tag.extend(['M'] * num)
        tag.append('E')
    return tag


"""
计算array_Pi，array_A，array_B的各项概率的对数值
使用概率的对数，是为了将概率的连乘装换为对数概率的加和，防止下溢
对于概率为0的项，已-3.14e+100表示其对数概率值
"""
def Prob_Array():
    for stat in array_Pi:
        if array_Pi[stat] == 0:
            array_Pi[stat] = -3.14e+100
        else:
            array_Pi[stat] = log(array_Pi[stat] / line_num)
    for stat in array_A:
        for next_stat in array_A[stat]:
            if array_A[stat][next_stat] == 0.0:
                array_A[stat][next_stat] = -3.14e+100
            else:
                array_A[stat][next_stat] = log(array_A[stat][next_stat] / count_dic[stat])
    for stat in array_B:
        for word in array_B[stat]:
            if array_B[stat][word] == 0.0:
                array_B[stat][word] = -3.14e+100
            else:
                array_B[stat][word] = log(array_B[stat][word] / count_dic[stat])
    # 对于训练集中未出现过的观测字符，以'other'表示，
    # 并将其发射概率对数值设为当前状态所有观测字符的发射概率对数值的平均值
    for stat in array_B:
        Sum = sum(list(array_B[stat].values()))
        array_B[stat]['other'] = Sum / len(array_B[stat])


"""
Viterbi算法求当前待分词句子最优状态序列.
输入：
    sentence:待分词句子:string
    array_pi：训练好的初始状态分布矩阵:{}
    array_a：训练好的状态转移概率矩阵:{}
    array_b：训练好的状态发射概率矩阵:{}
输出：
    已分词语句：string
"""
def Viterbi(sentence, array_pi, array_a, array_b):
    tab = [{}]  # 动态规划表
    path = {}

    # 若训练集中没有以当前句子第一个字符为首的词，则视其为单字词，并调整相应发射概率
    if sentence[0] not in array_b['B']:
        for state in STATES:
            if state == 'S':
                array_b[state][sentence[0]] = 0  # 等价于发射概率为1
            else:
                array_b[state][sentence[0]] = -3.14e+100  # 等价于发射概率为0
    # 计算
    for state in STATES:
        tab[0][state] = array_pi[state] + array_b[state][sentence[0]]
        path[state] = [state]
    for i in range(1, len(sentence)):
        tab.append({})
        new_path = {}
        for state_cur in STATES:
            items = []
            for state_pre in STATES:
                if sentence[i] not in array_b[state_cur]:  # 所有在测试集出现但没有在训练集中出现的字符
                    prob = tab[i - 1][state_pre] + array_a[state_pre][state_cur] + array_b[state_cur]['other']
                else:  # 计算每个字符对应STATES的概率
                    prob = tab[i - 1][state_pre] + array_a[state_pre][state_cur] + array_b[state_cur][sentence[i]]
                items.append((prob, state_pre))
            # 计算当前时刻i、当前状态state_cur下，概率prob最大的路径的概率值，以及此路径前一时刻的状态
            best = max(items)  # bset:(prob,state_pre)
            # 记录当前时刻i、当前状态state_cur下，概率prob最大的路径的概率值
            tab[i][state_cur] = best[0]
            # 记录当前时刻i、当前状态state_cur下，概率prob最大的路径
            new_path[state_cur] = path[best[1]] + [state_cur]
        path = new_path

    prob, state = max([(tab[len(sentence) - 1][state], state) for state in STATES])
    return path[state]


"""
根据状态序列进行分词
输入：
    sentence:待分词句子：string
    tag:待分词句子对应的状态标签：[]
输出：
    sentence分词得到的词语列表：[]
"""
def tag_seg(sentence, tag):
    word_list = []
    start = -1
    started = False

    if len(tag) != len(sentence):
        return None

    if len(tag) == 1:
        word_list.append(sentence[0])  # 语句只有一个字，直接输出

    else:
        if tag[-1] == 'B' or tag[-1] == 'M':  # 最后一个字状态不是'S'或'E'则修改
            if tag[-2] == 'B' or tag[-2] == 'M':
                tag[-1] = 'E'
            else:
                tag[-1] = 'S'

        for i in range(len(tag)):
            if tag[i] == 'S':
                if started:
                    started = False
                    word_list.append(sentence[start:i])
                word_list.append(sentence[i])
            elif tag[i] == 'B':
                if started:
                    word_list.append(sentence[start:i])
                start = i
                started = True
            elif tag[i] == 'E':
                started = False
                word = sentence[start:i + 1]
                word_list.append(word)
            elif tag[i] == 'M':
                continue

    return word_list


if __name__ == '__main__':
    trainset = open('CTBtrainingset.txt', encoding='utf-8')  # 读取训练集
    testset = open('CTBtestingset.txt', encoding='utf-8')  # 读取测试集

    Init_Array()  # 初始化所有概率矩阵和状态计数数组

    """ 
    逐行处理训练语料
    按行读取训练数据，并去除每行的首尾空格
    例如读取第一行句子，“上海 浦东 开发 与 法制 建设 同步”
    """
    for line in trainset:
        line = line.strip()
        line_num += 1

        # 将当前句子处理成字符列表存入word_list
        word_list = []
        for k in range(len(line)):
            if line[k] == ' ':
                continue
            word_list.append(line[k])  # [上 海 浦 东 开 发 与 法 制 建 设 同 步]

        line = line.split(' ')  # [上海 浦东 开发 与 法制 建设 同步]
        line_state = []  # 当前句子的状态序列

        for i in line:
            line_state.extend(get_tag(i))  # [B E B E B E S B E B E B E]
        array_Pi[line_state[0]] += 1  # array_Pi用于计算初始状态分布概率

        for j in range(len(line_state) - 1):
            array_A[line_state[j]][line_state[j + 1]] += 1  # array_A计算状态转移概率

        for p in range(len(line_state)):
            count_dic[line_state[p]] += 1  # 记录每一个状态的出现次数
            for state in STATES:
                if word_list[p] not in array_B[state]:
                    array_B[state][word_list[p]] = 0.0  # 保证每个字都在STATES的字典中
            array_B[line_state[p]][word_list[p]] += 1  # array_B用于计算发射概率

    # 计算array_Pi, array_A, array_B
    Prob_Array()

    outputfile = open('output.txt', mode='w', encoding='utf-8')  # 分词结果输出文件
    # 逐行读取测试文本进行分词
    for line in testset:
        line = line.strip()
        # 根据维特比算法，预测当前句子的BMES标签
        tag = Viterbi(line, array_Pi, array_A, array_B)
        # 根据标签状态对当前句子进行分词，获得分词列表
        seg = tag_seg(line, tag)
        print(seg)
        # 将分词列表中的词语，依序用空格拼接成字符串，并追加到结果文件末尾
        output_line = ''
        for i in range(len(seg)):
            output_line = output_line + seg[i] + ' '
        output_line = output_line + '\n'
        outputfile.write(output_line)
        print(output_line)
    # 关闭打开的文件
    trainset.close()
    testset.close()
    outputfile.close()
