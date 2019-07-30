from numpy import * #导入numpy科学计算包
import operator     #导入运算符模块

#加载数据的方法，返回样本数据（每一行是一个样本）和样本标签
def createDataSet():
    group = array([[90,100],[88,90],[85,95],[10,20],[30,40],[50,30]])    #样本点数据
    labels = ['A','A','A','D','D','D']

    print("Points in class A:")
    print(group[:3, :])
    print("Points in class D:")
    print(group[3:, :])

    return group,labels

#分类方法  传入的dataset需是array数组
def classify0(inX, dataSet, labels, k):  #inX为输入样本，例如[85,90]
    dataSetSize = dataSet.shape[0]   #求出输入数据矩阵的行数（样本个数）
    diffMat = tile(inX, (dataSetSize,1)) - dataSet  #求矩阵差
    sqDiffMat = diffMat ** 2
    sqDistance = sqDiffMat.sum(axis = 1)  #平方和
    distance = sqDistance ** 0.5          #测试样本点距离每个样本点的距离
    sortedDistance = distance.argsort()     #将距离按升序排列
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistance[i]]      #遍历前k个样本的标签
        classCount[voteLabel] = classCount.get(voteLabel,0) + 1  #对标签进行计数，即每一类出现的次数
        sortedClassCount = sorted(classCount.items(),key = operator.itemgetter(1),reverse = True)  #将计数后的标签按降序进行排列，得到元组列表
    return sortedClassCount[0][0]

if __name__ == '__main__':
    # 创建数据集
    group,labels = createDataSet()
    # 测试集
    test = [85, 90]
    # kNN分类
    test_class = classify0(test, group, labels, 6)
    # 打印分类结果
    print("Point {} belongs to class {}".format(test, test_class))
