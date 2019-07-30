from numpy import *
from matplotlib import pyplot as plt

'''加载数据集'''
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    # datArr = [map(float,line) for line in stringArr]
    datArr = [list(map(float,line)) for line in stringArr]
    # 注意这里和python2的区别，需要在map函数外加一个list()，否则显示结果为 map at 0x3fed1d0
    return mat(datArr)

'''
    pca算法
    cov协方差=[(x1-x均值)*(y1-y均值)+(x2-x均值)*(y2-y均值)+...+(xn-x均值)*(yn-y均值)]/(n-1)
    Args:
        dataMat   原数据集矩阵
        topNfeat  应用的N个特征
    Returns:
        lowDDataMat  降维后数据集
        reconMat     新的数据集空间
'''
def pca(dataMat, topNfeat=9999999):
    # 计算每一列的均值
    meanVals = mean(dataMat, axis=0)
    # print('meanVals', meanVals)
    # 每个向量同时都减去均值
    meanRemoved = dataMat - meanVals
    # print('meanRemoved=', meanRemoved)
    # rowvar=0，传入的数据一行代表一个样本，若非0，传入的数据一列代表一个样本
    covMat = cov(meanRemoved, rowvar=0)
    # eigVals为特征值， eigVects为特征向量
    eigVals, eigVects = linalg.eig(mat(covMat))
    # print('eigVals=', eigVals)
    # print('eigVects=', eigVects)

    # 对特征值，进行从小到大的排序，返回从小到大的index序号
    # 特征值的逆序就可以得到topNfeat个最大的特征向量
    eigValInd = argsort(eigVals)
    # print('eigValInd1=', eigValInd)
    # -1表示倒序，返回topN的特征值[-1到-(topNfeat+1)不包括-(topNfeat+1)]
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    # print('eigValInd2=', eigValInd)
    # 重组 eigVects 最大到最小
    redEigVects = eigVects[:, eigValInd]
    # print('redEigVects=', redEigVects.T)

    # 将数据转换到新空间
    # print( "---", shape(meanRemoved), shape(redEigVects))
    lowDDataMat = meanRemoved * redEigVects
    reconMat = (lowDDataMat * redEigVects.T) + meanVals
    # print('lowDDataMat=', lowDDataMat)
    # print('reconMat=', reconMat)
    return lowDDataMat, reconMat

'''将数据集中NaN替换成平均值函数'''
def replaceNanWithMean():
    datMat = loadDataSet('secom.data', ' ')
    numFeat = shape(datMat)[1]
    for i in range(numFeat):
        # 对value不为NaN的求均值
        # .A 返回矩阵基于的数组
        meanVal = mean(datMat[nonzero(~isnan(datMat[:, i].A))[0], i])
        # 将value为NaN的值赋值为均值
        datMat[nonzero(isnan(datMat[:, i].A))[0],i] = meanVal
    return datMat

'''降维后的数据和原始数据可视化'''
def show_picture(dataMat, reconMat):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(dataMat[:, 0].flatten().A[0], dataMat[:, 1].flatten().A[0], marker='^', s=90,c='green')
    ax.scatter(reconMat[:, 0].flatten().A[0], reconMat[:, 1].flatten().A[0], marker='o', s=50, c='red')
    plt.show()

'''分析数据'''
def analyse_data(dataMat):
    datMat = replaceNanWithMean()
    meanVals = mean(dataMat, axis=0)
    meanRemoved = dataMat-meanVals
    covMat = cov(meanRemoved, rowvar=0)
    eigvals, eigVects = linalg.eig(mat(covMat))
    eigValInd = argsort(eigvals)

    topNfeat = 20
    eigValInd = eigValInd[:-(topNfeat+1):-1]
    cov_all_score = float(sum(eigvals))
    sum_cov_score = 0
    for i in range(0, len(eigValInd)):
        line_cov_score = float(eigvals[eigValInd[i]])
        sum_cov_score += line_cov_score
        print('主成分：%s, 方差占比：%s%%, 累积方差占比：%s%%' % (format(i+1, '2.0f'), format(line_cov_score/cov_all_score*100, '4.2f'), format(sum_cov_score/cov_all_score*100, '4.1f')))
    

    # 2 主成分分析降维特征向量设置
    lowDmat, reconMat = pca(dataMat, 20)
    print(shape(lowDmat))
    # 3 将降维后的数据和原始数据一起可视化
    show_picture(dataMat, reconMat)

if __name__ == "__main__":
    datMat = replaceNanWithMean()
    analyse_data(datMat)
