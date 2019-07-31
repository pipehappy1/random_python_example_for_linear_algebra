'''
主成份分析  principal component analysis  PCA
先求解 协方差矩阵
再求解 协方差矩阵的特征值和特征向量
'''
import numpy as np
import matplotlib

#  载入文件数据   文件名  分隔符
def loadDataSet(fileName, delim='\t'):
    fr = open(fileName)
    stringArr = [line.strip().split(delim) for line in fr.readlines()]
    datArr = [list(map(float,line)) for line in stringArr]  # 转换成浮点型列表
    return np.mat(datArr)  # 转换成 numpy 下的数组mat类型
 
def pca(dataMat, topNfeat=9999999):
    meanVals = np.mean(dataMat, axis=0)  # 每个样本的平均值   一行
    meanRemoved = dataMat - meanVals  # 减去平均值
    covMat = np.cov(meanRemoved, rowvar=0)  # 计算协方差矩阵
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))  # 计算协方差矩阵的特征值和特征向量
    eigValInd = np.argsort(eigVals)                          # 特征值从小到大排列
    eigValInd = eigValInd[:-(topNfeat+1):-1]      # 前topNfeat个较大的特征值对应的 排序下标
    redEigVects = eigVects[:,eigValInd]                # 逆序排列得到前topNfeat个特征向量
    lowDDataMat = meanRemoved * redEigVects   # 将原去均值花后的数据 乘以特征向量转换到新 坐标空间
    reconMat = (lowDDataMat * redEigVects.T) + meanVals  #按照逆转换 到原来的空间数据
    return lowDDataMat, reconMat    # 降维后的数据，以及还原后的主成分数据
 
# 测试
def pca_test(file_name='testSet.txt'):
    import matplotlib
    import matplotlib.pyplot as plot
    datMat = loadDataSet(file_name)
    lowDMat, reconMat  = pca(datMat, 1)   # 降维成1维矩阵前一个主成份
    print("原数据维度: ")
    print(np.shape(datMat))
    print("降维后数据维度: ")
    print(shape(lowDMat))
    fig = plot.figure()
    ax = fig.add_subplot(111)
    ax.scatter(datMat[:,0].flatten().A[0], datMat[:,1].flatten().A[0],marker="^",s=90)
    ax.scatter(reconMat[:,0].flatten().A[0], reconMat[:,1].flatten().A[0],marker="o",s=50,c='red')
    plot.show()
 
# 处理缺失值  用均值代替
def replaceNanWithMean(): 
    datMat = loadDataSet('secom.data', ' ')   #半导体数据
    numFeat = np.shape(datMat)[1]   # 样本的特征维度
    for i in range(numFeat):
        meanVal = np.mean(datMat[np.nonzero(~np.isnan(datMat[:,i].A))[0], i])   # 某特征非缺失值的均值
        datMat[np.nonzero(np.isnan(datMat[:,i].A))[0],i] = meanVal        # 缺失值由上述均值替代
    return datMat
 
# 实验
def secomTest():
    datMat = replaceNanWithMean()
    meanVals = np.mean(datMat, axis=0)   # 每个样本的平均值   一行
    meanRemoved = datMat - meanVals   # 减去平均值
    covMat = np.cov(meanRemoved, rowvar=0)  # 计算协方差矩阵
    eigVals,eigVects = np.linalg.eig(np.mat(covMat))  # 计算协方差矩阵的特征值和特征向量
    print("Eigenvalues are:")
    print(eigVals)
    lowDMat1, reconMat1  = pca(datMat, 1)  # 降维成1维矩阵  前一个主成份
    lowDMat2, reconMat2  = pca(datMat, 2)  # 降维成2维矩阵  前两个主成份
    lowDMat3, reconMat3  = pca(datMat, 3)  # 降维成3维矩阵  前三个主成份
    lowDMat6, reconMat6  = pca(datMat, 6)  # 降维成6维矩阵  前六个主成份
    print("原数据维度: ")
    print(np.shape(datMat))
    print("降维后数据维度1: ")
    print(np.shape(lowDMat1))
    print("降维后数据维度2: ")
    print(np.shape(lowDMat2))
    print("降维后数据维度3: ")
    print(np.shape(lowDMat3))
    print("降维后数据维度6: ")
    print(np.shape(lowDMat6))

if __name__ == "__main__":
    secomTest()
