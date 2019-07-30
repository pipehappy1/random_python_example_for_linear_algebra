from numpy import *
import numpy as np

#这个函数是用来统计得到一个总体单词向量的，就是：所有的文本一共有多少个不同的单词，
#由这些不同的单词组成一个单词向量，向量可以理解成为列表
#dataSet：是训练数据集
def createVocabList(dataSet):
    #使用python内置set函数创建一个空的集合，用来保存数据集的单词向量，集合里面的元素都是独一无二的
    vocabSet = set([])  
    #使用for循环逐行逐行地读取数据集
    for document in dataSet:
        #set(document) ：将当前这一行数据去重，得到这行数据的单词向量
        #然后通过集合的或运算（和数学的集合或运算一样），把当前行的单词向量保存到集合vocabSet中
        #在保存到集合vocabSet前，都要把当前的集合vocabSet与当前的行的单词向量set(document)做集合的或运算处理
        #目的是为了保证每次最后保存到集合vocabSet中的单词都是独一无二的
        vocabSet = vocabSet | set(document) 
    #最后将集合vocabSet变成为列表类型的数据返回
    return list(vocabSet)

#这个函数是获取训练数据集和标签
def loadDataSet():
    postingList=[['a', 'great', 'game'],
                   ['the', 'election', 'was', 'over'],
                   ['very', 'clean', 'match'],
                   ['a', 'clean', 'but', 'forgettable', 'game'],
                   ['it', 'was', 'a', 'close', 'election']]
    classVec = [1, 0, 1, 1, 0]
    return postingList,classVec

dataSet,labels = loadDataSet()

#验证数据集规模和数据集标签
print('训练数据集规模：')
print(np.shape(dataSet))
print('打印训练数据集：')
print(dataSet)
print('打印训练数据集标签：')
print(labels)

#这个函数是用来统计得到一个总体单词向量的，就是：所有的文本一共有多少个不同的单词，
#由这些不同的单词组成一个单词向量，向量可以理解成为列表
#dataSet：是训练数据集
def createVocabList(dataSet):
    #使用python内置set函数创建一个空的集合，用来保存数据集的单词向量，集合里面的元素都是独一无二的
    vocabSet = set([])  
    #使用for循环逐行逐行地读取数据集
    for document in dataSet:
        #set(document) ：将当前这一行数据去重，得到这行数据的单词向量
        #然后通过集合的或运算（和数学的集合或运算一样），把当前行的单词向量保存到集合vocabSet中
        #在保存到集合vocabSet前，都要把当前的集合vocabSet与当前的行的单词向量set(document)做集合的或运算处理
        #目的是为了保证每次最后保存到集合vocabSet中的单词都是独一无二的
        vocabSet = vocabSet | set(document) 
    #最后将集合vocabSet变成为列表类型的数据返回
    return list(vocabSet)

#这个函数的作用是将原来文本的一条记录，转变成为与单词向量一样长度的，只有0和1两个值的一条数据记录
#vocabList：原数据集单词向量
#inputSet：数据集的一条文本记录
def setOfWords2Vec(vocabList, inputSet):
    #生成一个长度与单词向量一样的列表，里面的元素默认都是零
    returnVec = [0]*len(vocabList)
    #逐个地把每条记录的单词读出来
    for word in inputSet:
        #判断这个单词在单词向量中存不存在
        if word in vocabList:
            #如果单词向量存在这个单词，那就获取这个单词所在单词向量的位置下标vocabList.index(word)
            #然后根据这个下标，把returnVec的对应位置下标的值修改为1
            returnVec[vocabList.index(word)] = 1
        else:
            #如果这个单词不存在单词向量中，打印一条提示的信息该单词不在单词向量中。（向量和列表是一样的）
            print("the word: %s is not in my Vocabulary!" % word)
    #将这条转换得到的数据返回
    return returnVec

##trainData = []
##for i in range(len(dataSet)):
##    returnVec = setOfWords2Vec(world,dataSet[i])
##    trainData.append(returnVec)
    
#这个函数的作用是获取原数据集类别的概率和数据集每种类别对应的特征的概率
#trainMatrix：转换后的数据集
#trainCategory:数据集的标签
def trainNB0(trainMatrix,trainCategory):
    #取得数据集行数
    numTrainDocs = len(trainMatrix)
    #取得列数
    numWords = len(trainMatrix[0])
    #计算标签列1的概率
    pAbusive = sum(trainCategory)/float(numTrainDocs)
    #生成一个变量用来保存属于0这个类别的记录的所有特征对应位置累加的结果，现在默认里面的数值都是1
    p0Num = np.ones(numWords)
    #生成一个变量用来保存属于1这个类别的记录的所有特征对应位置累加的结果，现在默认里面的数值都是1
    p1Num = np.ones(numWords)    
    #定义一个变量用来保存属于0这个类别的所有记录的所有元素的累加和，默认初始值是2
    p0Denom = 2.0
    #定义一个变量用来保存属于1这个类别的所有记录的所有元素的累加和，默认初始值是2
    p1Denom = 2.0    
    #使用for循环按行读取转换后的数据集
    for i in range(numTrainDocs):
        #如果当前行的数据记录是属于1这个类别的
        if trainCategory[i] == 1:
            #那就把这一条数据记录中的各个元素对应累加到p1Num中
            p1Num += trainMatrix[i]
            #把这条数据记录中的所有元素都累加到p1Denom中
            p1Denom += sum(trainMatrix[i])
        else:
            #如果当前行的数据记录是属于0这个类别的
            #那就把这一条数据记录中的各个元素对应累加到p0Num中
            p0Num += trainMatrix[i]
            #把这条数据记录中的所有元素都累加到p0Denom中
            p0Denom += sum(trainMatrix[i])
    #求得1这个类别对应各个特征的概率p1Num/p1Denom，并且把这个概率取对数，使得每个特征属性对应的值绝对值不那么小
    p1Vect = np.log(p1Num/p1Denom)     
    #求得0这个类别对应各个特征的概率p0Num/p0Denom，并且把这个概率取对数，使得每个特征属性对应的值绝对值不那么小
    p0Vect = np.log(p0Num/p0Denom)
    #p0Vect：零这个类别对应每个特征属性的概率取值，（使用log函数处理了一下）
    #p1Vect：零这个类别对应每个特征属性的概率取值，（使用log函数处理了一下）
    #pAbusive：标签列1的概率
    return p0Vect,p1Vect,pAbusive

#vec2Classify:未知标签的数据记录
#p0Vec：零标签对应32个属性的概率
#p1Vec：1标签对应32个属性的概率
#pClass1：1标签所占的概率
def classifyNB(vec2Classify, p0Vec, p1Vec, pClass1):
    #vec2Classify* p1Vec：将这个未知标签的记录中每个元素对应乘以一个概率，
    #然后再使用sum函数求所有乘积的累加和sum(vec2Classify * p1Vec)，
    #最后再加上原数据集1这个类别标签概率的对数
    #最后得到的p1就是：这条vec2Classify未知标签数据记录属于1这个类别的概率的大小
    p1 = sum(vec2Classify * p1Vec) + np.log(pClass1) 
    #vec2Classify* p0Vec：将这个未知标签的记录中每个元素对应乘以一个概率，
    #然后再使用sum函数求所有乘积的累加和sum(vec2Classify * p0Vec)，
    #最后再加上原数据集0这个类别标签概率的对数
    #最后得到的p0就是：这条vec2Classify未知标签数据记录属于0这个类别的概率的大小
    p0 = sum(vec2Classify * p0Vec) + np.log(1.0 - pClass1)
    #如果p1>p0
    if p1 > p0:
        #那么返回这个未知标签的记录的预测标签是1
        return 1
    else: 
        #否则预测标签就是0
        return 0

#这个函数是用来测试算法的
def testingNB():
    #加载上面的训练数据集，对应的标签列
    listOPosts,listClasses = loadDataSet()
    #根据训练数据集得到数据集的单词向量，里面的每个单词都是独一无二的
    myVocabList = createVocabList(listOPosts)
    #定义一个列表，用来保存将原数据集转换之后的数据记录，然后就把这个列表数据集当成训练数据集
    trainMat=[]
    #使用for循环逐行地读取原数据集
    for postinDoc in listOPosts:
        #setOfWords2Vec(myVocabList, postinDoc：根据单词向量把当前这条原记录变成另一种数据记录形式
        #然后把这条转换得到的数据集记录保存到trainMat中
        trainMat.append(setOfWords2Vec(myVocabList, postinDoc))
    #np.array(trainMat)：将数据集的数据集类型变成数组类型
    #np.array(listClasses)：将数据集的标签类型变成数组类型
    #调用trainNB0函数获取新的训练数据集中
    #零这个类别对应每个特征属性的概率取值p0V，（使用log函数处理了一下）
    #1这个类别对应每个特征属性的概率取值p1V，（使用log函数处理了一下）
    #标签列1的概率pAb
    p0V,p1V,pAb = trainNB0(np.array(trainMat),np.array(listClasses))
    #testEntry：一条未知标签的原文本数据
    testEntry = ['a', 'very', 'close','game']
    #调用setOfWords2Vec函数根据单词列表来将这条测试文本数据转化一下数据形式
    #最后把得到的这个数据记录再转变从数组类型的数据记录
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    #调用classifyNB函数来获取该测试数据记录的标签
    #并且打印这条记录数据的最后预测标签类别
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    #这是第二条文本测试记录
    testEntry = ['a', 'very', 'close','election']
    #调用setOfWords2Vec函数根据单词列表来将这条测试文本数据转化一下数据形式
    #最后把得到的这个数据记录再转变从数组类型的数据记录
    thisDoc = np.array(setOfWords2Vec(myVocabList, testEntry))
    #调用classifyNB函数来获取该测试数据记录的标签
    #并且打印这条记录数据的最后预测标签类别
    print(testEntry,'classified as: ',classifyNB(thisDoc,p0V,p1V,pAb))
    
#调用testingNB这个函数，开始测试性预测
testingNB()
