from numpy import*
import operator

def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1],[0.1,0.1]]) # 样本坐标  注意，这是一个numpy包中的矩阵
    labels = ['A','A','B','B','B'] # 样本标签

    return group, labels

# k-近邻算法：给定一个需要分类的样本点inX，通过计算inX与其他样本距离（此处使用欧氏距离）找到与其距离最近的k个样本点，预测结果则为k个样本点中最多类别的那个
# 要求：完成函数 classify0(...) 使其能够使用KNN算法对inX进行分类
def classify0(inX, dataSet, labels, k): # 对应于分类点、样本集、标签向量、k， 返回inX预测的类别，如'A'

    dataSize= dataSet.shape[0]
    diffMat = tile(inX,(dataSize,1))-dataSet
    sqdiffMat = diffMat**2
    Sqdistance = sqdiffMat.sum(axis=1)
    distance =  Sqdistance **0.5
    SortedDdistance = distance.argsort()
    ClassCount = {}
    for i in range(k):
        votedLabels =labels[SortedDdistance[i]]
        ClassCount[votedLabels]= ClassCount.get(votedLabels,0)+1
    VotedClassCount = sorted(ClassCount.items(),key=operator.itemgetter(1),reverse= True)

    return VotedClassCount[0][0]


group,labels = createDataSet()

print(classify0([1.5,1.5], group, labels, 3))