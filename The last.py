# re是regular expression的缩写，表示正则表达式
# sub是substitute的缩写，表示替换；
# re.sub是个正则表达式方面的函数，用来实现通过正则表达式，实现更加强大的替换功能；
from re import sub
from os import listdir
# Counter类的目的是用来跟踪值出现的次数。
# 它是一个无序的容器类型，以字典的键值对形式存储，其中元素作为key，其计数作为value。
from collections import Counter
# chain函数则是可以串联多个迭代对象来形成一个更大的迭代对象
from itertools import chain
# 利用array创建一个矩阵，注意array()里面是一个python列表或者元组
from numpy import array
# 中文分词函数库，把文本精准地分开，不存在冗余，返回一个可迭代的数据类型
from jieba import cut
# 机器学习中的朴素贝叶斯，先验为多项式分布的朴素贝叶斯
from sklearn.naive_bayes import MultinomialNB
# SVC具有分类功能的SVM
from sklearn.svm import LinearSVC
# 存放所有文件中的单词
# 每个元素是一个子列表，其中存放一个文件中的单词
allWords = []
# 定义函数实现从数据文件中的取词过程
def getWordsFromFile(txtFile):
    words = []
    with open(txtFile, encoding='gbk') as fp:
        for line in fp:
            # 移除字符串头尾指定的字符(默认为空格或换行符)或字符序列
            line = line.strip()
            # 过滤干扰字符
            line = sub(r'[. 【】 0-9、——。，！ ~\*]', '', line)
            # 进行分词
            line = cut(line)
            # 过滤长度为1的词
            line = filter(lambda word: len(word) > 1, line)
            # 将line中分好的词逐个添加到words里面
            words.extend(line)
    return words
# 返回使用最多的N个词
def getTopNWords(topN):
    # 前八个为训练集，文件名从0-7
    # 按文件编号顺序处理当前文件中的所有记事本文件
    # 0-3为正常邮件，4-7为垃圾邮件
    txtFiles = [str(i)+'.txt' for i in range(8)]
    # 循环获取所有训练文件的词，并存放再allWords中
    for txtFile in txtFiles:
        allWords.append(getWordsFromFile(txtFile))
    # 对每个词出现的次数进行计数，以便获取出现最多的前N个词
    freq = Counter(chain(*allWords))
    print(freq)
    return [w[0] for w in freq.most_common(topN)]


# 调用getTopNWords函数，得到前60的词存放在topWords中
topWords = getTopNWords(60)
# 获取特征向量，前60个词在每个训练邮件中出现的频率
vector = []
for words in allWords:
    # 在每个邮件中对指定的词数x与前60个词做一一映射
    temp = list(map(lambda x: words.count(x), topWords))
    vector.append(temp)
vector = array(vector)
print(vector)
# 邮件的标签，判断是否为垃圾邮件，1表示垃圾邮件，0表示正常邮件
# 前四个为正常邮件，后四个为垃圾邮件
labels = array([0]*4 + [1]*4)

# 创建模型，使用已知分类的训练集进行训练
# 第一个模型为朴素贝叶斯
model = MultinomialNB()
# 用对应模型训练处理好的特征向量
model.fit(vector, labels)
# 第二个模型为支持向量机
model2 = LinearSVC()
model2.fit(vector, labels)

# 定义预测函数
def predict(txtFile):
    words = getWordsFromFile(txtFile)
    # 使用numpy.array方法将tuple转创建为 ndarray
    currentVector = array(tuple(map(lambda x: words.count(x), topWords)))
    result = model.predict(currentVector.reshape(1, -1))
    result1 = model2.predict(currentVector.reshape(1, -1))
    if result == 1 and result1 == 1:
        return 'NB预测为垃圾邮件,SVM预测为垃圾邮件'
    elif result == 0 and result1 == 1:
        return 'NB预测为正常邮件,SVM预测为垃圾邮件'
    elif result == 1 and result1 == 0:
        return 'NB预测为垃圾邮件,SVM预测为正常邮件'
    else:
        return 'NB预测为正常邮件,SVM预测为正常邮件'


print(predict('8.txt'))
print(predict('9.txt'))
print(topWords)



