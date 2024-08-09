# 导入包
import warnings  # 忽略错误提示
warnings.filterwarnings('ignore')  # 代码可以正常运行但是会提示警告，特别讨厌
warnings.simplefilter('ignore')  # 也是一种错误忽略
import pandas as pd
import re
import jieba
jieba.set_dictionary(r".\dict.txt")
jieba.initialize()
import numpy as np

# 读取数据
def data_load(path):
    data = pd.read_excel(path)
    return data
path = r'./datahouse/data.xlsx'
data = data_load(path)

# 数据清洗处理
def data_pre_process(Data):  # 数据预处理
    data = Data[:]
    data = data[['岗位名称', '地点', '工资', '岗位职责', '岗位要求', 'label']]
    data_drop = data.drop_duplicates(subset=['岗位名称', '岗位职责', '岗位要求'], keep='first', inplace=False)  # 按照岗位职责和岗位要求去除重复值
    data_drop = data_drop.dropna()  # 去除空值
    # 清洗
    data_drop['岗位职责_cut'] = data_drop['岗位职责'].apply(lambda x: x.replace('\n', ''))
    data_drop['岗位职责_cut'] = data_drop['岗位职责_cut'].apply(lambda x: x.replace(' ', ''))
    data_drop['岗位职责_cut'] = data_drop['岗位职责_cut'].apply(lambda x: re.sub('[0-9]', '', x))
    data_drop['岗位职责_cut'] = data_drop['岗位职责_cut'].apply(
        lambda x: re.sub("[; ！•、，\\\/\'+．:（）...：‘，。’,“”、\#•*~(\t；-]", '', x))

    data_drop['岗位要求_cut'] = data_drop['岗位要求'].apply(lambda x: x.replace('\n', ''))
    data_drop['岗位要求_cut'] = data_drop['岗位要求_cut'].apply(lambda x: x.replace(' ', ''))
    data_drop['岗位要求_cut'] = data_drop['岗位要求_cut'].apply(lambda x: re.sub('[0-9]', '', x))
    data_drop['岗位要求_cut'] = data_drop['岗位要求_cut'].apply(
        lambda x: re.sub("[; ！•、，\\\/\'+．:（）...：‘，。’,“”、\#•*~(\t；-]", '', x))

    # 分词
    jieba.load_userdict('interger.txt')  # 定义分隔词，保证数据词不被分开
    data_jieba = data_drop[:]
    data_jieba['岗位职责_cut'] = data_jieba['岗位职责_cut'].apply(lambda x: jieba.lcut(x))
    data_jieba['岗位要求_cut'] = data_jieba['岗位要求_cut'].apply(lambda x: jieba.lcut(x))

    # 去停用词
    stopwords = pd.read_csv('stoplist.txt', encoding='utf-8', sep='haha', header=None, engine='python')
    stopwords = ['要'] + list(stopwords.iloc[:, 0])  # 手动添加停用词
    data_jieba['岗位职责_stop'] = data_jieba['岗位职责_cut'].apply(lambda x: [i for i in x if i not in stopwords])
    data_jieba['岗位职责_stop'] = data_jieba['岗位职责_stop'].apply(lambda x: ' '.join(x))  # 转成字符串
    data_jieba['岗位要求_stop'] = data_jieba['岗位要求_cut'].apply(lambda x: [i for i in x if i not in stopwords])
    data_jieba['岗位要求_stop'] = data_jieba['岗位要求_stop'].apply(lambda x: ' '.join(x))  # 转成字符串
    data_jieba = data_jieba.reset_index(drop=True)
    return data_jieba

data_process = data_pre_process(data)

base_duty = data_process['岗位要求_stop']

from sklearn.feature_extraction.text import TfidfVectorizer
# 建立TfidfVectorizer模型
tfidf_vec = TfidfVectorizer(min_df=1,max_df=5000,max_features=200)
# 计算tfidf矩阵
sparse_result_tfidf = tfidf_vec.fit_transform(base_duty)

# 文本特征提取模块，CountVectorizer词频统计，TfidfTransformer把词频结果转化为TF-IDF类
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
countVectorizer = CountVectorizer()
data_tr = countVectorizer.fit_transform(data_process['岗位要求_stop'])  # 装换为权值向量
# 获得训练集对象的TF-IDF权值
X_tr = TfidfTransformer().fit_transform(data_tr.toarray()).toarray()

# 计算两个向量之间的余弦相似度
def cos_sin(vector_a , vector_b):
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = float(vector_a * vector_b.T)
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
    sim = 0.5 + 0.5 * cos
    return sim

while True:
    n = input('请输入你喜欢的一类岗位：')
    print('---------------------------------------------------------------------------')
    data = data_process[data_process['岗位名称'].str.contains(n)][['岗位名称','地点','工资','岗位职责','岗位要求']]
    if len(data)>=1:
        print(data)
        data.to_excel('您喜欢的岗位数据.xlsx')
        print('---------------------------------------------------------------------------')
        m = input('请输入你想了解的岗位序号：')
        print(data_process.loc[int(m),['岗位名称','地点','工资','岗位职责','岗位要求']])
        data_process.loc[int(m),['岗位名称','地点','工资','岗位职责','岗位要求']].to_excel('您想了解的岗位数据.xlsx')
        print('---------------------------------------------------------------------------')
        k = input('是否需要推荐岗位要求相关工作？请输入yes或者no：')
        a_dict = {}
        if k =='yes':
            n = 0
            for i in X_tr:
                a_dict[n] = cos_sin(X_tr[int(m)],i)
                n += 1
            L = sorted(a_dict.items(),key=lambda item:item[1],reverse=True)
            data_frame = pd.DataFrame(columns=['岗位名称','地点','工资','岗位职责','岗位要求','相似度'])
            for i in range(5):
                print(L[i][0],end='\t')
                a = int(L[i][0])
                print(data_process.iloc[a][['岗位名称','地点','工资','岗位职责','岗位要求']])
                data_frame.loc[i] = data_process.iloc[a][['岗位名称','地点','工资','岗位职责','岗位要求']]
                data_frame.loc[i][5] = L[i][1]
                print('基于岗位要求的岗位相似度：',L[i][1])
            data_frame.to_excel('基于岗位要求的相似岗位推荐.xlsx')
        else:
            pass
        print('---------------------------------------------------------------------------')
        o = input('是否继续查询，请输入yes或者no：')
        if o == 'yes':
            continue
        else:
            break
    else:
        print('---------------------------------------------------------------------------')
        print('暂时无法查询您输入的工作岗位，请重新输入！')
        continue