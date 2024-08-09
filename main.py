import warnings
warnings.filterwarnings('ignore')  # 忽略警告
import pandas as pd
import re
import jieba
import numpy as np
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics.pairwise import cosine_similarity
import os
from wordcloud import WordCloud
import matplotlib.pyplot as plt

jieba.set_dictionary(r"./dict.txt")
jieba.initialize()

# 读取数据
def data_load(path):
    data = pd.read_excel(path)
    return data

path = r'./datahouse/data.xlsx'
data = data_load(path)

# 数据清洗处理并合并岗位要求和岗位职责
def data_pre_process(data):
    data = data[['岗位名称', '地点', '工资', '岗位职责', '岗位要求', 'label']]
    data.drop_duplicates(subset=['岗位名称', '岗位职责', '岗位要求'], keep='first', inplace=True)  # 去重
    data.dropna(inplace=True)  # 去空值

    # 合并岗位职责和岗位要求
    data['岗位描述'] = data['岗位职责'] + " " + data['岗位要求']

    # 清洗文本
    def clean_text(text):
        text = text.replace('\n', '').replace(' ', '')
        text = re.sub('[0-9]', '', text)
        text = re.sub("[; ！•、，\\\/\'+．:（）...：‘，。’,“”、\#•*~(\t；-]", '', text)
        return text

    data['岗位描述_clean'] = data['岗位描述'].apply(clean_text)

    # 分词
    jieba.load_userdict('interger.txt')
    data['岗位描述_cut'] = data['岗位描述_clean'].apply(lambda x: jieba.lcut(x))

    # 去停用词
    stopwords = pd.read_csv('stoplist.txt', encoding='utf-8', sep='haha', header=None, engine='python')
    stopwords = set(stopwords.iloc[:, 0])  # 加载停用词

    data['岗位描述_stop'] = data['岗位描述_cut'].apply(lambda x: ' '.join([i for i in x if i not in stopwords]))

    return data

data_process = data_pre_process(data)

# 保存和加载模型
def save_model(model, model_path):
    model.save(model_path)

def load_model(model_path):
    if os.path.exists(model_path):
        return Word2Vec.load(model_path)
    return None

# 训练并保存模型
def train_word2vec(sentences, model_path):
    model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    save_model(model, model_path)
    return model

model_path = "word2vec.model"
word2vec_model = load_model(model_path)
if word2vec_model is None:
    sentences = [d.split() for d in data_process['岗位描述_stop']]
    word2vec_model = train_word2vec(sentences, model_path)

# 获取Word2Vec向量
def get_word2vec_vector(text, model):
    words = text.split()
    word_vectors = [model.wv[word] for word in words if word in model.wv]
    return np.mean(word_vectors, axis=0) if word_vectors else np.zeros(model.vector_size)

word2vec_vectors = np.array([get_word2vec_vector(text, word2vec_model) for text in data_process['岗位描述_stop']])

# TF-IDF计算
countVectorizer = CountVectorizer()
data_tr = countVectorizer.fit_transform(data_process['岗位描述_stop'])
X_tr_tfidf = TfidfTransformer().fit_transform(data_tr).toarray()

stopwords = pd.read_csv('stoplist.txt', encoding='utf-8', sep='haha', header=None, engine='python')
stopwords = set(stopwords.iloc[:, 0])

# 生成词云图
def Get_cloud(words):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    font = r'simhei.ttf'
    wordcloud = WordCloud(font_path=font, background_color='white')
    wordcloud.generate(str(words))
    plt.imshow(wordcloud)
    plt.axis("off")  # 关闭坐标轴
    plt.show()

while True:
    user_resume = input('请输入您的个人简介：')

    # 处理用户简介
    user_resume_cut = jieba.lcut(user_resume)
    user_resume_clean = ' '.join([word for word in user_resume_cut if word not in stopwords])

    # 将用户简介向量化
    user_tfidf_vector = TfidfTransformer().fit_transform(
        countVectorizer.transform([user_resume_clean]).toarray()).toarray()
    user_word2vec_vector = get_word2vec_vector(user_resume_clean, word2vec_model)

    # 计算相似度并推荐工作
    a_dict = {}
    for i in range(len(X_tr_tfidf)):
        tfidf_similarity = cosine_similarity(user_tfidf_vector, X_tr_tfidf[i].reshape(1, -1))[0][0]
        word2vec_similarity = cosine_similarity(user_word2vec_vector.reshape(1, -1), word2vec_vectors[i].reshape(1, -1))[0][0]
        combined_similarity = 0.5 * tfidf_similarity + 0.5 * word2vec_similarity
        a_dict[i] = combined_similarity

    L = sorted(a_dict.items(), key=lambda item: item[1], reverse=True)
    data_frame = pd.DataFrame(columns=['岗位名称', '地点', '工资', '岗位职责', '岗位要求', '相似度'])

    print('为您推荐的工作岗位：')
    combined_descriptions = ""
    for i in range(5):  # 推荐前5个最相似的岗位
        a = int(L[i][0])
        print(data_process.iloc[a][['岗位名称', '地点', '工资', '岗位职责', '岗位要求']])
        data_frame.loc[i] = data_process.iloc[a][['岗位名称', '地点', '工资', '岗位职责', '岗位要求']]
        data_frame.loc[i]['相似度'] = L[i][1]
        combined_descriptions += " " + data_process.iloc[a]['岗位描述_stop']
        print('基于您的个人简介与该岗位的相似度：', L[i][1])

    # 生成词云图
    Get_cloud(combined_descriptions)

    data_frame.to_excel('基于个人简介的相似岗位推荐.xlsx')

    o = input('是否继续查询，请输入yes或者no：')
    if o == 'yes':
        continue
    else:
        break
