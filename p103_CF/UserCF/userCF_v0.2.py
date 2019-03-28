#coding=utf-8
import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

#定义数据路径，使用pandas导入数据
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

moviesPath = "/Users/Keson/Desktop/test_data/movies.csv"
ratingsPath = "/Users/Keson/Desktop/test_data/ratings.csv"
moviesDF = pd.read_csv(moviesPath, index_col = None)
ratingsDF = pd.read_csv(ratingsPath, index_col= None)

#ratingsDF:
#userId	movieId	rating	timestamp
#  1	   1	   4	964982703

#moviesDF:
#          movieId         title                     genres
#0            1       Toy Story (1995)  Adventure|Animation|Children|Comedy|Fantasy


#按照9：1的比例将数据拆分为训练集和测试集
trainRatingsDF, testRatingsDF = train_test_split(ratingsDF, test_size=0.1)
print("total_movie_count:" + str(len(set(ratingsDF['movieId'].values.tolist()))))
print("total_user_count:" + str(len(set(ratingsDF['userId'].values.tolist()))))
print("train_movie_count:" + str(len(set(trainRatingsDF['movieId'].values.tolist()))))
print("test_movie_count:" + str(len(set(testRatingsDF['movieId'].values.tolist()))))
print("train_user_count:" + str(len(set(trainRatingsDF['userId'].values.tolist()))))
print("test_user_count:" + str(len(set(testRatingsDF['userId'].values.tolist()))))





#使用pivot_table得到用户-电影的评分矩阵
trainRatingsPivotDF = pd.pivot_table(trainRatingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                 index=['userId'], values='rating', fill_value=0)

# movieID 1 2 3 ...
# userID 1
#        2
#value = 用户对电影的rating

#得到电影id、用户id与其索引的映射关系
# enumerate返回穷举序列号与值
moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
ratingValues = trainRatingsPivotDF.values.tolist()

#moviesMap : {0:1,1:2,3:3...}
#usersMap: {0:1,1:2,3:3...}
#ratingValues: 矩阵变成list 每一行变成list的一个值!   用户对每一个电影打的分，没有就是0.0 [0.0, 0.0, 0.0, 1.5...]


#利用余弦相似度计算用户之间的相似度
def calCosineSimilarity(list1, list2):
    res = 0
    denominator1 = 0
    denominator2 = 0
    for (val1, val2) in zip(list1, list2):
        res += (val1 * val2)
        denominator1 += val1 ** 2
        denominator2 += val2 ** 2
    return res / (math.sqrt(denominator1 * denominator2))

## 根据用户对电影的评分，来判断每个用户间相似度
userSimMatrix = np.zeros((len(ratingValues), len(ratingValues)), dtype=np.float32)
for i in range(len(ratingValues) - 1):
    for j in range(i + 1, len(ratingValues)):
        userSimMatrix[i, j] = calCosineSimilarity(ratingValues[i], ratingValues[j])
        userSimMatrix[j, i] = userSimMatrix[i, j]


#接下来，我们要找到与每个用户最相近的K个用户，用这K个用户的喜好来对目标用户进行物品推荐，这里K=10
#这里我们选择最相近的10个用户
userMostSimDict = dict()
for i in range(len(ratingValues)):
    userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])), key=lambda x: x[1], reverse=True)[:2]

# 用这K个用户的喜好中目标用户没有看过的电影进行推荐
userRecommendValues = np.zeros((len(ratingValues), len(ratingValues[0])), dtype=np.float32)  #

for i in range(len(ratingValues)):
    for j in range(len(ratingValues[i])):
        if ratingValues[i][j] == 0:
            val = 0
            for (user, sim) in userMostSimDict[i]:
                val += (ratingValues[user][j] * sim)
            userRecommendValues[i, j] = val

#为每个用户推荐10部电影：
userRecommendDict = dict()
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])), key=lambda x: x[1], reverse=True)[:2]



userRecommendList = []
for key, value in userRecommendDict.items():
    user = usersMap[key]
    for (movieId, val) in value:
        userRecommendList.append([user, moviesMap[movieId],val])


# 将推荐结果的电影id转换成对应的电影名
recommendDF = pd.DataFrame(userRecommendList, columns=['userId', 'movieId','Recommend_Val'])
recommendDF = pd.merge(recommendDF, moviesDF[['movieId', 'title']], on='movieId', how='inner')
recommendDF = recommendDF.sort_values(by=['userId'])


#print(recommendDF)

recommendDF.to_csv('userCF_Result.csv', index=False, header=True)

























