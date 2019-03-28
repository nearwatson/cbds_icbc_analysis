import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import operator

print("Data Import Starting...")

moviesPath = "/Users/Keson/Desktop/test_data/movies.csv"
ratingsPath = "/Users/Keson/Desktop/test_data/ratings.csv"
userinfosPath = "/Users/Keson/Desktop/test_data/user_info.csv"
moviesDF = pd.read_csv(moviesPath, index_col = None)
ratingsDF = pd.read_csv(ratingsPath, index_col = None)
userinfoDF = pd.read_csv(userinfosPath, index_col = None)

print("Import data finished.")

trainRatingsPivotDF = pd.pivot_table(ratingsDF[['userId', 'movieId', 'rating']], columns=['movieId'],
                                 index=['userId'], values='rating', fill_value=0)
moviesMap = dict(enumerate(list(trainRatingsPivotDF.columns)))
usersMap = dict(enumerate(list(trainRatingsPivotDF.index)))
ratingValues = trainRatingsPivotDF.values.tolist()

#print("Total movie number: " + len(moviesMap))
#print("Total user number: " + len(usersMap))

def occu_to_ohe():
	data_occu = pd.read_csv("/Users/Keson/Desktop/test_data/user_info.csv",usecols=[3])
	train_occu_data = np.array(data_occu)
	train_data_occu_list = train_occu_data.tolist()
	#print(train_data_occu_list)

	ohe = OneHotEncoder()
	ohe_occu = np.array(train_data_occu_list)
	ohe.fit(ohe_occu) # 样本数据
	c = ohe.transform(ohe_occu).toarray()
	return c
	#print(c)

def gender_to_ohe():
	data_gender = pd.read_csv("/Users/Keson/Desktop/test_data/user_info.csv",usecols=[1])
	train_gender_data = np.array(data_gender)
	train_data_gender_list = train_gender_data.tolist()
	#print(train_data_gender_list)

	ohe = OneHotEncoder()
	ohe_gender = np.array(train_data_gender_list)
	ohe.fit(ohe_gender) # 样本数据
	c = ohe.transform(ohe_gender).toarray()
	return c
	#print(c)

def age_to_list():
	data_age = pd.read_csv("/Users/Keson/Desktop/test_data/user_info.csv",usecols=[2])
	train_age_data = np.array(data_age)
	train_data_age_list = train_age_data.tolist()
	return train_data_age_list
	#print(train_data_age_list)
	
def income_to_list():
	data_income = pd.read_csv("/Users/Keson/Desktop/test_data/user_info.csv",usecols=[4])
	train_income_data = np.array(data_income)
	train_data_income_list = train_income_data.tolist()
	return train_data_income_list
	#print(train_data_income_list)

def userId_to_list():
	data_userId = pd.read_csv("/Users/Keson/Desktop/test_data/user_info.csv",usecols=[0])
	train_userId_data = np.array(data_userId)
	train_data_userId_list = train_userId_data.tolist()
	return train_data_userId_list
	#print(train_data_userId_list)

def movieId_to_list():
	data_movieId = pd.read_csv("/Users/Keson/Desktop/test_data/movies.csv", usecols=[0])
	train_movieId_data = np.array(data_movieId)
	train_data_movieId_list = train_movieId_data.tolist()
	return train_data_movieId_list


# print(train_data_userId_list)
	
def MaxMinNormalization(x,Max,Min):
	x = (x - Min) / (Max - Min)
	return x
	
def z_score_stand(x,mu,sigma):
	x = (x - mu) / sigma
	return x
#mu（即均值）用np.average()，sigma（即标准差）用np.std()即可
#找大小的方法直接用np.max()和np.min()就行了

occu = occu_to_ohe()
gender = gender_to_ohe()
age = age_to_list()
income = income_to_list()
userId = userId_to_list()
movieId = movieId_to_list()

mu_age = np.average(age)
mu_income = np.average(income)
sigma_age = np.std(age)
sigma_income = np.std(income)


for i in range(len(age)):
	age[i] = z_score_stand(age[i],mu_age,sigma_age)
for j in range(len(income)):
	income[j] = z_score_stand(income[i], mu_income, sigma_income)

final_sim = np.hstack((userId,gender,age,occu,income))

print("Data transformed finished.")
#print (final_sim)


print("User Sim compute start...")

#余弦相似性
def calCosineSimilarity(list1, list2):
	res = 0
	denominator1 = 0
	denominator2 = 0
	for (val1, val2) in zip(list1, list2):
		res += (val1 * val2)
		denominator1 += val1 ** 2
		denominator2 += val2 ** 2
	return res / (math.sqrt(denominator1 * denominator2))

userSimMatrix = np.zeros((len(userId), len(userId)), dtype=np.float32)
for i in range(len(userId) - 1):
	for j in range(i + 1, len(userId)):
		userSimMatrix[i, j] = calCosineSimilarity(final_sim[i], final_sim[j])
		userSimMatrix[j, i] = userSimMatrix[i, j]

print("User Sim matrix finished.")

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

#print(userRecommendValues)

print("User recommend finding...")
#为每个用户推荐10部电影：
userRecommendDict = dict() #this is result
for i in range(len(ratingValues)):
    userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])), key=lambda x: x[1], reverse=True)[:2]

#print(userRecommendDict)
#print(userRecommendDict)

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

print("Recommend System stop. Check the result.")




	


	










