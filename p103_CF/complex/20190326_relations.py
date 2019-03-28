import math
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from sklearn.preprocessing import OneHotEncoder
import operator

relationsPath = "/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv"
userinfosPath = "/Users/Keson/Desktop/ICBC_deliver/test_data/user_info.csv"
productinfosPath = "/Users/Keson/Desktop/ICBC_deliver/test_data/product_info.csv"
relationsDF = pd.read_csv(relationsPath, index_col = None)
userinfosDF = pd.read_csv(userinfosPath, index_col = None)
productinfosDF = pd.read_csv(productinfosPath, index_col = None)


productsMap = dict(enumerate(list(relationsDF.columns)))
usersMap = dict(enumerate(list(relationsDF.index)))
ratingValues = relationsDF.values.tolist()

computeDF = pd.merge(userinfosDF, relationsDF[['userId','pId1','pId2','pId3','pId4','pId5','pId6','pId7','pId8','pId9','pId10']], on='userId', how='inner')



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

def productId_to_list():
	data_productId = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/product_info.csv", usecols=[0])
	train_productId_data = np.array(data_productId)
	train_data_productId_list = train_productId_data.tolist()
	return train_data_productId_list

def pId1_to_list():
	data_pId1 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[1])
	train_pId1_data = np.array(data_pId1)
	train_data_pId1_list = train_pId1_data.tolist()
	return train_data_pId1_list

def pId2_to_list():
	data_pId2 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[2])
	train_pId2_data = np.array(data_pId2)
	train_data_pId2_list = train_pId2_data.tolist()
	return train_data_pId2_list

def pId3_to_list():
	data_pId3 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[3])
	train_pId3_data = np.array(data_pId3)
	train_data_pId3_list = train_pId3_data.tolist()
	return train_data_pId3_list	

def pId4_to_list():
	data_pId4 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[4])
	train_pId4_data = np.array(data_pId4)
	train_data_pId4_list = train_pId4_data.tolist()
	return train_data_pId4_list	

def pId5_to_list():
	data_pId5 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[5])
	train_pId5_data = np.array(data_pId5)
	train_data_pId5_list = train_pId5_data.tolist()
	return train_data_pId5_list	

def pId6_to_list():
	data_pId6 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[6])
	train_pId6_data = np.array(data_pId6)
	train_data_pId6_list = train_pId6_data.tolist()
	return train_data_pId6_list	

def pId7_to_list():
	data_pId7 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[7])
	train_pId7_data = np.array(data_pId7)
	train_data_pId7_list = train_pId7_data.tolist()
	return train_data_pId7_list	

def pId8_to_list():
	data_pId8 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[8])
	train_pId8_data = np.array(data_pId8)
	train_data_pId8_list = train_pId8_data.tolist()
	return train_data_pId8_list	

def pId9_to_list():
	data_pId9 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[9])
	train_pId9_data = np.array(data_pId9)
	train_data_pId9_list = train_pId9_data.tolist()
	return train_data_pId9_list	

def pId10_to_list():
	data_pId10 = pd.read_csv("/Users/Keson/Desktop/ICBC_deliver/test_data/relations.csv", usecols=[10])
	train_pId10_data = np.array(data_pId10)
	train_data_pId10_list = train_pId10_data.tolist()
	return train_data_pId10_list
# print(train_data_userId_list)
	
def MaxMinNormalization(x,Max,Min):
	x = (x - Min) / (Max - Min)
	return x
	
def z_score_stand(x,mu,sigma):
	x = (x - mu) / sigma
	return x

occu = occu_to_ohe()
gender = gender_to_ohe()
age = age_to_list()
income = income_to_list()
userId = userId_to_list()
productId = productId_to_list()
pId1 = pId1_to_list()
pId2 = pId2_to_list()
pId3 = pId3_to_list()
pId4 = pId4_to_list()
pId5 = pId5_to_list()
pId6 = pId6_to_list()
pId7 = pId7_to_list()
pId8 = pId8_to_list()
pId9 = pId9_to_list()
pId10 = pId10_to_list()


max_age = np.max(age)
max_income = np.max(income)
max_pId1 = np.max(pId1)
max_pId2 = np.max(pId2)
max_pId3 = np.max(pId3)
max_pId4 = np.max(pId4)
max_pId5 = np.max(pId5)
max_pId6 = np.max(pId6)
max_pId7 = np.max(pId7)
max_pId8 = np.max(pId8)
max_pId9 = np.max(pId9)
max_pId10 = np.max(pId10)

min_age = np.min(age)
min_income = np.min(income)
min_pId1 = np.min(pId1)
min_pId2 = np.min(pId2)
min_pId3 = np.min(pId3)
min_pId4 = np.min(pId4)
min_pId5 = np.min(pId5)
min_pId6 = np.min(pId6)
min_pId7 = np.min(pId7)
min_pId8 = np.min(pId8)
min_pId9 = np.min(pId9)
min_pId10 = np.min(pId10)

for i in range(len(age)):
	age[i] = MaxMinNormalization(age[i],max_age,min_age)
for j in range(len(income)):
	income[j] = MaxMinNormalization(income[i], max_income, min_income)
for a in range(len(pId1)):
	pId1[a] = MaxMinNormalization(pId1[a],max_pId1,min_pId1)
for b in range(len(pId2)):
	pId2[b] = MaxMinNormalization(pId2[b],max_pId2,min_pId2)
for c in range(len(pId3)):
	pId3[c] = MaxMinNormalization(pId3[c],max_pId3,min_pId3)
for d in range(len(pId4)):
	pId4[d] = MaxMinNormalization(pId4[d],max_pId4,min_pId4)
for e in range(len(pId5)):
	pId5[e] = MaxMinNormalization(pId5[e],max_pId5,min_pId5)
for f in range(len(pId6)):
	pId6[f] = MaxMinNormalization(pId6[f],max_pId6,min_pId6)
for g in range(len(pId7)):
	pId7[g] = MaxMinNormalization(pId7[g],max_pId7,min_pId7)
for h in range(len(pId8)):
	pId8[h] = MaxMinNormalization(pId8[h],max_pId8,min_pId8)
for k in range(len(pId9)):
	pId9[k] = MaxMinNormalization(pId9[k],max_pId9,min_pId9)
for l in range(len(pId10)):
	pId10[l] = MaxMinNormalization(pId10[l],max_pId10,min_pId10)

	
final_sim = np.hstack((userId,gender,age,occu,income,pId1,pId2,pId3,pId4,pId5,pId6,pId7,pId8,pId9,pId10))
final_sim_product = np.hstack((pId1,pId2,pId3,pId4,pId5,pId6,pId7,pId8,pId9,pId10))


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
userMostSimDict = dict()
for i in range(len(userId)):
	userMostSimDict[i] = sorted(enumerate(list(userSimMatrix[i])), key=lambda x: x[1], reverse=True)[:2]


userRecommendValues = np.zeros((len(userId), len(final_sim_product[0])), dtype=np.float32)  #

for i in range(len(userId)):
	for j in range(len(final_sim_product[i])):
		if final_sim_product[i][j] == 0:
			val = 0
			for (user, sim) in userMostSimDict[i]:
				val += (final_sim_product[user][j] * sim)
			userRecommendValues[i, j] = val

userRecommendDict = dict() #this is result
for i in range(len(final_sim_product)):
	userRecommendDict[i] = sorted(enumerate(list(userRecommendValues[i])), key=lambda x: x[1], reverse=True)[:2]


userRecommendList = []
for key, value in userRecommendDict.items():
	user = usersMap[key]
	for (productId, val) in value:
		userRecommendList.append([user, productsMap[productId],val])



recommendDF = pd.DataFrame(userRecommendList, columns=['userId', 'PID','Recommend_Val'])
recommendDF = pd.merge(recommendDF, productinfosDF[['PID', '材质','重量','类型','产品名称']], on='PID', how='inner')
recommendDF = recommendDF.sort_values(by=['userId'])

recommendDF.to_csv('complex_recommend.csv', index=False, header=True)



