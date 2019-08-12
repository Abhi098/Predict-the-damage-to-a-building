import pandas as pd
import numpy as np
import time
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from os import system
from matplotlib import pyplot as plt
from matplotlib import style

# system('clear')

prev_time = time.time()


# pd.set_option('max_columns',None)

label_enc = preprocessing.LabelEncoder()

mtrain_filepath = "../newDataset/merged_train.csv"
# mtest_filepath = "../newDataset/merged_test.csv"


print('\nreading dataset....')

train = pd.read_csv(mtrain_filepath)
# test = pd.read_csv(mtest_filepath)


print('\nread successful!')
train_headers = ['building_id', 'district_id', 'vdcmun_id', 'ward_id', 'legal_ownership_status', 'count_families', 'has_secondary_use', 'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police', 'has_secondary_use_other', 'count_floors_pre_eq', 'count_floors_post_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq', 'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other', 'condition_post_eq', 'area_assesed', 'damage_grade', 'has_geotechnical_risk', 'has_geotechnical_risk_fault_crack', 'has_geotechnical_risk_flood', 'has_geotechnical_risk_land_settlement', 'has_geotechnical_risk_landslide', 'has_geotechnical_risk_liquefaction', 'has_geotechnical_risk_other', 'has_geotechnical_risk_rock_fall', 'has_repair_started']

grade_enc = {
	'Grade 1':1,
	'Grade 2':2,
	'Grade 3':3,
	'Grade 4':4,
	'Grade 5':5
}

grade_dec = {
	'1':'Grade 1',
	'2':'Grade 2',
	'3':'Grade 3',
	'4':'Grade 4',
	'5':'Grade 5'
}


# train['dist_vdc_ward_id'] = train['district_id'].map(str) + train['vdcmun_id'].map(str) + train['ward_id'].map(str)
# test['dist_vdc_ward_id'] = test['district_id'].map(str) + test['vdcmun_id'].map(str) + test['ward_id'].map(str)

train['count_floors_diff'] = train['count_floors_pre_eq'] - train['count_floors_post_eq']
# test['count_floors_diff'] = test['count_floors_pre_eq'] - test['count_floors_post_eq']


train['height_ft_diff'] = train['height_ft_pre_eq'] - train['height_ft_post_eq']
# test['height_ft_diff'] = test['height_ft_pre_eq'] - test['height_ft_post_eq']

# failed attempt
# train['damage_grade'] = grade_enc[train['damage_grade']]


train = train.dropna()
# test = test.dropna()

train_y = pd.Series([grade_enc[i] for i in train['damage_grade']])




# [i**2 for i in range(10)]

# [grade_enc[i] for i in train['damage_grade']]


# print('\ntrain cols length : ',len(train.columns))
# print('test cols length : ',len(test.columns))


trainx = train.drop(columns=['building_id','damage_grade','has_secondary_use_other','has_secondary_use_industry','has_secondary_use_school','has_secondary_use_institution','has_secondary_use_rental','has_secondary_use_hotel','has_secondary_use_agriculture','has_secondary_use_use_police','has_secondary_use_gov_office','has_secondary_use_health_post','has_geotechnical_risk_other','has_geotechnical_risk_liquefaction','has_superstructure_rc_engineered','has_geotechnical_risk_flood'])
# test_x = test.drop(columns=['building_id','district_id','vdcmun_id','ward_id','count_floors_pre_eq','count_floors_post_eq','height_ft_diff','height_ft_pre_eq'])


train_x, test_x, train_y, test_y = train_test_split(trainx,train_y,train_size=0.8,test_size=0.2,random_state=22)


# print(len(train_x.columns))

# print('\ntrain_x null values : ',train_x.isnull().sum().sum())
# print('train_y null values : ',train_y.isnull().sum().sum())
# print('test_x null values : ',test_x.isnull().sum().sum())
# print('length train_x: ',len(train_x))
# print('length train_y: ',len(train_y))
# print('length test_x: ',len(test_x))



# print('\ntrain_x null values : ',train_x.isnull().sum().sum())
# print('train_y null values : ',train_y.isnull().sum().sum())
# print('test_x null values : ',test_x.isnull().sum().sum())
# print('length train_x: ',len(train_x))
# print('length train_y: ',len(train_y))
# print('length test_x: ',len(test_x))



# print('\nobject type attributes : ',len(train_x.dtypes[train.dtypes == 'object']))

print('\nlabel encoding....')

for col in test_x.columns.values:
	if test_x.loc[:,col].dtype == 'object':
		# data = train_x[col].append(test_x[col])
		label_enc.fit(trainx.loc[:,col])
		train_x.loc[:,col] = label_enc.transform(train_x.loc[:,col])
		test_x.loc[:,col] = label_enc.transform(test_x.loc[:,col])


# print(train_x.head())
# print(test_x.head())
# print(set(train_x.columns) - set(test_x.columns))


# print('\ntrain_x cols length : ',len(train_x.columns))
# print('test_x cols length : ',len(test_x.columns))






# label_enc = preprocessing.LabelEncoder()
# ids = train['dist_vdc_ward_id'].append(test['dist_vdc_ward_id'])
# label_enc.fit(ids.values)
# train = train.assign(
# 		dist_vdc_ward_id = label_enc.transform(train['dist_vdc_ward_id'])
# 	)



# print(train.head()['dist_vdc_ward_id'])

# print(list(train.columns))



# train_x = []
# train_y = train['damage_grade']




print('\ntraining model....')

clf = RandomForestClassifier(n_estimators=150,n_jobs=-1,max_features=0.25) ## n_estimators=100, ,max_features=0.32
clf.fit(train_x,train_y)


print('\npredicting data....\n')


predictions_train = clf.predict(train_x)
predictions_test = clf.predict(test_x)


# for i in range(11,21):
# 	print("Actual outcome : Grade - {}   and predicted outcome : Grade - {}".format(list(test_y)[i],predictions_test[i]))



print('\ncalculation accuracy....')



print("\nTrain Accuracy : ",round(accuracy_score(train_y,predictions_train)*100,4))
print("Test Accuracy : ",round(accuracy_score(test_y,predictions_test)*100,4))


# print("\nConfusion Matrix : ")
# print(confusion_matrix(test_y,predictions_test))

# print('model score : ',clf.score(test_x,test_y))

importances = pd.DataFrame({'feature':train_x.columns,'importance':np.round(clf.feature_importances_,4)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

print(importances)

# dicfinal={}
# for i in train_x.columns:
# 	dicfinal[i]=importances[i]

# print(dicfinal)
# print(dicfinal.columns)
# print(dicfinal.shape)	
# data6=pd.DataFrame.from_dict(dicfinal,orient='index')
# data6.to_csv('Dictionary.csv',index=False)

# print(importances)
# print(importances.columns)
# importances.plot(x=index,y='importance',kind='bar')
# plt.show()

# print('prediction_test : ',predictions_test)

# predictions_test = pd.Series(predictions_test)

# predict_test = pd.Series([grade_dec[str(i)] for i in predictions_test[i]])
# predict_test = 'Grade ' + predictions_test.map(str)

# print(len(test['building_id']))
# print(len(predictions_test))
# submissions = pd.DataFrame([test['building_id'],predict_test])

# print(submissions.head())

new_time = time.time()

total_time = round((new_time - prev_time),2)
print('\ntotal time : ',total_time)