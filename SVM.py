import pandas as pd
import numpy as np
import time
from pprint import pprint
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn import preprocessing
from os import system
from sklearn.preprocessing import Imputer
from sklearn.utils import resample
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
# system('clear')

prev_time = time.time()


# pd.set_option('max_columns',None)

label_enc = preprocessing.LabelEncoder()

mtrain_filepath = "newDataset/merged_train.csv"
mtest_filepath = "newDataset/merged_test.csv"
test_filepath = "Dataset/test.csv"

result_filepath = 'result-gridsearchcv1'


print('\nreading dataset....')

train = pd.read_csv(mtrain_filepath)
test = pd.read_csv(mtest_filepath)


print('\nread successful!')
train_headers = ['building_id', 'district_id', 'vdcmun_id', 'ward_id', 'legal_ownership_status', 'count_families', 'has_secondary_use', 'has_secondary_use_agriculture', 'has_secondary_use_hotel', 'has_secondary_use_rental', 'has_secondary_use_institution', 'has_secondary_use_school', 'has_secondary_use_industry', 'has_secondary_use_health_post', 'has_secondary_use_gov_office', 'has_secondary_use_use_police', 'has_secondary_use_other', 'count_floors_pre_eq', 'count_floors_post_eq', 'age_building', 'plinth_area_sq_ft', 'height_ft_pre_eq', 'height_ft_post_eq', 'land_surface_condition', 'foundation_type', 'roof_type', 'ground_floor_type', 'other_floor_type', 'position', 'plan_configuration', 'has_superstructure_adobe_mud', 'has_superstructure_mud_mortar_stone', 'has_superstructure_stone_flag', 'has_superstructure_cement_mortar_stone', 'has_superstructure_mud_mortar_brick', 'has_superstructure_cement_mortar_brick', 'has_superstructure_timber', 'has_superstructure_bamboo', 'has_superstructure_rc_non_engineered', 'has_superstructure_rc_engineered', 'has_superstructure_other', 'condition_post_eq', 'area_assesed', 'damage_grade', 'has_geotechnical_risk', 'has_geotechnical_risk_fault_crack', 'has_geotechnical_risk_flood', 'has_geotechnical_risk_land_settlement', 'has_geotechnical_risk_landslide', 'has_geotechnical_risk_liquefaction', 'has_geotechnical_risk_other', 'has_geotechnical_risk_rock_fall', 'has_repair_started']

grade_enc = {
	'Grade 1':1,
	'Grade 2':2,
	'Grade 3':3,
	'Grade 4':4,
	'Grade 5':5
}


# preprocessing

train['dist_vdc_ward_id'] = train['district_id'].map(str) + train['vdcmun_id'].map(str) + train['ward_id'].map(str)
test['dist_vdc_ward_id'] = test['district_id'].map(str) + test['vdcmun_id'].map(str) + test['ward_id'].map(str)

train['count_floors_diff'] = train['count_floors_pre_eq'] - train['count_floors_post_eq']
test['count_floors_diff'] = test['count_floors_pre_eq'] - test['count_floors_post_eq']

train['height_ft_diff'] = train['height_ft_pre_eq'] - train['height_ft_post_eq']
test['height_ft_diff'] = test['height_ft_pre_eq'] - test['height_ft_post_eq']

train['risk_count'] = train['has_geotechnical_risk_other'] + train['has_geotechnical_risk_liquefaction'] + train['has_geotechnical_risk_landslide'] + train['has_geotechnical_risk_flood'] + train['has_geotechnical_risk_rock_fall'] + train['has_geotechnical_risk_land_settlement'] + train['has_geotechnical_risk_fault_crack']
test['risk_count'] = test['has_geotechnical_risk_other'] + test['has_geotechnical_risk_liquefaction'] + test['has_geotechnical_risk_landslide'] + test['has_geotechnical_risk_flood'] + test['has_geotechnical_risk_rock_fall'] + test['has_geotechnical_risk_land_settlement'] + test['has_geotechnical_risk_fault_crack']


imput = Imputer(strategy='most_frequent')
imput1 = Imputer(strategy='mean')
train['count_families'] = imput1.fit_transform(train[['count_families']]).astype('int')
train['has_repair_started'] = imput.fit_transform(train[['has_repair_started']]).astype('int')
train['has_geotechnical_risk'] = train['has_geotechnical_risk'].map(int)
train['has_secondary_use'] = train['has_secondary_use'].map(int)


# test['has_repair_started'] = imput.fit_transform(test[['has_repair_started']]).astype('int')
# test['has_geotechnical_risk'] = test['has_geotechnical_risk'].map(int)
# test['has_secondary_use'] = test['has_secondary_use'].map(int)
# test['count_families'] = test['count_families'].map(int)

train_y = pd.Series([grade_enc[i] for i in train['damage_grade']])
train_y1=[]
train_y1=train_y.values
print(train_y1.shape)

train_x = train.drop(columns=['building_id','damage_grade']) #,'has_secondary_use_rental','has_secondary_use_other','has_superstructure_rc_engineered','has_geotechnical_risk_flood','has_geotechnical_risk_liquefaction','has_secondary_use_industry','has_geotechnical_risk_other','has_secondary_use_institution','has_secondary_use_gov_office','has_secondary_use_use_police','has_secondary_use_health_post','has_secondary_use_school','has_superstructure_bamboo','has_secondary_use_agriculture','has_superstructure_adobe_mud','has_superstructure_cement_mortar_brick','legal_ownership_status','has_superstructure_mud_mortar_brick','has_geotechnical_risk','has_superstructure_rc_non_engineered','has_geotechnical_risk_landslide','has_superstructure_stone_flag','has_secondary_use_hotel','has_superstructure_cement_mortar_stone','has_geotechnical_risk_land_settlement','has_geotechnical_risk_fault_crack','has_superstructure_other','has_geotechnical_risk_rock_fall'])
test_x = test.drop(columns=['building_id']) #,'has_secondary_use_rental','has_secondary_use_other','has_superstructure_rc_engineered','has_geotechnical_risk_flood','has_geotechnical_risk_liquefaction','has_secondary_use_industry','has_geotechnical_risk_other','has_secondary_use_institution','has_secondary_use_gov_office','has_secondary_use_use_police','has_secondary_use_health_post','has_secondary_use_school','has_superstructure_bamboo','has_secondary_use_agriculture','has_superstructure_adobe_mud','has_superstructure_cement_mortar_brick','legal_ownership_status','has_superstructure_mud_mortar_brick','has_geotechnical_risk','has_superstructure_rc_non_engineered','has_geotechnical_risk_landslide','has_superstructure_stone_flag','has_secondary_use_hotel','has_superstructure_cement_mortar_stone','has_geotechnical_risk_land_settlement','has_geotechnical_risk_fault_crack','has_superstructure_other','has_geotechnical_risk_rock_fall'])


print("train_y",train_y.shape)



print('\nlabel encoding....')

for col in test_x.columns.values:
	if test_x[col].dtype == 'object':
		data = train_x[col].append(test_x[col])
		label_enc.fit(data)
		train_x[col] = label_enc.transform(train_x[col])
		test_x[col] = label_enc.transform(test_x[col])



X_train, X_test, y_train, y_test = train_test_split(train_x, train_y1, test_size=0.2)

# print("Train Shape",X_train.shape)
print("Y shape",y_train.shape)
# print("Train SHape",X_train.info())
# print(y_train.head())
# print(y_train.describe())
clf=svm.SVC(decision_function_shape='ovo',verbose=2)

# predictint("outliers")
# 	clf1=svm.OneClassSVM()
# clf1.fit_predict(X_train,y_train)

print("Fit")
clf.fit(X_train,y_train)

y=clf.predict(X_test)

print("ACCURACY:")
print(accuracy_score(y_test,y))


new_time = time.time()

total_time = round((new_time - prev_time),2)
print('\ntotal time : ',total_time)