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


test['has_repair_started'] = imput.fit_transform(test[['has_repair_started']]).astype('int')
test['has_geotechnical_risk'] = test['has_geotechnical_risk'].map(int)
test['has_secondary_use'] = test['has_secondary_use'].map(int)
test['count_families'] = test['count_families'].map(int)

train_y = pd.Series([grade_enc[i] for i in train['damage_grade']])


train_x = train.drop(columns=['building_id','damage_grade']) #,'has_secondary_use_rental','has_secondary_use_other','has_superstructure_rc_engineered','has_geotechnical_risk_flood','has_geotechnical_risk_liquefaction','has_secondary_use_industry','has_geotechnical_risk_other','has_secondary_use_institution','has_secondary_use_gov_office','has_secondary_use_use_police','has_secondary_use_health_post','has_secondary_use_school','has_superstructure_bamboo','has_secondary_use_agriculture','has_superstructure_adobe_mud','has_superstructure_cement_mortar_brick','legal_ownership_status','has_superstructure_mud_mortar_brick','has_geotechnical_risk','has_superstructure_rc_non_engineered','has_geotechnical_risk_landslide','has_superstructure_stone_flag','has_secondary_use_hotel','has_superstructure_cement_mortar_stone','has_geotechnical_risk_land_settlement','has_geotechnical_risk_fault_crack','has_superstructure_other','has_geotechnical_risk_rock_fall'])
test_x = test.drop(columns=['building_id']) #,'has_secondary_use_rental','has_secondary_use_other','has_superstructure_rc_engineered','has_geotechnical_risk_flood','has_geotechnical_risk_liquefaction','has_secondary_use_industry','has_geotechnical_risk_other','has_secondary_use_institution','has_secondary_use_gov_office','has_secondary_use_use_police','has_secondary_use_health_post','has_secondary_use_school','has_superstructure_bamboo','has_secondary_use_agriculture','has_superstructure_adobe_mud','has_superstructure_cement_mortar_brick','legal_ownership_status','has_superstructure_mud_mortar_brick','has_geotechnical_risk','has_superstructure_rc_non_engineered','has_geotechnical_risk_landslide','has_superstructure_stone_flag','has_secondary_use_hotel','has_superstructure_cement_mortar_stone','has_geotechnical_risk_land_settlement','has_geotechnical_risk_fault_crack','has_superstructure_other','has_geotechnical_risk_rock_fall'])


print(test.shape)



print('\nlabel encoding....')

for col in test_x.columns.values:
	if test_x[col].dtype == 'object':
		data = train_x[col].append(test_x[col])
		label_enc.fit(data)
		train_x[col] = label_enc.transform(train_x[col])
		test_x[col] = label_enc.transform(test_x[col])

# print(test_x.shape)
# print(test_x.dtypes)

print('\ntraining model....')

clf = RandomForestClassifier()
# clf.fit(train_x,train_y)

# grid-search-cv

params = {
	'n_jobs': [-1],
	'oob_score' : [True],
	'random_state' : [321],
	'n_estimators' : [25,45,50,80,100,120],
	'max_features' : ['auto',.2,'sqrt',.5],
	'max_depth' : [5,10,15,20,25,35,40,50,70,None],
	'min_samples_leaf' : [1,5,10,20,25,30,40,50,70,80,95]
}

grid_clf = GridSearchCV(clf,params,verbose=2,refit=True)

print('gridsearch started')

grid_clf.fit(train_x,train_y)

print('gridsearch completed')

with open(result_filepath,'w') as result_file:
	result_file.write('\n\nparameters applied : ',params)
	result_file.write('\n\nresults : ',grid_clf.cv_results_)
	result_file.write('\n\nbest score : ', grid_clf.best_score_)
	result_file.write('\n\nbest parameters : ', grid_clf.best_params_)
	result_file.write('\n\nbest index : ', grid_clf.best_index_)
	result_file.write('\n\nrefit time : ', grid_clf.refit_time_)


print('\npredicting data....')


predictions_train = clf.predict(train_x)
predictions_test = clf.predict(test_x)


print('\ncalculation accuracy....')

print("\nTrain Accuracy : ",accuracy_score(train_y,predictions_train))

print("\nConfusion Matrix : ")
print(confusion_matrix(train_y,predictions_train))

predictions_test = pd.Series(predictions_test)

# predict_test = pd.Series([grade_dec[str(i)] for i in predictions_test[i]])
predict_test = 'Grade ' + predictions_test.map(str)

print(len(test['building_id']))
print(len(predictions_test))
subm = {
	'building_id' : test['building_id'],
	'damage_grade' : predict_test
}
submissions = pd.DataFrame(subm)
submissions = submissions.set_index('building_id')

print(submissions.head())


importances = pd.DataFrame({'feature':train_x.columns,'importance':np.round(clf.feature_importances_,4)})
importances = importances.sort_values('importance',ascending=False).set_index('feature')

print(importances)

submissions.to_csv('submission/2.csv')

new_time = time.time()

total_time = round((new_time - prev_time),2)
print('\ntotal time : ',total_time)