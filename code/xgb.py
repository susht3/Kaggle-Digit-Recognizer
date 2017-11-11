import re
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
from sklearn import preprocessing
from sklearn.preprocessing import  Imputer
from sklearn.preprocessing import LabelEncoder

def fit_model(alg, x, y, useTrainCV=True, cv_folds=5, early_stopping_rounds=60):
	print('Fitting model...')
	if useTrainCV:
		xgb_param = alg.get_xgb_params()
		xgtrain = xgb.DMatrix(x.values, label=y.values)
		cvresult = xgb.cv(xgb_param, xgtrain, 
			num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,num_class=10,
            metrics='logloss', early_stopping_rounds=early_stopping_rounds)
		alg.set_params(n_estimators=cvresult.shape[0])
    
	alg.fit(x, y, eval_metric='None')

	dtrain_predictions = alg.predict(x)
	dtrain_predprob = alg.predict_proba(x)[:,1]
    
	print ("Accuracy : %.4g" % metrics.accuracy_score(y.values, dtrain_predictions))
	#print ("AUC Score (Train): %f" % metrics.roc_auc_score(y, dtrain_predprob))
    
	return alg 


def test_model(alg, x, id_list):
	print('Testing model...')
	dtest_predictions = alg.predict(x)
	dtest_predprob = alg.predict_proba(x)[:,1]   

	print(type(dtest_predictions))

	res = pd.DataFrame({'PassengerId':id_list, 'Survived':dtest_predictions})
	res.to_csv('../data/submission_v1.csv', index=False)
	print('Test over')


def baseline(x, y):
	#Accuracy : 0.8866   AUC Score (Train): 0.945776
	xgb = XGBClassifier(
		learning_rate =0.1,
		n_estimators=5000,
		max_depth=4,
		min_child_weight=3,
		gamma=0.1,
		subsample=0.9,
		colsample_bytree=0.6,
		reg_alpha=0.1,
		objective= 'multi:softmax', 
		nthread=4,
		scale_pos_weight=1,
		seed=27)

	alg = fit_model(xgb, x, y)
	return alg


def tune_model(x, y):
	print('Tuning model... ')
	param_test = {

	# {'max_depth': 4, 'min_child_weight': 3}   0.8738239080307759
	#'max_depth': [i for i in range(3,10)],
	#min_child_weight': [i for i in range(1,6)]

	# {'gamma': 0.1} 0.8738239080307759
	#'gamma':[i/10.0 for i in range(0,5)]

	# {'colsample_bytree': 0.9, 'subsample': 0.6} 0.8772317173234019
	#'subsample':[i/10.0 for i in range(6,10)],
	#'colsample_bytree':[i/10.0 for i in range(6,10)]
	
	#'subsample':[i/100.0 for i in range(85,100,5)],
	#'colsample_bytree':[i/100.0 for i in range(85,100,5)]
        
	#{'reg_alpha': 0.1}  0.874364387557975
	#'reg_alpha':[1e-4, 1e-6, 1e-3, 1e-5, 1e-2, 0.1, 1, 100]
    
	#{'n_estimators': 140} 0.8849902113415625
	#'n_estimators': [140, 100, 80, 200]
	#'scale_pos_weight': [1]
        
	#{'learning_rate': 0.1} 0.874364387557975
	'learning_rate':[1e-4, 1e-3, 1e-5, 1e-2, 0.1]
	}

	gsearch = GridSearchCV(estimator = XGBClassifier( learning_rate =0.1, 
		n_estimators=140, max_depth=4, min_child_weight=3, gamma=0.1, subsample=0.9, 
		colsample_bytree=0.6, reg_alpha=0.1, objective= 'multi:softmax', num_class=10, nthread=4, scale_pos_weight=1, seed=27), 
		param_grid = param_test, scoring='None',n_jobs=4,iid=False, cv=5)

	gsearch.fit(x, y)
	print(gsearch.grid_scores_, '\n', gsearch.best_params_, '\n', gsearch.best_score_)


def get_minmax_data(train_x, test_x):
	train_imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
	train_imp.fit(train_x)
	train_x = train_imp.transform(train_x)
	test_x = train_imp.transform(test_x)
	
	min_max_scaler = preprocessing.MinMaxScaler()
	minmax_train = min_max_scaler.fit_transform(train_x)
	minmax_test = min_max_scaler.transform(test_x)	

	train_x = pd.DataFrame(minmax_train)
	test_x = pd.DataFrame(minmax_test)

	return train_x, test_x


def get_norm_data(train_x, test_x):
	train_imp = Imputer(missing_values='NaN',strategy='mean',axis=0)
	train_imp.fit(train_x)
	train_x = train_imp.transform(train_x)
	test_x = train_imp.transform(test_x)

	normalizer = preprocessing.Normalizer().fit(train_x)
	train_x = normalizer.transform(train_x)
	test_x = normalizer.transform(test_x)

	train_x = pd.DataFrame(train_x)
	test_x = pd.DataFrame(test_x)

	return train_x, test_x


def get_clean_x(dtrain, att_list, str_list):
	train_x = dtrain[att_list]

	'''
	for feature in str_list:
		encoder = LabelEncoder()
		dtrain_x[feature] = encoder.fit_transform(dtrain[feature])
	'''

	train_x['Sex'] = pd.factorize(dtrain.Sex)[0]
	train_x['Embarked'] = pd.factorize(dtrain.Embarked)[0]

	train_x = np.asarray(train_x)
	print('x shape: ', train_x.shape)

	return train_x

	
if __name__ == '__main__':
	train_path = '../data/train.csv'
	test_path = '../data/test.csv'
	dtrain = pd.read_csv(train_path)
	dtest = pd.read_csv(test_path)

	#test_id_list = dtest['Imageld']

	pixel_list = [p for p in dtrain.columns if p != 'label']
	train_x = dtrain[pixel_list]
	train_y = dtrain['label']

	test_x = dtest
	train_x, test_x = get_minmax_data(train_x, test_x)

	alg = baseline(train_x, train_y)
	#test_model(alg, test_x, test_id_list)

	#tune_model(train_x, train_y)


