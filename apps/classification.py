import sys; sys.path.insert(0, '../')
import json
import matplotlib.pyplot as plt
import numpy as np
from libs.dataset import DataSet
from libs.models import BaselineClassifier, DecisionTreeClassifier, LogisticRegression
from libs.tests import McNemar


sys.stdout = open('logs/classification.log', 'w')
filepath_1 = '../data/adult.data'
filepath_2 = '../data/adult.test'
predicted_attributes = ['annual-income']
dnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'annual-income']
dformats = [int, str, int, str, int, str, str, str, str, str, int, int, int, str, str]
inner_splits = 10
outer_splits = 10
ds = DataSet()
ds.load_data(filepath_1, dtype=np.ndarray, delimiter=', ')
ds.append_data(filepath_2, dtype=np.ndarray, delimiter=', ')
ds.set_attributes_name(dnames)
ds.set_attributes_format(dformats)
ds.remove_missing_values()
ds.bool_attribute(dname='annual-income', true_ref='>50K')
ds.bool_attribute(dname='sex', true_ref='Male')
ds.encode_as_1_out_of_K()
ds.format_all_attributes(int)
blc = BaselineClassifier()
dtc = DecisionTreeClassifier(criterion='gini', min_samples_split=2)
rlr = LogisticRegression(max_iter=1000, tol=1e-12)
lamda_array = np.arange(0, 1e4, 1e2)
lamda_optimal = {}
max_depth_array = np.arange(1, 51, 1)
max_depth_optimal = {}
json_dumps = {'2_level_cross_validation': {}}
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['b']
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.9, wspace=0.25, hspace=0.25)
print('(1) Finding optimal number of max depth for Decision Tree Classifier.')
for i,predicted_attribute in enumerate(predicted_attributes):
	print('>> Predicted attribute:', predicted_attribute)
	X = ds.get_data_but_exception(predicted_attribute)
	y = ds.get_attribute_values(predicted_attribute)
	gen_error_array = []
	gen_error_min = 1e308
	for max_depth in max_depth_array:
		dtc.update_params(max_depth=max_depth)
		gen_error = dtc.get_error_k_fold(X, y, n_splits=outer_splits)
		if gen_error < gen_error_min:
			gen_error_min = gen_error
			max_depth_optimal[predicted_attribute] = int(max_depth)
		gen_error_array.append(gen_error)
		print('Generalization error ({}-fold cross-validation):'.format(outer_splits), gen_error)
	ax.plot(max_depth_array, gen_error_array, '-o', color=colors[i], mec='k', mew=.5, label=predicted_attribute)
	ax.set_ylabel('generalization error for {}'.format(predicted_attribute))
ax.set_xlabel('Max depth')
fig.legend(loc='upper center', ncol=2)
plt.savefig('figs/dt_class_error_vs_depth.png')
json_dumps['Optimal_max_depth'] = max_depth_optimal
print('')
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['b']
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.95, wspace=0.25, hspace=0.25)
print('(2) Finding optimal number of regularization parameter for Logistic Regression.')
for i,predicted_attribute in enumerate(predicted_attributes):
	print('>> Predicted attribute:', predicted_attribute)
	X = ds.get_data_but_exception(predicted_attribute)
	y = ds.get_attribute_values(predicted_attribute)
	gen_error_array = []
	gen_error_min = 1e308
	for lamda in lamda_array:
		rlr.update_params(lamda=lamda)
		gen_error = rlr.get_error_k_fold(X, y, n_splits=outer_splits)
		if gen_error < gen_error_min:
			gen_error_min = gen_error
			lamda_optimal[predicted_attribute] = int(lamda)
		gen_error_array.append(gen_error)
		print('Generalization error ({}-fold cross-validation):'.format(outer_splits), gen_error)
	ax.plot(lamda_array, gen_error_array, '-o', color=colors[i], mec='k', mew=.5, label=predicted_attribute)
	ax.set_ylabel('generalization error for {}'.format(predicted_attribute))
ax.set_xlabel('Regularization parameter')
fig.legend(loc='upper center', ncol=2)
plt.savefig('figs/log_reg_error_vs_regul.png')
json_dumps['Optimal_lamda'] = lamda_optimal
print('')
print('(3) Performing two-level {}-{}-fold cross validation.'.format(outer_splits, inner_splits))
for i,predicted_attribute in enumerate(predicted_attributes):
	json_dumps['2_level_cross_validation'][predicted_attribute] = {'Baseline_classifier': {'Errors': None}, 'Decision_tree_classifier': {'Errors': None, 'Params': None}, 'Regularized_logistic_regression': {'Errors': None, 'Params': None}}
	print('>> Predicted attribute:', predicted_attribute)
	X = ds.get_data_but_exception(predicted_attribute)
	y = ds.get_attribute_values(predicted_attribute)
	gen_error, best_params, test_errors = blc.get_error_k_fold_2_level(X, y, inner_splits=inner_splits, outer_splits=outer_splits, param_name='lamda', param_array=[0.]*2)
	json_dumps['2_level_cross_validation'][predicted_attribute]['Baseline_classifier']['Errors'] = test_errors.tolist()
	print('Generalization error ({}-{}-fold cross-validation) for Baseline Classifier:'.format(outer_splits, inner_splits), gen_error)
	gen_error, best_params, test_errors = dtc.get_error_k_fold_2_level(X, y, inner_splits=inner_splits, outer_splits=outer_splits, param_name='max_depth', param_array=max_depth_array)
	json_dumps['2_level_cross_validation'][predicted_attribute]['Decision_tree_classifier']['Errors'] = test_errors.tolist()
	json_dumps['2_level_cross_validation'][predicted_attribute]['Decision_tree_classifier']['Params'] = best_params.tolist()
	print('Generalization error ({}-{}-fold cross-validation) for Decision Tree Classifier:'.format(outer_splits, inner_splits), gen_error)
	gen_error, best_params, test_errors = rlr.get_error_k_fold_2_level(X, y, inner_splits=inner_splits, outer_splits=outer_splits, param_name='lamda', param_array=lamda_array)
	json_dumps['2_level_cross_validation'][predicted_attribute]['Regularized_logistic_regression']['Errors'] = test_errors.tolist()
	json_dumps['2_level_cross_validation'][predicted_attribute]['Regularized_logistic_regression']['Params'] = best_params.tolist()
	print('Generalization error ({}-{}-fold cross-validation) for Regularized Logistic Regression:'.format(outer_splits, inner_splits), gen_error)
print('')
json_dumps['Paired_mcnemar_tests'] = {}
print('(4) Performing paired McNemar tests.')
for i,predicted_attribute in enumerate(predicted_attributes):
	print('>> Predicted attribute:', predicted_attribute)
	ds.split(test_size=.1, random_state=0)
	X_train = ds.get_data_but_exception(predicted_attribute)
	y_train = ds.get_attribute_values(predicted_attribute)
	X_test = ds.get_test_set_but_exception(predicted_attribute)
	y_test = ds.get_test_attribute_values(predicted_attribute)
	blc.fit(X_train, y_train)
	dtc.update_params(max_depth=max_depth_optimal[predicted_attribute])
	dtc.fit(X_train, y_train)
	rlr.update_params(lamda=lamda_optimal[predicted_attribute])
	rlr.fit(X_train, y_train)
	print('Performing first paired test.')
	print('Model A: Regularized Logistic Regression')
	print('Model B: Baseline Classifier')
	test = McNemar(alpha=0.05, model_a=rlr, model_b=blc)
	test.set_test_data(X_test, y_test)
	theta, confidence_interval, p = test.perform_test()
	print('Theta:', theta)
	print('Confidence interval:', confidence_interval)
	print('p-value:', p)
	json_dumps['Paired_mcnemar_tests']['Test_1'] = {'Model_a': 'Regularized_logistic_regression', 'Model_b': 'Baseline_classifier', 'theta': theta, 'confidence_interval': confidence_interval, 'p-value': p}
	print('Performing second paired test.')
	print('Model A: Regularized Logistic Regression')
	print('Model B: Decision Tree Classifier')
	test.set_models(model_a=rlr, model_b=dtc)
	theta, confidence_interval, p = test.perform_test()
	print('Theta:', theta)
	print('Confidence interval:', confidence_interval)
	print('p-value:', p)
	json_dumps['Paired_mcnemar_tests']['Test_2'] = {'Model_a': 'Regularized_logistic_regression', 'Model_b': 'Decision_tree_classifier', 'theta': theta, 'confidence_interval': confidence_interval, 'p-value': p}
	print('Performing third paired test.')
	print('Model A: Decision Tree Classifier')
	print('Model B: Baseline Classifier')
	test.set_models(model_a=dtc, model_b=blc)
	theta, confidence_interval, p = test.perform_test()
	print('Theta:', theta)
	print('Confidence interval:', confidence_interval)
	print('p-value:', p)
	json_dumps['Paired_mcnemar_tests']['Test_3'] = {'Model_a': 'Decision_tree_classifier', 'Model_b': 'Baseline_classifier', 'theta': theta, 'confidence_interval': confidence_interval, 'p-value': p}
json_file =	 open('results/classification.json', 'w')
json_file.write(json.dumps(json_dumps, indent="\t", separators=(",", ": ")))
json_file.close()
sys.stdout.close()