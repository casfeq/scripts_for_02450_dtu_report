import sys; sys.path.insert(0, '../')
import json
import matplotlib.pyplot as plt
import numpy as np
from libs.dataset import DataSet
from libs.models import ANNRegression, BaselineRegression, LinearRegression
from libs.tests import TStudent


sys.stdout = open('logs/regression_part_b.log', 'w')
filepath_1 = '../data/adult.data'
filepath_2 = '../data/adult.test'
predicted_attributes = ['capital-gain', 'capital-loss']
dnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'annual-income']
dformats = [int, str, int, str, int, str, str, str, str, str, int, int, int, str, str]
inner_splits = 5
outer_splits = 5
ds = DataSet()
ds.load_data(filepath_1, dtype=np.ndarray, delimiter=', ')
ds.append_data(filepath_2, dtype=np.ndarray, delimiter=', ')
ds.set_attributes_name(dnames)
ds.set_attributes_format(dformats)
ds.remove_missing_values()
ds.bool_attribute(dname='annual-income', true_ref='>50K')
ds.bool_attribute(dname='sex', true_ref='Male')
ds.encode_as_1_out_of_K()
ds.normalize_all_attributes()
blr = BaselineRegression()
rlr = LinearRegression(max_iter=1000, tol=1e-12)
ann = ANNRegression(n_attributes=ds.get_num_of_attributes()-1, max_iter=2000, n_replicates=3)
lamda_array = np.arange(1e4, 5e4, 5e3)
n_hidden_array = np.arange(1, 16, 1)
n_hidden_optimal = {}
json_dumps = {}
fig, ax = plt.subplots(figsize=(8, 6))
ax = [ax, ax.twinx()]
colors = ['b', 'r']
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.9, wspace=0.25, hspace=0.25)
print('(1) Finding optimal number of hidden units for Artificial Neural Network.')
for i,predicted_attribute in enumerate(predicted_attributes):
	print('>> Predicted attribute:', predicted_attribute)
	X = ds.get_data_but_exception(predicted_attribute)
	y = ds.get_attribute_values(predicted_attribute)
	gen_error_array = []
	gen_error_min = 1e308
	for n_hidden in n_hidden_array:
		ann.update_params(n_hidden_units=n_hidden)
		gen_error = ann.get_error_k_fold(X, y, n_splits=outer_splits)
		if gen_error < gen_error_min:
			gen_error_min = gen_error
			n_hidden_optimal[predicted_attribute] = int(n_hidden)
		gen_error_array.append(gen_error)
		print('Generalization error ({}-fold cross-validation):'.format(outer_splits), gen_error)
	ax[i].plot(n_hidden_array, gen_error_array, '-o', color=colors[i], mec='k', mew=.5, label=predicted_attribute)
	ax[i].set_ylabel('generalization error for {}'.format(predicted_attribute))
ax[0].set_xlabel('Number of hidden units')
fig.legend(loc='upper center', ncol=2)
plt.savefig('figs/ann_error_vs_n_hidden.png')
print('')
json_dumps['Optimal_n_hidden_units'] = n_hidden_optimal
json_file_part_a = open('results/regression_part_a.json', 'r')
lamda_optimal = json.loads(json_file_part_a.read())['Optimal_lamda']
json_dumps['2_level_cross_validation'] = {}
print('(2) Performing two-level {}-{}-fold cross validation.'.format(outer_splits, inner_splits))
for i,predicted_attribute in enumerate(predicted_attributes):
	json_dumps['2_level_cross_validation'][predicted_attribute] = {'Baseline_regression': {'Errors': None}, 'Regularized_linear_regression': {'Errors': None, 'Params': None}, 'Artificial_neural_network': {'Errors': None, 'Params': None}}
	print('>> Predicted attribute:', predicted_attribute)
	X = ds.get_data_but_exception(predicted_attribute)
	y = ds.get_attribute_values(predicted_attribute)
	gen_error, best_params, test_errors = blr.get_error_k_fold_2_level(X, y, inner_splits=inner_splits, outer_splits=outer_splits, param_name='lamda', param_array=[0.]*2)
	json_dumps['2_level_cross_validation'][predicted_attribute]['Baseline_regression']['Errors'] = test_errors.tolist()
	print('Generalization error ({}-{}-fold cross-validation) for Baseline Regression:'.format(outer_splits, inner_splits), gen_error)
	gen_error, best_params, test_errors = rlr.get_error_k_fold_2_level(X, y, inner_splits=inner_splits, outer_splits=outer_splits, param_name='lamda', param_array=lamda_array)
	json_dumps['2_level_cross_validation'][predicted_attribute]['Regularized_linear_regression']['Errors'] = test_errors.tolist()
	json_dumps['2_level_cross_validation'][predicted_attribute]['Regularized_linear_regression']['Params'] = best_params.tolist()
	print('Generalization error ({}-{}-fold cross-validation) for Regularized Linear Regression:'.format(outer_splits, inner_splits), gen_error)
	gen_error, best_params, test_errors = ann.get_error_k_fold_2_level(X, y, inner_splits=inner_splits, outer_splits=outer_splits, param_name='n_hidden_units', param_array=n_hidden_array)
	json_dumps['2_level_cross_validation'][predicted_attribute]['Artificial_neural_network']['Errors'] = test_errors.tolist()
	json_dumps['2_level_cross_validation'][predicted_attribute]['Artificial_neural_network']['Params'] = best_params.tolist()
	print('Generalization error ({}-{}-fold cross-validation) for Artificial Neural Network:'.format(outer_splits, inner_splits), gen_error)
print('')
json_dumps['Paired_t_tests'] = {'Test_1': {}, 'Test_2': {}, 'Test_3': {}}
print('(3) Performing paired t-tests.')
for i,predicted_attribute in enumerate(predicted_attributes):
	print('>> Predicted attribute:', predicted_attribute)
	ds.split(test_size=.15, random_state=0)
	X_train = ds.get_data_but_exception(predicted_attribute)
	y_train = ds.get_attribute_values(predicted_attribute)
	X_test = ds.get_test_set_but_exception(predicted_attribute)
	y_test = ds.get_test_attribute_values(predicted_attribute)
	blr.fit(X_train, y_train)
	rlr.update_params(lamda=lamda_optimal[predicted_attribute])
	rlr.fit(X_train, y_train)
	ann.update_params(n_hidden_units=n_hidden_optimal[predicted_attribute])
	ann.fit(X_train, y_train)
	print('Performing first paired test.')
	print('Model A: Artificial Neural Network')
	print('Model B: Linear Regression')
	test = TStudent(alpha=0.05, model_a=ann, model_b=blr)
	test.set_test_data(X_test, y_test)
	confidence_interval, p = test.perform_test()
	print('Confidence interval:', confidence_interval)
	print('p-value:', p)
	json_dumps['Paired_t_tests']['Test_1'][predicted_attribute] = {'Model_a': 'Artificial_neural_network', 'Model_b': 'Baseline_regression', 'confidence_interval': confidence_interval, 'p-value': p}
	print('Performing second paired test.')
	print('Model A: Artificial Neural Network')
	print('Model B: Regularized Linear Regression')
	test.set_models(model_a=ann, model_b=rlr)
	confidence_interval, p = test.perform_test()
	print('Confidence interval:', confidence_interval)
	print('p-value:', p)
	json_dumps['Paired_t_tests']['Test_2'][predicted_attribute] = {'Model_a': 'Artificial_neural_network', 'Model_b': 'Regularized_linear_regression', 'confidence_interval': confidence_interval, 'p-value': p}
	print('Performing third paired test.')
	print('Model A: Regularized Linear Regression')
	print('Model B: Linear Regression')
	test.set_models(model_a=rlr, model_b=blr)
	confidence_interval, p = test.perform_test()
	print('Confidence interval:', confidence_interval)
	print('p-value:', p)
	json_dumps['Paired_t_tests']['Test_3'][predicted_attribute] = {'Model_a': 'Regularized_linear_regression', 'Model_b': 'Baseline_regression', 'confidence_interval': confidence_interval, 'p-value': p}
json_file =	 open('results/regression_part_b.json', 'w')
json_file.write(json.dumps(json_dumps, indent="\t", separators=(",", ": ")))
json_file.close()
sys.stdout.close()