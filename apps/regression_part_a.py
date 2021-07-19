import sys; sys.path.insert(0, '../')
import json
import matplotlib.pyplot as plt
import numpy as np
from libs.dataset import DataSet
from libs.models import LinearRegression


sys.stdout = open('logs/regression_part_a.log', 'w')
filepath_1 = '../data/adult.data'
filepath_2 = '../data/adult.test'
predicted_attributes = ['capital-gain', 'capital-loss']
dnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'annual-income']
dformats = [int, str, int, str, int, str, str, str, str, str, int, int, int, str, str]
K = 10
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
lamda_array = np.arange(1e-6, 1e5, 1e3)
lamda_optimal = {}
model = LinearRegression(max_iter=1000, tol=1e-12)
fig, ax = plt.subplots(figsize=(8, 6))
ax = [ax, ax.twinx()]
colors = ['b', 'r']
fig.subplots_adjust(top=0.9, bottom=0.1, left=0.10, right=0.9, wspace=0.25, hspace=0.25)
print('(1) Finding optimal regularization parameter.')
for i,predicted_attribute in enumerate(predicted_attributes):
	print('>> Predicted attribute:', predicted_attribute)
	X = ds.get_data_but_exception(predicted_attribute)
	y = ds.get_attribute_values(predicted_attribute)
	gen_error_array = []
	gen_error_min = 1e308
	for lamda in lamda_array:
		model.update_params(lamda=lamda)
		gen_error = model.get_error_k_fold(X, y, n_splits=K)
		if gen_error < gen_error_min:
			gen_error_min = gen_error
			lamda_optimal[predicted_attribute] = lamda
		gen_error_array.append(gen_error)
		print('Generalization error ({}-fold cross-validation):'.format(K), gen_error)
	gen_error_array = np.array(gen_error_array)
	ax[i].plot(lamda_array, gen_error_array, '-o', color=colors[i], mec='k', mew=.5, label=predicted_attribute)
	ax[i].set_ylabel('generalization error for {}'.format(predicted_attribute))
ax[0].set_xlabel('regularization parameter')
fig.legend(loc='upper center', ncol=2)
plt.savefig('figs/lin_reg_error_vs_regul.png')
print('')
print('(2) Getting coefficients for optimal regularization parameter.')
json_dumps = {'Coefficients': {}, 'Optimal_lamda': lamda_optimal}
for i,predicted_attribute in enumerate(predicted_attributes):
	print('>> Predicted attribute:', predicted_attribute)
	model.update_params(lamda=lamda_optimal[predicted_attribute])
	X = ds.get_data_but_exception(predicted_attribute)
	y = ds.get_attribute_values(predicted_attribute)
	model.fit(X, y)
	dnames = ds.get_names_but_exception(predicted_attribute)
	coefficients = model.get_coefficients()
	print('Attributes:', dnames)
	print('Coefficients', coefficients)
	json_dumps['Coefficients'][predicted_attribute] = {}
	for j,dname in enumerate(dnames):
		json_dumps['Coefficients'][predicted_attribute][dname] = coefficients[i]
json_file =	 open('results/regression_part_a.json', 'w')
json_file.write(json.dumps(json_dumps, indent="\t", separators=(",", ": ")))
json_file.close()
sys.stdout.close()