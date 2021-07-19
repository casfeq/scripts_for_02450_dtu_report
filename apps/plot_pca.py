import sys; sys.path.insert(0, '../')
from matplotlib import pyplot as plt
import numpy as np
from libs.dataset import DataSet


filepath_1 = '../data/adult.data'
filepath_2 = '../data/adult.test'
dfilters = ['age', 'fnlwgt', 'education-num', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'annual-income']
dnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'annual-income']
dformats = [int, str, float, str, int, str, str, str, str, str, int, int, int, str, str]
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
ds.compute_principal_components()
ds.compute_variance_by_pca()
rho = ds.get_variance()
threshold = 0.9
plt.figure()
plt.plot([1, len(rho)], [threshold, threshold], 'k--', label='Threshold ({})'.format(str(threshold)))
plt.plot(range(1, len(rho) + 1), rho, 's-', markersize=5, mec='k', mew=0.5, label='Individual')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-', markersize=5, mec='k', mew=0.5, label='Cumulative')
plt.title('adult data set - variance by PCA');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend()
plt.grid()
plt.savefig('figs/varianceplot.png')
plt.savefig('figs/varianceplot.pdf')
ds.filter_data(dfilters)
ds.compute_principal_components()
ds.compute_variance_by_pca()
rho = ds.get_variance()
threshold = 0.9
plt.figure()
plt.plot([1, len(rho)], [threshold, threshold], 'k--', label='Threshold ({})'.format(str(threshold)))
plt.plot(range(1, len(rho) + 1), rho, 's-', markersize=5, mec='k', mew=0.5, label='Individual')
plt.plot(range(1, len(rho) + 1), np.cumsum(rho), 'o-', markersize=5, mec='k', mew=0.5, label='Cumulative')
plt.title('adult data set - variance by PCA: reduced');
plt.xlabel('Principal component');
plt.ylabel('Variance explained');
plt.legend()
plt.grid()
plt.savefig('figs/varianceplot_reduced.png')
plt.savefig('figs/varianceplot_reduced.pdf')
dlabels = ['age', 'fnlwgt', 'edu.-num', 'sex', 'cap.-gain', 'cap.-loss', 'h.-per-wk.', 'anual-in.']
fig = plt.figure(figsize=(7, 5))
fig.subplots_adjust(top=0.95, bottom=0.2, left=0.10, right=0.95, wspace=0.25, hspace=0.25)
V = ds.get_principal_components()
bw = .8/len(dlabels)
r = np.arange(1, len(dlabels) + 1)
for i,dfilter in enumerate(dlabels):
    plt.bar(r + (i + 3/2 - len(dlabels)/2)*bw, V[:, i], width=bw, align='center', label='PC{}'.format(i + 1))
plt.xticks(r + bw, dlabels, fontsize=8)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
fig.legend(loc='lower center', ncol=4)
plt.grid()
plt.title('adult data set - PC coefficients')
plt.savefig('figs/principal_components_reduced.png')
plt.savefig('figs/principal_components_reduced.pdf')