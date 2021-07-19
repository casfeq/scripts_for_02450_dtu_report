import sys; sys.path.insert(0, '../')
from matplotlib import pyplot as plt
import numpy as np
from libs.dataset import DataSet


attribs = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
filepath_1 = '../data/adult.data'
filepath_2 = '../data/adult.test'
dnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'annual-income']
dformats = [int, str, float, str, int, str, str, str, str, str, int, int, int, str, str]
ds = DataSet()
ds.load_data(filepath_1, dtype=np.ndarray, delimiter=', ')
ds.append_data(filepath_2, dtype=np.ndarray, delimiter=', ')
ds.set_attributes_name(dnames)
ds.set_attributes_format(dformats)
ds.remove_missing_values()
x = []
fig = plt.figure(figsize=(5, 5))
fig.subplots_adjust(top=0.90, bottom=0.25, left=0.15, right=0.96)
for attrib in attribs:
	ds.normalize_attribute(dname=attrib)
	x.append(ds.get_attribute_values(dname=attrib))
plt.boxplot(x)
plt.suptitle('adult data set - boxplot')
plt.xticks(range(1, 7), attribs, rotation='vertical')
plt.ylabel('normalized attribute value')
plt.savefig('figs/boxplot_adult.pdf')
plt.savefig('figs/boxplot_adult.png')