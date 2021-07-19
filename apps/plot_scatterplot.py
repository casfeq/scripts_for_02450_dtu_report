import sys; sys.path.insert(0, '../')
from matplotlib import pyplot as plt
import numpy as np
from libs.dataset import DataSet


attribs = ['age', 'fnlwgt', 'education-num', 'capital-gain', 'capital-loss', 'hours-per-week']
dfilters = ['race', 'sex', 'workclass']
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
for attrib in attribs:
	ds.normalize_attribute(dname=attrib)
fig = plt.figure(figsize=(8, 8))
fig.subplots_adjust(top=0.95, bottom=0.15, left=0.10, right=0.95, wspace=0.25, hspace=0.25)
for i in range(6):
	for j in range(6):
		plt.subplot(6, 6, i*6 + j + 1)
		x = ds.get_attribute_values(dname=attribs[j])
		y = ds.get_attribute_values(dname=attribs[i])
		plt.plot(x, y, '.', alpha=0.4, mec='k', mew=0.5)
		if i == 5:
			plt.xlabel(attribs[j])
		else:
			plt.xticks([])
		if j == 0:
			plt.ylabel(attribs[i])
		else:
			plt.yticks([])
plt.suptitle('adult data set - scatterplot')
plt.savefig('figs/scatterplot_adult.png')
plt.savefig('figs/scatterplot_adult.pdf')
for dfilter in dfilters:
	fig = plt.figure(figsize=(8, 8))
	fig.subplots_adjust(top=0.95, bottom=0.15, left=0.10, right=0.95, wspace=0.25, hspace=0.25)
	for i in range(6):
		for j in range(6):
			plt.subplot(6, 6, i*6 + j + 1)
			for dname in ds.get_attribute_unique_values(dfilter):
				x = ds.get_attribute_filtered_values(dname=attribs[j], dfilter=dname)
				y = ds.get_attribute_filtered_values(dname=attribs[i], dfilter=dname)
				if (i == 0 and j == 0):
					plt.plot(x, y, '.', label=dname, alpha=0.4, mec='k', mew=0.5)
				else:
					plt.plot(x, y, '.', alpha=0.4, mec='k', mew=0.5)
			if i == 5:
				plt.xlabel(attribs[j])
			else:
				plt.xticks([])
			if j == 0:
				plt.ylabel(attribs[i])
			else:
				plt.yticks([])
	plt.suptitle('adult data set - scatterplot: {}'.format(dfilter))
	fig.legend(loc='lower center', ncol=min(len(ds.get_attribute_values(dfilter)), 4))
	plt.savefig('figs/scatterplot_adult_{}.png'.format(dfilter))
	plt.savefig('figs/scatterplot_adult_{}.pdf'.format(dfilter))
