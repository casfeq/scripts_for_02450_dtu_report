import sys; sys.path.insert(0, '../')
from matplotlib import pyplot as plt
import numpy as np
from libs.dataset import DataSet


attribs = ['age', 'fnlwgt', 'education-num', 'hours-per-week','capital-gain', 'capital-loss']
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
	nbins = ds.get_attribute_num_of_unique(attrib)
	x = ds.get_attribute_values(dname=attrib)
	mean = ds.get_attribute_mean(dname=attrib)
	stdev = ds.get_attribute_stdev(dname=attrib)
	y = np.random.normal(loc=mean, scale=stdev, size=1000000)
	fig = plt.figure(figsize=(10, 4))
	fig.subplots_adjust(top=0.90, bottom=0.20, left=0.12, right=0.96, wspace=0.35)
	plt.subplot(1, 2, 1)
	plt.hist(x, color='b', bins=nbins, histtype='step', density=True)
	plt.hist(y, color='r', bins=nbins, histtype='step', density=True)
	ax = plt.gca()
	ax.set_ylabel('density')
	ax.set_xlabel(attrib)
	plt.subplot(1, 2, 2)
	plt.hist(x, color='b', bins=nbins, label='data', histtype='step', density=True, cumulative=True)
	plt.hist(y, color='r', bins=nbins, label='normal', histtype='step', density=True, cumulative=True)
	ax = plt.gca()
	ax.set_ylabel('cumulative density')
	ax.set_xlabel(attrib)
	fig.suptitle('adult data set - distribution: {}'.format(attrib))
	fig.legend(loc='lower center', ncol=2)
	plt.savefig('figs/histogram_adult_{}.pdf'.format(attrib))
	plt.savefig('figs/histogram_adult_{}.png'.format(attrib))
attribs = ['capital-gain', 'capital-loss']
for attrib in attribs:
	nbins = ds.get_attribute_num_of_unique(attrib)
	x = ds.get_attribute_values(dname=attrib).astype(float)
	x = np.asarray(np.nonzero(x))[0]
	mean = x.mean(axis=0)
	stdev = x.std(axis=0)
	y = np.random.normal(loc=mean, scale=stdev, size=1000000)
	fig = plt.figure(figsize=(10, 4))
	fig.subplots_adjust(top=0.90, bottom=0.20, left=0.12, right=0.96, wspace=0.35)
	plt.subplot(1, 2, 1)
	plt.hist(x, color='b', bins=nbins, histtype='step', density=True)
	plt.hist(y, color='r', bins=nbins, histtype='step', density=True)
	ax = plt.gca()
	ax.set_ylabel('density')
	ax.set_xlabel(attrib)
	plt.subplot(1, 2, 2)
	plt.hist(x, color='b', bins=nbins, label='data', histtype='step', density=True, cumulative=True)
	plt.hist(y, color='r', bins=nbins, label='normal', histtype='step', density=True, cumulative=True)
	ax = plt.gca()
	ax.set_ylabel('cumulative density')
	ax.set_xlabel(attrib)
	fig.suptitle('adult data set - distribution: {}'.format(attrib))
	fig.legend(loc='lower center', ncol=2)
	plt.savefig('figs/histogram_adult_{}_nonzeros.pdf'.format(attrib))
	plt.savefig('figs/histogram_adult_{}_nonzeros.png'.format(attrib))