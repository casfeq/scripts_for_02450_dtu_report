import numpy as np
from scipy import linalg
from sklearn import model_selection


def categoric2numeric(x):
	x = np.asarray(x).ravel()
	x_labels = np.unique(x)
	x_labels_str = x_labels.astype(str).tolist()
	N = len(x)
	M = len(x_labels)
	xc = np.zeros((N, M), dtype=int)
	for i in range(M):
		flags = (x == x_labels[i])
		xc[flags,i] = 1
	return xc, x_labels_str


class DataSet(object):
	def load_data(self, filepath, dtype=np.ndarray, delimiter=', '):
		self._data = np.loadtxt(fname=filepath, dtype=dtype, delimiter=delimiter)
		self._data_dict = {}

	def append_data(self, filepath, dtype=np.ndarray, delimiter=', '):
		self._data = np.append(self._data, np.loadtxt(fname=filepath, dtype=dtype, delimiter=delimiter), axis=0)

	def set_attributes_name(self, dnames):
		self._dnames = np.asarray(dnames)
		self._odnames = self._dnames

	def set_attributes_format(self, dformats):
		self._dformats = np.asarray(dformats)
		for i,dformat in enumerate(self._dformats):
			self._data[:, i] = self._data[:, i].astype(dformat)
		self._odata = self._data
		self._odformats = self._dformats

	def split(self, test_size=None, random_state=None):
		self._data, self._test_set= model_selection.train_test_split(self._data, test_size=test_size, random_state=random_state)

	def merge_back(self):
		self._data = np.append(self._data, self._test_set, axis=0)

	def format_attribute(self, dname, dformat):
		index = np.where(self._dnames == dname)[0][0]
		self._data[:, index] = self._data[:, index].astype(dformat)

	def bool_attribute(self, dname, true_ref):
		index = np.where(self._dnames == dname)[0][0]
		self._data[:, index] = (self._data[:, index] == true_ref).astype(int)
		self._dformats[index] = int

	def threshold_attribute(self, dname, break_pnt, closed=True):
		index = np.where(self._dnames == dname)[0][0]
		self._data[:, index] = (self._data[:, index] >= break_pnt).astype(int) if closed else (self._data[:, index] > break_pnt).astype(int)
		self._dformats[index] = int

	def normalize_attribute(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		attrib = self._data[:, index]
		N = len(attrib)
		mean = attrib.mean(axis=0)
		stdev = attrib.std(axis=0)
		self._data[:, index] = (attrib - np.ones((N))*mean)/stdev
		self._dformats[index] = float

	def normalize_all_attributes(self):
		for dname in self._dnames:
			self.normalize_attribute(dname)
		self._data = np.vstack(self._data[:, :]).astype(float)

	def format_all_attributes(self, dformat):
		self._data = np.vstack(self._data[:, :]).astype(dformat)

	def remove_missing_values(self):
		self._data = self._data[np.all(self._data != '?', axis=1)]

	def encode_as_1_out_of_K(self):
		data_1K = list()
		format_1K = list()
		names_1K = list()
		for i,dformat in enumerate(self._dformats):
			if dformat in (int, float):
				data_1K.append(self._data[:, i])
				names_1K.append(self._dnames[i])
				format_1K.append(dformat)
			else:
				data_num, data_names = categoric2numeric(self._data[:, i])
				for i,name in enumerate(data_names):
					data_1K.append(data_num[:, i])
					names_1K.append(name)
					format_1K.append(int)
		self._data = np.asarray(data_1K).T
		self._dnames = np.asarray(names_1K)
		self._dformats = np.asarray(format_1K)

	def numify_attribute(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		if self._dformats[index] == str:
			self._data_dict[dname] = {}
			for i,value in enumerate(self.get_attribute_unique_values(dname)):
				self._data_dict[dname][value] = i
				self._data[:, index] = np.where(self._data[:, index] == value, i, self._data[:, index])
			self._dformats[index] = int

	def get_data(self):
		return self._data

	def get_test_set(self):
		return self._test_set

	def get_data_dict(self):
		return self._data_dict

	def get_names(self):
		return self._dnames

	def get_names_but_exception(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		dfilters = np.delete(self._dnames, index, axis=0)
		indices = np.in1d(self._dnames, np.asarray(dfilters))
		return self._dnames[indices]

	def get_formats(self):
		return self._dformats

	def get_num_of_cases(self):
		return len(self._data)

	def get_num_of_test_cases(self):
		return len(self._test_set)

	def get_num_of_attributes(self):
		return len(self._dnames)

	def get_attribute_values(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		return self._data[:, index].astype(self._dformats[index])

	def get_test_attribute_values(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		return self._test_set[:, index].astype(self._dformats[index])

	def sort_data_based_on_attribute(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		self._data = self._data[self._data[:, index].argsort()]

	def get_attribute_filtered_values(self, dname, dfilter):
		index = np.where(self._dnames == dname)[0][0]
		indices = np.where(self._data == dfilter)
		return self._data[indices, index][0]

	def filter_data(self, dfilters):
		indices = np.in1d(self._dnames, np.asarray(dfilters))
		self._data = self._data[:, indices]
		self._dformats = self._dformats[indices]
		self._dnames = self._dnames[indices]

	def get_attribute_mean(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		return self._data[:, index].mean(axis=0)

	def get_attribute_stdev(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		return self._data[:, index].std(axis=0)

	def get_attribute_unique_values(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		return np.unique(self._data[:, index])

	def get_attribute_num_of_unique(self, dname):
		return len(self.get_attribute_unique_values(dname))

	def get_data_but_exception(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		dfilters = np.delete(self._dnames, index, axis=0)
		indices = np.in1d(self._dnames, np.asarray(dfilters))
		return self._data[:, indices]

	def get_test_set_but_exception(self, dname):
		index = np.where(self._dnames == dname)[0][0]
		dfilters = np.delete(self._dnames, index, axis=0)
		indices = np.in1d(self._dnames, np.asarray(dfilters))
		return self._test_set[:, indices]

	def save_data(self, fname):
		np.savetxt('{}.csv'.format(fname), self._dnames.reshape(1, self._dnames.shape[0]), delimiter=';', fmt='%s')
		with open('{}.csv'.format(fname), 'ab') as f:
			np.savetxt(f, self._data, delimiter=';', fmt='%s')
			f.close()

	def compute_principal_components(self):
		self._U, self._S, self._V = linalg.svd(self._data, full_matrices=False)

	def compute_variance_by_pca(self):
		self._rho = (self._S*self._S)/(self._S*self._S).sum()

	def get_variance(self):
		return self._rho

	def get_principal_components(self):
		return self._V.T