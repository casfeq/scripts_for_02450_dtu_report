import sys; sys.path.insert(0, '../')
import numpy as np
from libs.dataset import DataSet


filepath_1 = '../data/adult.data'
filepath_2 = '../data/adult.test'
dnames = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'annual-income']
dformats = [int, str, int, str, int, str, str, str, str, str, int, int, int, str, str]
ds = DataSet()
print('>> Loading first part of the data set.')
ds.load_data(filepath_1, dtype=np.ndarray, delimiter=', ')
ds.set_attributes_name(dnames)
ds.set_attributes_format(dformats)
print('Data set size:', ds.get_num_of_cases(), 'No of attributes:', ds.get_num_of_attributes())
print('>> Appending second part of the data set.')
ds.append_data(filepath_2, dtype=np.ndarray, delimiter=', ')
ds.set_attributes_format(dformats)
print('Data set size:', ds.get_num_of_cases(), 'No of attributes:', ds.get_num_of_attributes())
print('>> Removing missing values from data set.')
ds.remove_missing_values()
print('Data set size:', ds.get_num_of_cases(), 'No of attributes:', ds.get_num_of_attributes())
print('>> Binarizing attributes.')
ds.bool_attribute(dname='annual-income', true_ref='>50K')
ds.bool_attribute(dname='sex', true_ref='Female')
print('Data set size:', ds.get_num_of_cases(), 'No of attributes:', ds.get_num_of_attributes())
print('>> Encoding as one-out-of-K.')
ds.encode_as_1_out_of_K()
print('Data set size:', ds.get_num_of_cases(), 'No of attributes:', ds.get_num_of_attributes())
print('>> Normalizing.')
ds.normalize_all_attributes()
print('Data set size:', ds.get_num_of_cases(), 'No of attributes:', ds.get_num_of_attributes())