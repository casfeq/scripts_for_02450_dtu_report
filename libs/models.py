import numpy as np
from scipy.linalg import LinAlgWarning
from sklearn import linear_model
from sklearn import model_selection
from sklearn import naive_bayes 
from sklearn import neighbors
from sklearn import tree
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils._testing import ignore_warnings
import torch


class Model(object):
	def update_params(self, **kwargs):
		for key, value in kwargs.items():
			self._kwargs[key] = value
		self.set_params()

	@ignore_warnings(category=ConvergenceWarning)
	def fit(self, X, y):
		self._model.fit(X, y)
		return self._model

	def predict(self, X):
		return self._model.predict(X)

	def predict_proba(self, X):
		return self._model.predict_proba(X)

	def get_error(self, X_test, y_test):
		y_predicted = self.predict(X_test)
		error = np.abs(y_predicted - y_test)
		return np.sum(error, axis=0)/len(error)

	@ignore_warnings(category=LinAlgWarning)
	def get_error_k_fold(self, X, y, n_splits):
		kf = model_selection.KFold(n_splits=n_splits)
		gen_error = 0
		N = len(y)
		for train_indices, test_indices in kf.split(X):
			X_train = X[train_indices, :]
			y_train = y[train_indices]
			X_test = X[test_indices, :]
			y_test = y[test_indices]
			Nk = len(y_test)
			self.fit(X_train, y_train)
			error = self.get_error(X_test, y_test)
			gen_error += (Nk/N)*error
		return gen_error

	def get_error_k_fold_2_level(self, X, y, inner_splits, outer_splits, param_name, param_array):
		kfo = model_selection.KFold(n_splits=outer_splits)
		gen_error = 0
		test_errors = []
		best_params = []
		N = len(y)
		i = 0
		for train_indices_o, test_indices_o in kfo.split(X):
			X_par = X[train_indices_o, :]
			y_par = y[train_indices_o]
			X_test = X[test_indices_o, :]
			y_test = y[test_indices_o]
			Nk_test = len(y_test)
			min_error = 1e308
			best_param = None
			for param_value in param_array:
				kwargs = {param_name: param_value}
				self.update_params(**kwargs)
				error = self.get_error_k_fold(X_par, y_par, inner_splits)
				if min_error > error:
					best_param = param_value
					min_error = error
			best_params.append(best_param)
			kwargs = {param_name: best_param}
			self.update_params(**kwargs)
			self.fit(X_par, y_par)
			test_error = self.get_error(X_test, y_test)
			test_errors.append(test_error)
			gen_error += (Nk_test/N)*test_error
		return gen_error, np.array(best_params), np.array(test_errors)


class Ridge(Model):
	def __init__(self, max_iter=100, tol=1e-4, lamda=0.):
		self._kwargs = {'lamda': lamda, 'max_iter': max_iter, 'tol': tol}
		self.set_params()

	def set_params(self):
		self._kwargs['alpha'] = self._kwargs['lamda']
		self._kwargs.pop('lamda')
		self._model = linear_model.Ridge(**self._kwargs)

	def get_coefficients(self):
		return self._model.coef_


class ANN(Model):
	@ignore_warnings(category=UserWarning)
	def __init__(self, n_attributes, n_hidden_units=1, max_iter=10000, tol=1e-6, n_replicates=2):
		self._kwargs = {'n_attributes': n_attributes, 'n_hidden_units': n_hidden_units, 'max_iter': max_iter, 'tol': tol, 'n_replicates': n_replicates}
		if torch.cuda.is_available():
			self._dev = torch.device('cuda')
		else:
			self._dev = torch.device('cpu')
		self.set_params()

	@ignore_warnings(category=UserWarning)
	def fit(self, X, y):
		tol = self._kwargs['tol']
		max_iter = self._kwargs['max_iter']
		n_replicates = self._kwargs['n_replicates']
		X = torch.from_numpy(X).float().to(self._dev)
		y = torch.from_numpy(y.reshape(len(y), 1)).float().to(self._dev)
		best_final_loss = 1e100
		for r in range(n_replicates):
			net = self._model().to(self._dev)
			torch.nn.init.xavier_uniform_(net[0].weight).to(self._dev)
			torch.nn.init.xavier_uniform_(net[2].weight).to(self._dev)
			optimizer = torch.optim.Adam(net.parameters())
			learning_curve = []
			old_loss = 1e6
			for i in range(max_iter):
				y_est = net(X)
				loss = self._loss_fn(y_est, y)
				loss_value = loss.data.numpy()
				learning_curve.append(loss_value)
				p_delta_loss = np.abs(loss_value - old_loss)/old_loss
				if p_delta_loss < tol:
					break 
				old_loss = loss_value
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()
			if loss_value < best_final_loss: 
				best_net = net
				best_final_loss = loss_value
				best_learning_curve = learning_curve
		self._network = best_net
		return best_final_loss, best_learning_curve


class BaselineRegression(Model):
	def __init__(self):
		self._kwargs = {}

	def set_params(self, **kwargs):
		for key, value in kwargs.items():
			self._kwargs[key] = value

	def fit(self, X, y):
		self._output = np.mean(y)

	def predict(self, X):
		output = [self._output]*len(X)
		return np.array(output)


class LinearRegression(Ridge):
	def predict(self, X):
		y = self._model.predict(X)
		return y.reshape(len(y))

		
class LogisticRegression(Model):
	def __init__(self, max_iter=100, tol=1e-4, lamda=0.):
		self._kwargs = {'max_iter': max_iter, 'tol': tol, 'lamda': lamda}
		self.set_params()

	def set_params(self):
		self._kwargs['C'] = 1./max(self._kwargs['lamda'], 1e-308)
		self._kwargs.pop('lamda')
		self._model = linear_model.LogisticRegression(**self._kwargs)


class ANNRegression(ANN):
	@ignore_warnings(category=UserWarning)
	def set_params(self):
		self._loss_fn = torch.nn.MSELoss().to(self._dev)
		self._model = lambda: torch.nn.Sequential(torch.nn.Linear(self._kwargs['n_attributes'], self._kwargs['n_hidden_units']), torch.nn.Tanh(), torch.nn.Linear(self._kwargs['n_hidden_units'], 1))

	@ignore_warnings(category=UserWarning)
	def predict(self, X):
		y = self._network(torch.from_numpy(X).float()).data.numpy()
		return y.reshape(len(y))


class BaselineClassifier(Model):
	def __init__(self):
		self._kwargs = {}

	def set_params(self, **kwargs):
		for key, value in kwargs.items():
			self._kwargs[key] = value

	def fit(self, X, y):
		nonzeros = np.count_nonzero(y)
		zeros = len(y) - nonzeros
		self._output = 1 if nonzeros > zeros else 0

	def predict(self, X):
		output = [self._output]*len(X)
		return np.array(output)


class LinearClassifier(Ridge):
	def predict(self, X):
		y = self._model.predict(X)
		y = (y > .5).astype(int)
		return y.reshape(len(y))


class DecisionTreeClassifier(Model):
	def __init__(self, criterion='gini', min_samples_split=2, max_depth=None):
		self._kwargs = {'criterion': criterion, 'min_samples_split': min_samples_split, 'max_depth': max_depth}
		self.set_params()

	def set_params(self):
		self._model = tree.DecisionTreeClassifier(**self._kwargs)


class KNeighborsClassifier(Model):
	def __init__(self, n_neighbors, p=2, metric='minkowski', metric_params={}):
		self._kwargs = {'n_neighbors': n_neighbors, 'p': p, 'metric': metric, 'metric_params': metric_params}
		self.set_params()

	def set_params(self):
		self._model = neighbors.KNeighborsClassifier(**self._kwargs)


class NBClassifier(Model):
	def __init__(self, alpha=1., fit_prior=True):
		self._kwargs = {'alpha': alpha, 'fit_prior': fit_prior}
		self.set_params()

	def set_params(self):
		self._model = naive_bayes.MultinomialNB(**self._kwargs)


class ANNClassifier(ANN):
	@ignore_warnings(category=UserWarning)
	def set_params(self):
		self._loss_fn = torch.nn.BCELoss().to(self._dev)
		self._model = lambda: torch.nn.Sequential(torch.nn.Linear(self._kwargs['n_attributes'], self._kwargs['n_hidden_units']), torch.nn.Tanh(), torch.nn.Linear(self._kwargs['n_hidden_units'], 1), torch.nn.Sigmoid())

	@ignore_warnings(category=UserWarning)
	def predict(self, X):
		y = self._network(torch.from_numpy(X).float()).data.numpy()
		y = (y > .5).astype(int)
		return y.reshape(len(y))