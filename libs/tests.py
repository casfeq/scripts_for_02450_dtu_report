import numpy as np
import scipy as sp
from sklearn.utils._testing import ignore_warnings


class Test(object):
	def __init__(self, alpha, model_a, model_b):
		self.set_confidence_level(alpha)
		self.set_models(model_a, model_b)

	def set_confidence_level(self, alpha):
		self.alpha = alpha

	def set_models(self, model_a, model_b):
		self.model_a = model_a
		self.model_b = model_b

	def set_test_data(self, X_test, y_test):
		self.X_test = X_test
		self.y_test = y_test


class TStudent(Test):
	@ignore_warnings(category=RuntimeWarning)
	def perform_test(self):
		y_predicted_a = self.model_a.predict(self.X_test)
		y_predicted_b = self.model_b.predict(self.X_test)
		z_a = np.abs(self.y_test - y_predicted_a)**2
		z_b = np.abs(self.y_test - y_predicted_b)**2
		z = z_a - z_b
		confidence_interval = sp.stats.t.interval(1 - self.alpha, len(z) - 1, loc=np.mean(z), scale=sp.stats.sem(z))
		p = sp.stats.t.cdf(-np.abs(np.mean(z))/sp.stats.sem(z), df=len(z) - 1)
		return confidence_interval, p
		

class McNemar(Test):
	@ignore_warnings(category=RuntimeWarning)
	def perform_test(self):
		nn = np.zeros((2, 2))
		y_predicted_a = self.model_a.predict(self.X_test)
		y_predicted_b = self.model_b.predict(self.X_test)
		condition_1 = (y_predicted_a - self.y_test == 0)
		condition_2 = (y_predicted_b - self.y_test == 0)
		nn[0, 0] = sum(condition_1 & condition_2)
		nn[0, 1] = sum(condition_1 & ~condition_2)
		nn[1, 0] = sum(~condition_1 & condition_2)
		nn[1, 1] = sum(~condition_1 & ~condition_2)
		n = sum(nn.flat)
		n12 = nn[0, 1]
		n21 = nn[1, 0]
		theta = (n12 - n21)/n
		Q = n**2*(n + 1)*(theta + 1)*(1 - theta)/((n*(n12 + n21) - (n12 - n21)**2))
		p = (theta + 1)*0.5*(Q - 1)
		q = (1 - theta)*0.5*(Q - 1)
		confidence_interval = tuple(2*x - 1 for x in sp.stats.beta.interval(1 - self.alpha, a=p, b=q))
		p = 2*sp.stats.binom.cdf(min([n12, n21]), n=n12+n21, p=0.5)
		return theta, confidence_interval, p