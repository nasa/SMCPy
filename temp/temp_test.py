import matplotlib.pyplot as plt
import numpy as np
import pymc
from scipy.stats import multivariate_normal as mv

n_params = 2
n_samples = 10000
centers = np.array([2, 1.0])
scales = np.array([1, 0.5])

var = (centers*scales)**2
cov = np.eye(n_params)*var
scipy_rv = mv(centers, cov)
pymc_rv = [pymc.Normal('rv%s' % i, centers[i], 1/var[i]) \
           for i in range(n_params)]

scipy_samples = scipy_rv.rvs(n_samples)
pymc_samples = np.array([[pymc_rv[i].random() for i in range(n_params)] \
                          for _ in range(n_samples)])

print scipy_rv.logpdf(pymc_samples[-1])
print np.sum([pymc_rv[i].logp for i in range(n_params)])


fig = plt.figure()
ax0 = fig.add_subplot(211)
ax0.scatter(pymc_samples[:, 0], pymc_samples[:, 1])
ax0.set_xlim(-10, 10)
ax0.set_ylim(-1, 3)

ax1 = fig.add_subplot(212)
ax1.scatter(scipy_samples[:, 0], scipy_samples[:, 1])
ax1.set_xlim(-10, 10)
ax1.set_ylim(-1, 3)

plt.show()
