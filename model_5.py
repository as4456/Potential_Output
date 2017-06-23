import numpy as np
import pandas as pd
import pymc as mc
import statsmodels.api as sm
import matplotlib.pyplot as plt
#import fredapi
import statsmodels.tsa.statespace.kalman_smoother as ks
from state_space_filters import  HP_filter_Model4_test, get_authors_data, get_fredapi_data, get_diff_from_cesano

np.set_printoptions(precision=4, suppress=True, linewidth=120)

file_path = './data/author_data.csv'

cutoff_date_min='1979-12-31'
cutoff_date_max = '2017-03-01'
cutoff_date_min_cesano = '1975-01-01'

# data_gdp_sa, d_data_gdp_sa, d_data_houses, _ = get_authors_data(file_path, cutoff_date_min=cutoff_date_min,
#                                                              cutoff_date_max = cutoff_date_max,
#                                                              cutoff_date_min_cesano = cutoff_date_min_cesano)

data_gdp_sa, d_data_gdp_sa, d_data_houses, _ = get_fredapi_data(cutoff_date_min=cutoff_date_min,
                                                                            cutoff_date_max=cutoff_date_max,
                                                                            cutoff_date_min_cesano=cutoff_date_min_cesano)


d_data_houses = d_data_houses - get_diff_from_cesano(d_data_houses)
d_data_houses = d_data_houses.shift(4)
d_data_houses = d_data_houses[(d_data_houses.index>cutoff_date_min) & (d_data_houses.index<cutoff_date_max)]



fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(d_data_gdp_sa.index, d_data_gdp_sa)
ax.set_ylabel('Delta log GDP')
ax.grid()
ax = fig.add_subplot(2,1,2)
ax.plot(d_data_houses.index, d_data_houses)
ax.set_ylabel('Delta log houses')
ax.grid()
fig.savefig('./output/macro_variables.pdf')

# Running the built-in HP filter
log_cycle, log_trend = sm.tsa.filters.hpfilter(data_gdp_sa)
trend = log_trend
cycle = data_gdp_sa - log_trend

y = pd.concat([data_gdp_sa, d_data_houses], axis=1)

lamda = 36
gap_var_estimate = cycle.diff().var()
true_gdp_var_estimate = gap_var_estimate / lamda
x_var_estimate = d_data_houses.diff().var()

init_state = np.r_[data_gdp_sa[0], data_gdp_sa[0], 0, d_data_houses[0], data_gdp_sa[0]]
init_state_cov = np.array([[true_gdp_var_estimate, 0, 0, 0, 0],
                           [0, true_gdp_var_estimate, 0, 0, 0],
                           [0, 0, gap_var_estimate, 0, 0],
                           [0,0,0,x_var_estimate,0],
                           [0,0,0,0,true_gdp_var_estimate]])

xxx = np.zeros((y.shape[0],2))
xxx[:, 0] = y.values[:, 0]
xxx[:, 1] = y.values[:, 1]

hpf_model = HP_filter_Model4_test(xxx, init_state, init_state_cov, init_params=[1.0, 0.5, 0.5], to_update_params=True)

sigma2_prior = mc.Gamma('sigma_exog', 1.0, 1.0)
beta_prior = mc.Gamma('beta', 0.7, 1.0)
gamma_prior = mc.Gamma('gamma', 0.7, 1.0)

# Create the stochastic (observed) component
@mc.stochastic(dtype=HP_filter_Model4_test, observed=True)
def local_level(value=hpf_model, q=sigma2_prior, b=beta_prior, g=gamma_prior):
    if (b<0 or b>0.95):
        return -10000.0
    if g<0:
        return -10000.0
    return value.loglike([q, b, g], transformed=True)

# Create the PyMC model
ll_mc = mc.Model((sigma2_prior, beta_prior, gamma_prior, local_level))

# Create a PyMC sample
ll_sampler = mc.MCMC(ll_mc)

# Sample
res = ll_sampler.sample(iter=10000, burn=1000, thin=10)

ll_sampler.summary()
# Plot traces
mc.Matplot.plot(ll_sampler)


beta_estimate = np.median(ll_sampler.trace('beta')[:])
sigma_estimate = np.median(ll_sampler.trace('sigma_exog')[:])
gamma_estimate = np.median(ll_sampler.trace('gamma')[:])




# Note that the variance parameter here is important!!! It needs to be set to the correct value
# since it is not being updated
number_of_observations = xxx.shape[0]
k_states = 5
transition = np.array([[0, 2., beta_estimate, gamma_estimate, -1.],
                       [0, 2.0, 0, 0, -1.0],
                       [0.0, 0.0, beta_estimate, gamma_estimate, 0.0],
                       [0.0, 0.0, 0.0, 1.0, 0.0],
                       [0.0, 1.0, .0, 0.0, 0.0]])
design = np.array([[1, 0, 0, 0, 0], [0.0,  0.0, 0.0, 1.0, 0.0]])
selection = np.array([[1.0, 1.0, gamma_estimate], [1.0,  0.0, 0.0], [0.0, 1.0, gamma_estimate],
                      [0, 0, 1.0], [0, 0, 0]])
obs_cov = np.array([[0.0, 0.0], [0.0, 0.0]])
state_cov = np.array([[true_gdp_var_estimate, 0, 0], [0, gap_var_estimate, 0], [0, 0, sigma_estimate]])

kf_smoother = ks.KalmanSmoother(2, k_states, k_posdef=3, transition=transition, design=design,
                                selection=selection, state_cov=state_cov, obs_cov=obs_cov,
                                initial_state=init_state, initial_state_cov=init_state_cov,
                                initialization='known', loglikelihood_burn=1, nobs=number_of_observations)

kf_smoother.bind(xxx)

kf_res = kf_smoother.smooth()

y_smoothed = pd.Series(np.reshape(kf_res.smoothed_state[1, :], -1), index=y.index)

fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.plot(data_gdp_sa.index, data_gdp_sa, label='Observed')
ax.plot(y_smoothed.index, y_smoothed, label='Smoothed')
ax.legend(loc='best')
ax.grid()

hp1 = cycle
hp2 = (data_gdp_sa - y_smoothed)

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(1,1,1)
ax.plot(cycle.index, hp1, label = 'HP filter')
ax.plot(cycle.index, hp2, label='MLE HP filter')
ax.legend(loc='best')
ax.grid()
ax.set_ylabel('Real GDP cycle as percentage of GDP (%)')

# Now that we have the estimates the init_cov_matrix should change
init_state_cov = np.array([[true_gdp_var_estimate, 0, 0, 0, 0],
                           [0, true_gdp_var_estimate, 0, 0, 0],
                           [0, 0, gap_var_estimate, 0, 0],
                           [0, 0, 0, sigma_estimate, 0],
                           [0, 0, 0, 0, true_gdp_var_estimate]])
hpf_model = HP_filter_Model4_test(xxx, init_state, init_state_cov,
                                  init_params=[sigma_estimate, beta_estimate, gamma_estimate], to_update_params=False)

np.alltrue(kf_smoother.design == hpf_model.ssm.design)
np.alltrue(kf_smoother.transition == hpf_model.ssm.transition)
np.alltrue(kf_smoother.selection == hpf_model.ssm.selection)

np.alltrue(kf_smoother.state_cov == hpf_model.ssm.state_cov)

hpf_res = hpf_model.fit()

y_smoothed_2 = pd.Series(np.reshape(hpf_res.smoothed_state[1, :], -1), index=data_gdp_sa.index)
hp3 = (data_gdp_sa - y_smoothed_2)

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(2,1,1)
ax.plot(cycle.index, hp1, label = 'HP filter')
ax.plot(cycle.index, hp2, label='KF filter')
ax.plot(cycle.index, hp3, label='MLE filter')
ax.legend(loc='best')
ax.grid()
ax.set_ylabel('Real GDP cycle as percentage of GDP (%)')
ax = fig.add_subplot(2,1,2)
ax.plot(data_gdp_sa.index, (hp2-hp3)/hp3*100)
ax.grid()
ax.set_ylabel('Difference between MLE and KS (%)')
fig.savefig(('./output/model_5.pdf'))