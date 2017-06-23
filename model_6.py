import numpy as np
import pandas as pd
import pymc as mc
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.statespace.kalman_smoother as ks
from state_space_filters import  HP_filter_Model6, get_authors_data, get_fredapi_data, get_diff_from_cesano, _data_transformations

file_path = './data/author_data.csv'

# These dates are used to define the intervals to be used in the simulation
cutoff_date_min='1979-12-31'
cutoff_date_max = '2017-03-01'
cutoff_date_min_cesano = '1975-01-01'

# Data can be read either from the author's data files or straight from FRED API database
# data_gdp_sa, d_data_gdp_sa, d_data_houses, d_data_credit = get_authors_data(file_path, cutoff_date_min=cutoff_date_min,
#                                                              cutoff_date_max = cutoff_date_max,
#                                                              cutoff_date_min_cesano = cutoff_date_min_cesano)

import fredapi
api_key = 'e47df9c3446af26f9c52fc03015ee6ec'
fred = fredapi.Fred(api_key=api_key)
diction = {'CPI':'CPIAUCSL', 'RPP':'QUSN628BIS', 'CR':'CRDQUSAPABIS',
                                   'PGDP':'GDPDEF', 'GDP':'GDP'}

cpi = fred.get_series(diction['CPI'])
print cpi
RPP = fred.get_series(diction['RPP'])
CR = fred.get_series(diction['CR'])
pgdp = fred.get_series(diction['PGDP'])
gdp = fred.get_series(diction['GDP'])
# cpi = fred.get_series('CPIAUCSL')
# RPP = fred.get_series('QUSN628BIS')
# CR = fred.get_series('CRDQUSAPABIS')
# pgdp = fred.get_series('GDPDEF')
# gdp = fred.get_series('GDP')

# Resampling and making sure we get the correct dates as indices
cpi2 = cpi.resample('Q', label='left',how='mean')
print cpi2
cpi2 = pd.Series(cpi2.values, index=cpi2.index + pd.DateOffset(1))
data_gdp_sa = 100 * np.log(gdp / pgdp)
data_houses = 100 * np.log(RPP / cpi2)
data_credit = 100 * np.log(CR / pgdp)
data_gdp_sa, d_data_gdp_sa, d_data_houses, d_data_credit = _data_transformations(data_gdp_sa, data_houses, data_credit, cutoff_date_min_cesano,
                          cutoff_date_min, cutoff_date_max)
#
##data_gdp_sa, d_data_gdp_sa, d_data_houses, d_data_credit = get_fredapi_data(cutoff_date_min=cutoff_date_min,
##                                                             cutoff_date_max = cutoff_date_max,
##                                                             cutoff_date_min_cesano = cutoff_date_min_cesano)
#
#
# Subtract the Cesano means from the timeseries and keep only the appropriate intervals
d_data_credit = d_data_credit - get_diff_from_cesano(d_data_credit)
d_data_credit = d_data_credit[(d_data_credit.index>cutoff_date_min) & (d_data_credit.index<cutoff_date_max)]

d_data_houses = d_data_houses - get_diff_from_cesano(d_data_houses)
# Get the initial variance estimate for the Kalman Smoother
z_var_estimate = d_data_houses[(d_data_houses.index>cutoff_date_min) & (d_data_houses.index<cutoff_date_max)].diff().var()
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

y = pd.concat([data_gdp_sa, d_data_credit, d_data_houses], axis=1)

# Define the simulation initial parameters according to the paper
lamda = 19.8
gap_var_estimate = cycle.diff().var()
true_gdp_var_estimate = gap_var_estimate / lamda
x_var_estimate = d_data_credit.diff().var()

# Initial state and state covariance estimates
init_state = np.r_[data_gdp_sa[0], data_gdp_sa[0], 0, d_data_credit[0], d_data_houses[0], data_gdp_sa[0],
                   d_data_houses[0], d_data_houses[0], d_data_houses[0]]
init_state_cov = np.array([[true_gdp_var_estimate, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, true_gdp_var_estimate, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, gap_var_estimate, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, x_var_estimate, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, z_var_estimate, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, true_gdp_var_estimate, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, z_var_estimate, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, z_var_estimate, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, z_var_estimate]])

# THis is needed to make sure the inputs have the right numpy dimensions
xxx = np.zeros((y.shape[0],3))
xxx[:, 0] = y.values[:, 0]
xxx[:, 1] = y.values[:, 1]
xxx[:, 2] = y.values[:, 2]

# Define the state space model for model 6 of the paper
hpf_model = HP_filter_Model6(xxx, init_state, init_state_cov,
                             init_params=[1.0, 0.5, 0.5, 0.5, 1.0], to_update_params=True)

# Define the prior distributions for all the parameters to be used in the MCMC parameters estimation
sigma1_prior = mc.Gamma('sigma1_exog', 1.0, 1.0)
sigma2_prior = mc.Gamma('sigma2_exog', 1.0, 1.0)
beta_prior = mc.Gamma('beta', 0.7, 1.0)
gamma1_prior = mc.Gamma('gamma1', 0.7, 1.0)
gamma2_prior = mc.Gamma('gamma2', 0.7, 1.0)

# Degfine the boundaries to be used for rejecting paths based on the likelihood
VERY_LOW = -1000000.0
# Create the stochastic (observed) component
@mc.stochastic(dtype=HP_filter_Model6, observed=True)
def local_level(value=hpf_model, q=sigma1_prior, b=beta_prior, g1=gamma1_prior, g2=gamma2_prior, q2=sigma2_prior):
    if (b<0 or b>0.95):
        return VERY_LOW
    if g1<0:
        return VERY_LOW
    if g2<0:
        return VERY_LOW
    return value.loglike([q, b, g1, g2, q2], transformed=True)

# Create the PyMC model
ll_mc = mc.Model((sigma2_prior, beta_prior, gamma1_prior, gamma2_prior, local_level))

# Create a PyMC sample
ll_sampler = mc.MCMC(ll_mc)

# Sample
res = ll_sampler.sample(iter=10000, burn=1000, thin=10)

# Plot traces
ll_sampler.summary()
mc.Matplot.plot(ll_sampler)

# The final state space parameters can be taken either from the MCMC simulation or to use the ones from the paper.
# Using the ones from the Matlab implementation of the code for now to compare results.
beta_estimate = np.median(ll_sampler.trace('beta')[:])
sigma1_estimate = np.median(ll_sampler.trace('sigma1_exog')[:])
sigma2_estimate = np.median(ll_sampler.trace('sigma2_exog')[:])
gamma1_estimate = np.median(ll_sampler.trace('gamma1')[:])
gamma2_estimate = np.median(ll_sampler.trace('gamma2')[:])
beta_estimate = 0.8175
# sigma1_estimate = 0.648
# sigma2_estimate = 2.782
gamma1_estimate = 0.4665
gamma2_estimate = 0.088



# Note that the variance parameter here is important!!! It needs to be set to the correct value
# since it is not being updated. Setting up the inputs for the direct Kalman smoother approach
number_of_observations = xxx.shape[0]
k_states = 9
transition = np.array([[0.0, 2., beta_estimate, gamma1_estimate, 0.0, -1., 0.0, 0.0, gamma2_estimate],
                       [0, 2.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, beta_estimate, gamma1_estimate, 0.0, 0.0, 0.0, 0.0, gamma2_estimate],
                       [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                       [0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0],
                       [0, 0, 0, 0, 0, 0, 1, 0, 0],
                       [0, 0, 0, 0, 0, 0, 0, 1, 0]])
design = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0, 0, 0],
                   [0, 0, 0, 0, 1.0, 0.0, 0.0, 0.0, 0.0]])

selection = np.array([[1.0, 1.0, gamma1_estimate, 0.0], [1.0,  0.0, 0.0, 0.0],
                      [0.0, 1.0, gamma1_estimate, 0.0],
                      [0, 0, 1.0, 0.0], [0, 0, 0, 1.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0],
                      [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
obs_cov = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0, 0, 0]])
state_cov = np.array([[true_gdp_var_estimate, 0, 0, 0], [0, gap_var_estimate, 0, 0], [0, 0, sigma1_estimate, 0.0],
                      [0.0, 0.0, 0.0, sigma2_estimate]])

kf_smoother = ks.KalmanSmoother(3, k_states, k_posdef=4, transition=transition, design=design,
                                selection=selection, state_cov=state_cov, obs_cov=obs_cov,
                                initial_state=init_state, initial_state_cov=init_state_cov,
                                initialization='known', loglikelihood_burn=1, nobs=number_of_observations)

# Bind the smoother object to the data
kf_smoother.bind(xxx)

# Run the smoother and get the desired smoothed states
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


# Setting the initial state covatiance again but now use the outputs of the MCMC simulation for the variances
init_state_cov = np.array([[true_gdp_var_estimate, 0, 0, 0, 0, 0, 0, 0, 0],
                           [0, true_gdp_var_estimate, 0, 0, 0, 0, 0, 0, 0],
                           [0, 0, gap_var_estimate, 0, 0, 0, 0, 0, 0],
                           [0, 0, 0, sigma1_estimate, 0, 0, 0, 0, 0],
                           [0, 0, 0, 0, sigma2_estimate, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0, true_gdp_var_estimate, 0, 0, 0],
                           [0, 0, 0, 0, 0, 0, sigma2_estimate, 0, 0],
                           [0, 0, 0, 0, 0, 0, 0, sigma2_estimate, 0],
                           [0, 0, 0, 0, 0, 0, 0, 0, sigma2_estimate]])

# Computing again using the built-in wrapper object to make sure everything matches...
hpf_model = HP_filter_Model6(xxx, init_state, init_state_cov,
                             init_params=[sigma1_estimate, beta_estimate, gamma1_estimate, gamma2_estimate,
                                          sigma2_estimate], to_update_params=False)

# Sanity checks
np.alltrue(kf_smoother.design == hpf_model.ssm.design)
np.alltrue(kf_smoother.transition == hpf_model.ssm.transition)
np.alltrue(kf_smoother.selection == hpf_model.ssm.selection)
np.alltrue(kf_smoother.state_cov == hpf_model.ssm.state_cov)

hpf_res = hpf_model.fit()

y_smoothed_2 = pd.Series(np.reshape(hpf_res.smoothed_state[1, :], -1), index=data_gdp_sa.index)
hp3 = (data_gdp_sa - y_smoothed_2)

# There should be a small difference in the beginning because of the different initialisation
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
fig.savefig(('./output/model_6.pdf'))