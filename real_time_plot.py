import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import statsmodels.tsa.statespace.kalman_smoother as ks
from state_space_filters import get_authors_data, get_fredapi_data, get_diff_from_cesano

def get_model_6_output(y, data_gdp_sa, d_data_houses, d_data_credit):
    '''
    This function calculates the output gap timeseries based on model 6. The parameters are the ones defined in the
    paper
    '''

    xxx = np.zeros((y.shape[0], 3))
    xxx[:, 0] = y.values[:, 0]
    xxx[:, 1] = y.values[:, 1]
    xxx[:, 2] = y.values[:, 2]

    lamda = 19.8
    beta_estimate = 0.8175
    sigma1_estimate = 0.28915729735374207
    sigma2_estimate = 5.1
    gamma1_estimate = 0.4665
    gamma2_estimate = 0.088

    gap_var_estimate = 0.4766
    true_gdp_var_estimate = gap_var_estimate / lamda
    x_var_estimate = 0.2850
    z_var_estimate = 5.3954

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

    selection = np.array([[1.0, 1.0, gamma1_estimate, 0.0], [1.0, 0.0, 0.0, 0.0],
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

    kf_smoother.bind(xxx)

    kf_res = kf_smoother.smooth()

    y_smoothed = pd.Series(np.reshape(kf_res.smoothed_state[1, :], -1), index=y.index)

    return y.ix[:, 0] - y_smoothed



file_path = './data/author_data.csv'

cutoff_date_min='1979-12-31'
cutoff_date_max = '2017-03-01'
cutoff_date_min_cesano = '1975-01-01'

# data_gdp_sa, d_data_gdp_sa, d_data_houses, d_data_credit = get_authors_data(file_path, cutoff_date_min=cutoff_date_min,
#                                                              cutoff_date_max = cutoff_date_max,
#                                                              cutoff_date_min_cesano = cutoff_date_min_cesano)

data_gdp_sa, d_data_gdp_sa, d_data_houses, d_data_credit = get_fredapi_data(cutoff_date_min=cutoff_date_min,
                                                                            cutoff_date_max=cutoff_date_max,
                                                                            cutoff_date_min_cesano=cutoff_date_min_cesano)

d_data_credit = d_data_credit - get_diff_from_cesano(d_data_credit)
d_data_credit = d_data_credit[(d_data_credit.index>cutoff_date_min) & (d_data_credit.index<cutoff_date_max)]

d_data_houses = d_data_houses - get_diff_from_cesano(d_data_houses)
z_var_estimate = d_data_houses[(d_data_houses.index>cutoff_date_min) & (d_data_houses.index<cutoff_date_max)].diff().var()

d_data_houses = d_data_houses[(d_data_houses.index>cutoff_date_min) & (d_data_houses.index<cutoff_date_max)]


#Running the built-in HP filter
_, entire_trend = sm.tsa.filters.hpfilter(data_gdp_sa)
entire_cycle = data_gdp_sa - entire_trend


y = pd.concat([data_gdp_sa, d_data_credit, d_data_houses], axis=1)

# Get model 6 entire output
entire_model6_hp = get_model_6_output(y, data_gdp_sa, d_data_houses, d_data_credit)

fig = plt.figure(figsize=(9,9))
ax = fig.add_subplot(1,1,1)
ax.plot(entire_cycle.index, entire_cycle, label = 'HP filter')
ax.plot(entire_cycle.index, entire_model6_hp, label='MLE HP filter')
ax.legend(loc='best')
ax.grid()
ax.set_ylabel('Real GDP cycle as percentage of GDP (%)')

# Start date for the real-time output comparison
start_date = '2000-01-01'

all_dates = pd.date_range(start=start_date, end=cutoff_date_max, freq='Q').tolist()
real_time_hp_res = pd.Series(np.nan, index = all_dates)
real_time_model6_res = pd.Series(np.nan, index = all_dates)

# Get the output for all end-dates
for d in all_dates:

    yyy = y.ix[:d, :]


    log_cycle, log_trend = sm.tsa.filters.hpfilter(yyy[0])
    cycle = yyy[0] - log_trend

    real_time_hp_res.ix[d] = cycle[-1]

    t = get_model_6_output(yyy, data_gdp_sa, d_data_houses, d_data_credit)
    real_time_model6_res.ix[d] = t[-1]

# Plot the output
fig = plt.figure()
ax = fig.add_subplot(2,1,1)
ax.plot(entire_cycle.ix[start_date:].index, entire_cycle.ix[start_date:], label='Ex-post')
ax.plot(real_time_hp_res.ix[start_date:].index, real_time_hp_res.ix[start_date:], label='Real Time')
ax.legend(loc='best')
ax.grid()
ax = fig.add_subplot(2,1,2)
ax.plot(entire_model6_hp.ix[start_date:].index, entire_model6_hp.ix[start_date:], label='Ex-post')
ax.plot(real_time_model6_res.ix[start_date:].index, real_time_model6_res.ix[start_date:], label='Real Time')
ax.legend(loc='best')
ax.grid()
fig.savefig('./output/real_time.pdf')