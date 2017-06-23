import statsmodels.api as sm
import statsmodels.tsa.statespace.mlemodel as mle
import numpy as np
from statsmodels.tsa.statespace.tools import (constrain_stationary_univariate,
                                              unconstrain_stationary_univariate)
import fredapi
import pandas as pd

def get_diff_from_cesano(x):
    '''
    Calculates the time-series of Cesano means
    '''

    c_means = x.cumsum()
    nos = pd.Series(np.arange(len(c_means)) + 1, index=c_means.index)
    c_means2 = c_means / nos

    return c_means2.mean()

def _data_transformations(data_gdp_sa, data_houses, data_credit, cutoff_date_min_cesano,
                          cutoff_date_min, cutoff_date_max):

    d_data_gdp_sa = data_gdp_sa.diff(1)
    d_data_gdp_sa = d_data_gdp_sa[(d_data_gdp_sa.index > cutoff_date_min) & (d_data_gdp_sa.index < cutoff_date_max)]
    data_gdp_sa = data_gdp_sa[(data_gdp_sa.index > cutoff_date_min) & (data_gdp_sa.index < cutoff_date_max)]


    d_data_houses = data_houses.diff(1)
    d_data_houses = d_data_houses[
        (d_data_houses.index > cutoff_date_min_cesano) & (d_data_houses.index < cutoff_date_max)]

    d_data_credit = data_credit.diff(1)
    d_data_credit = d_data_credit[
    (d_data_credit.index > cutoff_date_min_cesano) & (d_data_credit.index < cutoff_date_max)]

    return data_gdp_sa, d_data_gdp_sa, d_data_houses, d_data_credit

def get_fredapi_data(diction = {'CPI':'CPIAUCSL', 'RPP':'QUSN628BIS', 'CR':'CRDQUSAPABIS',
                                   'PGDP':'GDPDEF', 'GDP':'GDP'}, api_key = 'your_fred_key',
                     cutoff_date_min='1979-12-31', cutoff_date_max = '2017-03-01',
                     cutoff_date_min_cesano = '1975-01-01'):
    '''
    This function is used to get the appropriate data from the FRED API repository
    '''

    fred = fredapi.Fred(api_key=api_key)

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
    #cpi2 = pd.Series(cpi2.values.ravel(), index=cpi2.index + pd.DateOffset(1))
    # Paper transformations
    data_gdp_sa = 100 * np.log(gdp / pgdp)
    data_houses = 100 * np.log(RPP / cpi2)
    data_credit = 100 * np.log(CR / pgdp)

    return _data_transformations(data_gdp_sa, data_houses, data_credit, cutoff_date_min_cesano,
                          cutoff_date_min, cutoff_date_max)



def get_authors_data(file_path, cutoff_date_min='1979-12-31', cutoff_date_max = '2013-01-01',
                     cutoff_date_min_cesano = '1975-01-01'):

    '''
    Reads and returns appropriate data from the authors data file.
    '''

    df = pd.read_csv('./data/author_data.csv')
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')

    # Paper transformations
    data_gdp_sa = 100 * np.log(df['GDP'] / df['PGDP'])
    data_houses = 100 * np.log(df['RPP'] / df['CPI'])
    data_credit = 100 * np.log(df['CR'] / df['PGDP'])

    return _data_transformations(data_gdp_sa, data_houses, data_credit, cutoff_date_min_cesano,
                                 cutoff_date_min, cutoff_date_max)

class HP_filter_Model6(mle.MLEModel):

    start_params = [0.0, 0.0, 0.0, 0.0, 0.0]
    param_names = ['sigma1_exog', 'beta', 'gamma1', 'gamma2', 'sigma2_exog']

    def __init__(self, endog, init_state, init_state_cov, init_params = [1.0, 0.5, 0.5, 0.5, 1.0], to_update_params=True):

        # Initialize the state space model
        super(HP_filter_Model6, self).__init__(endog, k_states=9, k_posdef=4,  loglikelihood_burn = 1,
                                                    initialization='known', initial_state_cov  = init_state_cov,
                                                    initial_state = init_state)

        self.to_update_params = to_update_params
        self.start_params = init_params
        self.nobs = len(endog)
        self.counter = 0


        # Initialize known components of the state space matrices
        self['design', 0, 0] = 1
        self['design', 1, 3] = 1
        self['design', 2, 4] = 1


        self['transition', 0, 1] = 2.
        self['transition', 0, 5] = -1.
        self['transition', 0, 3] = self.start_params[2]
        self['transition', 0, 2] = self.start_params[1]
        self['transition', 0, 8] = self.start_params[3]

        self['transition', 1, 1] = 2.
        self['transition', 1, 5] = -1.

        self['transition', 2, 2] = self.start_params[1]
        self['transition', 2, 3] = self.start_params[2]
        self['transition', 2, 8] = self.start_params[3]

        self['transition', 3, 3] = 1.

        self['transition', 4, 4] = 1.

        self['transition', 5, 1] = 1.

        self['transition', 6, 4] = 1.

        self['transition', 7, 6] = 1.

        self['transition', 8, 7] = 1.

        self['selection', 0, 0] = 1.
        self['selection', 0, 1] = 1.
        self['selection', 0, 2] = self.start_params[2]
        # self['selection', 0, 3] = self.start_params[3]

        self['selection', 1, 0] = 1.

        self['selection', 2, 1] = 1.
        self['selection', 2, 2] = self.start_params[2]
        # self['selection', 2, 3] = self.start_params[3]


        self['selection', 3, 2] = 1.
        self['selection', 4, 3] = 1.
        # self['selection', 5, 3] = 1.

        self['state_cov', 0, 0] = init_state_cov[0, 0]
        self['state_cov', 1, 1] = init_state_cov[2, 2]
        self['state_cov', 2, 2] = init_state_cov[3, 3]
        self['state_cov', 3, 3] = init_state_cov[5, 5]

        # self['obs_cov', 0, 0] = 0.0
        # self['obs_cov', 0, 1] = 0.0
        # self['obs_cov', 1, 0] = 0.0
        # self['obs_cov', 1, 1] = 0.0


    def transform_params(self, params):
        # beta = constrain_stationary_univariate(params[1:])
        gamma2 = params[3]
        gamma1 = params[2]
        beta = params[1]
        sigma1_2 = params[0]**2
        sigma2_2 = params[4] ** 2
        return np.r_[sigma1_2, beta, gamma1, gamma2, sigma2_2]

    def untransform_params(self, params):
        # beta = unconstrain_stationary_univariate(params[1:])
        gamma2 = params[3]
        gamma1 = params[2]
        beta = params[1]
        sigma1 = params[0]**0.5
        sigma2 = params[4]**0.5
        return np.r_[sigma1, beta, gamma1, gamma2, sigma2]

    def update(self, params, **kwargs):

        if self.to_update_params:
            params = super(HP_filter_Model6, self).update(params, **kwargs)

            sigma1_2, beta, gamma1, gamma2, sigma2_2 = params

            self['transition', 0, 3] = gamma1
            self['transition', 0, 2] = beta
            self['transition', 0, 8] = gamma2

            self['transition', 2, 2] = beta
            self['transition', 2, 3] = gamma1
            self['transition', 2, 8] = gamma2

            # self['selection', 0, 1] = beta
            self['selection', 0, 2] = gamma1
            self['selection', 2, 2] = gamma1

            # self['selection', 0, 3] = gamma2
            # self['selection', 2, 3] = gamma2

            self['obs_cov', 0, 0] = 0.0
            self['obs_cov', 0, 1] = 0.0
            self['obs_cov', 1, 0] = 0.0
            self['obs_cov', 1, 1] = 0.0

            self['state_cov', 2, 2] = sigma1_2
            self['state_cov', 3, 3] = sigma2_2



class HP_filter(mle.MLEModel):

    start_params = [0]
    param_names = ['sigma']

    def __init__(self, endog, var0 = 230.0, lamda = 1600.0):

        # Initialize the state space model
        super(HP_filter, self).__init__(endog, k_states=2, k_posdef = 1,
                                         initialization='approximate_diffuse',
                                         loglikelihood_burn = 1)

        self.start_params[0] = var0
        self.lamda = lamda

        # Initialize known components of the state space matrices
        self.ssm['design', 0, 0] = 1
        self.ssm['transition', 0, 0] = 2.
        self.ssm['transition', 0, 1] = -1.
        self.ssm['transition', 1, 0] = 1.
        self.ssm['selection', 0, 0] = 1.

    def transform_params(self, params):
        sigma_2 = params[0]**2
        return np.r_[sigma_2]

    def untransform_params(self, params):
        sigma = params[0]**0.5
        return np.r_[sigma]

    def update(self, params, **kwargs):
        params = super(HP_filter, self).update(params, **kwargs)

        sigma2_eps = params[0]
        sigma2_eta = sigma2_eps / self.lamda

        self.ssm['obs_cov', 0, 0] = sigma2_eps
        self.ssm['state_cov', 0, 0] = sigma2_eta

class HP_filter_Dynamic(mle.MLEModel):

    start_params = [0.0, 0.0]
    param_names = ['beta', 'sigma']

    def __init__(self, endog, init_state, init_state_cov, init_params = [0.95, 1.0], lamda = 72.5, to_update_params=True):

        # Initialize the state space model
        super(HP_filter_Dynamic, self).__init__(endog, k_states=3, k_posdef = 2,
                                         initialization='known', initial_state = init_state,
                                         initial_state_cov = init_state_cov,
                                         loglikelihood_burn = 1)

        self.to_update_params = to_update_params
        self.start_params = init_params
        self.lamda = lamda

        # Initialize known components of the state space matrices
        self.ssm['design', 0, 1] = 1

        self.ssm['transition', 0, 1] = 1.
        self.ssm['transition', 1, 1] = 2.
        self.ssm['transition', 1, 2] = -1.
        self.ssm['transition', 2, 1] = 1.

        self.ssm['design', 0, 0] = self.start_params[0]
        self.ssm['design', 0, 2] = -self.start_params[0]

        self.ssm['transition', 0, 0] = self.start_params[0]
        self.ssm['transition', 0, 2] = -self.start_params[0]

        self.ssm['selection', 0, 0] = 1.
        self.ssm['selection', 1, 1] = 1.

    def transform_params(self, params):
        beta = constrain_stationary_univariate(params[0:1])
        # beta = params[0]
        sigma_2 = params[1]**2
        return np.r_[beta, sigma_2]

    def untransform_params(self, params):
        beta = unconstrain_stationary_univariate(params[0:1])
        # beta = params[0]
        sigma = params[1]**0.5
        return np.r_[beta, sigma]

    def update(self, params, **kwargs):
        params = super(HP_filter_Dynamic, self).update(params, **kwargs)

        beta, sigma2 = params

        if self.to_update_params:
            self.ssm['design', 0, 0] = beta
            self.ssm['design', 0, 2] = -beta
            self.ssm['transition', 0, 0] = beta
            self.ssm['transition', 0, 2] = -beta

        sigma2_eps = params[1]
        sigma2_eta = sigma2_eps / self.lamda

        self.ssm['obs_cov', 0, 0] = sigma2_eps
        self.ssm['state_cov', 0, 0] = sigma2_eps
        self.ssm['state_cov', 1, 1] = sigma2_eta



class HP_filter_Model4_test(mle.MLEModel):

    start_params = [0.0, 0.0, 0.0]
    param_names = ['sigma_exog', 'beta', 'gamma']

    def __init__(self, endog, init_state, init_state_cov, init_params = [1.0, 0.5, 0.5], to_update_params=True):

        # Initialize the state space model
        super(HP_filter_Model4_test, self).__init__(endog, k_states=5, k_posdef=3,  loglikelihood_burn = 1,
                                                    initialization='known', initial_state_cov  = init_state_cov,
                                                    initial_state = init_state)

        self.to_update_params = to_update_params
        self.start_params = init_params
        self.nobs = len(endog)
        self.counter = 0


        # Initialize known components of the state space matrices
        self['design', 0, 0] = 1
        self['design', 1, 3] = 1


        self['transition', 0, 1] = 2.
        self['transition', 0, 4] = -1.
        self['transition', 0, 3] = self.start_params[2]
        self['transition', 0, 2] = self.start_params[1]

        self['transition', 1, 1] = 2.
        self['transition', 1, 4] = -1.

        self['transition', 2, 2] = self.start_params[1]
        self['transition', 2, 3] = self.start_params[2]

        self['transition', 3, 3] = 1.

        self['transition', 4, 1] = 1.

        self['selection', 0, 0] = 1.
        self['selection', 0, 1] = 1.
        self['selection', 0, 2] = self.start_params[2]

        self['selection', 1, 0] = 1.

        self['selection', 2, 1] = 1.
        self['selection', 2, 2] = self.start_params[2]

        self['selection', 3, 2] = 1.

        self['state_cov', 0, 0] = init_state_cov[0, 0]
        self['state_cov', 1, 1] = init_state_cov[2, 2]
        self['state_cov', 2, 2] = init_state_cov[3, 3]

        self['obs_cov', 0, 0] = 0.0
        self['obs_cov', 0, 1] = 0.0
        self['obs_cov', 1, 0] = 0.0
        self['obs_cov', 1, 1] = 0.0


    def transform_params(self, params):
        # beta = constrain_stationary_univariate(params[1:])
        gamma = params[2]
        beta = params[1]
        sigma_2 = params[0]**2
        return np.r_[sigma_2, beta, gamma]

    def untransform_params(self, params):
        # beta = unconstrain_stationary_univariate(params[1:])
        gamma = params[2]
        beta = params[1]
        sigma = params[0]**0.5
        return np.r_[sigma, beta, gamma]

    def update(self, params, **kwargs):

        if self.to_update_params:
            params = super(HP_filter_Model4_test, self).update(params, **kwargs)

            sigma2, beta, gamma = params

            # if self.to_update_params:
            #     self.ssm['design', 0, 0] = beta
            #     self.ssm['design', 0, 2] = -beta
            #     self.ssm['transition', 0, 0] = beta
            #     self.ssm['transition', 0, 2] = -beta

            # sigma2_eps = sigma2
            # sigma2_eta = sigma2_eps / self.lamda

            self['transition', 0, 2] = beta
            self['transition', 2, 2] = beta

            self['transition', 0, 3] = gamma
            self['transition', 2, 3] = gamma

            self['selection', 0, 2] = gamma
            self['selection', 2, 2] = gamma

            self['obs_cov', 0, 0] = 0.0
            self['obs_cov', 0, 1] = 0.0
            self['obs_cov', 1, 0] = 0.0
            self['obs_cov', 1, 1] = 0.0

            self['state_cov', 2, 2] = sigma2