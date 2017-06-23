# Potential_Output
Implementation of a finance-neutral output gap as explained in "Rethinking Potential Output" paper by Borio et al (2015).
The files implement Model 4, 5 and 6 explained in the paper below. 

For e.g Model 6 uses both variables that jointly proxy the financial cycle – credit and property prices. 

real_time_plot.py compares the real-time(expanding window) and ex-post (whole sample) estimates of a HP filtered output gap and finance-neutral gap.

Note: For running the code, please input the Fred api key in the get_fredapi_data fn in state_space_filters.py script

The paper is available at the below link:
https://www.pier.or.th/wp-content/uploads/2015/09/pier_dp_005.pdf
