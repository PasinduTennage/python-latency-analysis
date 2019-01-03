import os
import matplotlib.pyplot as plt
import sys
import math
import numpy as np
import scipy.stats as st
from scipy.stats._continuous_distns import _distn_names
from scipy.optimize import curve_fit


def getLatencyList(filename):
    if os.path.isfile(filename):

        latencies = []
        with open(filename) as f:
            content = f.readlines()
            content=content[1:]
            for row in content:
                success = row.strip().split(",")[7]
                latency = row.strip().split(",")[-3]
                if(success == "true"):
                    latencies.append(int(latency))
        return latencies

    else:
        print("File doesn't exists")
        return []


def getAverageLatency(latency_values):

    return sum(latency_values)/len(latency_values)


def get_percentile(latency_values, percentile):

    return np.percentile(latency_values, percentile)


def get_histogram(latency_values):
    return np.histogram(latency_values)


def draw_histogram(latency_values):
    plt.hist(latency_values)
    plt.show()
    plt.close()


def get_pdf(latency_list):
    np_array = np.array(latency_list)  # convert the list into a numpy array
    ag = st.gaussian_kde(np_array)  # calculate the kernel density function for the latency values
    # list of equidistant values in the range of the latency values
    x = np.linspace(min(latency_list), max(latency_list), (max(latency_list) - min(latency_list)) * 10)
    y = ag(x)  # evaluate the latency values for each x value
    return x, y


def fit_to_distribution(distribution, latency_values):
    distribution = getattr(st, distribution)
    params = distribution.fit(latency_values)

    return params


def make_distribution_pdf(distribution, latency_list):
    distribution = getattr(st, distribution)
    params = distribution.fit(latency_list)

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    x = np.linspace(min(latency_list), max(latency_list), 10000)
    y = distribution.pdf(x, loc=loc, scale=scale, *arg)
    return x, y


def get_q_q_plot(latency_values, distribution):

    distribution = getattr(st, distribution)
    params = distribution.fit(latency_values)

    latency_values.sort()

    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]

    x = []

    for i in range(1, len(latency_values)):
        x.append((i-0.5) / len(latency_values))

    y = distribution.ppf(x, loc=loc, scale=scale, *arg)

    y = list(y)

    emp_percentiles = latency_values[1:]
    dist_percentiles = y

    return emp_percentiles, dist_percentiles


def get_ks_test_stat(latency_values, str_distribution):

    distribution = getattr(st, str_distribution)
    params = distribution.fit(latency_values)
    return st.kstest(latency_values, str_distribution, args=params)  # returns D, p values


def fit_to_all_distributions(latency_values):

    dist_names = _distn_names

    params = {}
    for dist_name in dist_names:
        try:
            dist = getattr(st, dist_name)
            param = dist.fit(latency_values)
            params[dist_name] = param

        except Exception:
            print("Error occurred in fitting")
            params[dist_name] = "Error"

    return params


def get_best_distribution_using_kstest(latency_values):

    params = fit_to_all_distributions(latency_values)
    dist_names = _distn_names

    dist_results = []

    for dist_name in dist_names:
        try:
            param = params[dist_name]

            if param != "Error":

                # Applying the Kolmogorov-Smirnov test
                D, p = st.kstest(latency_values, dist_name, args=param)
                dist_results.append([dist_name, D, p])

        except Exception:

            print("Exception")

    # select the best fitted distribution
    best_dist, best_d, best_p = None, sys.maxsize, 0

    for item in dist_results:
        name = item[0]
        d = item[1]
        p = item[2]
        if not math.isnan(d) and not math.isnan(p):

            if d < best_d:
                best_d = d
                best_dist = name
                best_p = p

    #  store the name of the best fit and its p value

    print("Best fitting distribution: "+str(best_dist))
    print("Best D value: "+ str(best_d))
    print("Best p value: " + str(best_p))
    print("Parameters for the best fit: " + str(params[best_dist]))

    return best_dist, best_d, params[best_dist], dist_results


def get_chi_squared_test_stat(latency_values, str_distribution):

    distribution = getattr(st, str_distribution)
    params = distribution.fit(latency_values)
    histo, bin_edges = np.histogram(latency_values, bins=int(math.sqrt(len(latency_values))), normed=False)
    observed_values = histo
    arg = params[:-2]
    loc = params[-2]
    scale = params[-1]
    cdf = getattr(st, str_distribution).cdf(bin_edges, loc=loc, scale=scale, *arg)
    expected_values = len(latency_values) * np.diff(cdf)
    return st.chisquare(observed_values, expected_values, len(params))  # returns c, p values


def get_best_distribution_using_chisquared_test(latency_values):
    params = fit_to_all_distributions(latency_values)

    histo, bin_edges = np.histogram(latency_values, bins=int(math.sqrt(len(latency_values))), normed=False)

    observed_values = histo

    dist_names = _distn_names
    dist_results = []

    for dist_name in dist_names:

        param = params[dist_name]

        if param != "Error":
            # Applying the SSE test
            arg = param[:-2]
            loc = param[-2]
            scale = param[-1]
            cdf = getattr(st, dist_name).cdf(bin_edges, loc=loc, scale=scale, *arg)
            expected_values = len(latency_values) * np.diff(cdf)
            c, p = st.chisquare(observed_values, expected_values, len(param))
            dist_results.append([dist_name, c, p])

    # select the best fitted distribution
    best_dist, best_c, best_p = None, sys.maxsize, 0

    for item in dist_results:
        name = item[0]
        c = item[1]
        p = item[2]
        if (not math.isnan(c)) and (not math.isnan(p)):
            if p > best_p:

                best_c = c
                best_dist = name
                best_p = p

    # store the name of the best fit and its p value

    print("Best fitting distribution: " + str(best_dist))
    print("Best c value: " + str(best_c))
    print("Best p value: "+str(best_p))

    return best_dist, best_c, params[best_dist], dist_results


def get_max_likelyhood_pareto_index(latency_values):

    return getattr(st, "pareto").fit(latency_values)[0]


def get_max_n_samples(x, n):
    x.sort()
    return x[len(x)-n:]


def get_hill_estimator(latency_values, k=[1000, 2000, 5000, 10000, 20000, 50000, 100000, 200000]):

    hill_estimators = []

    for i in k:
        xs = get_max_n_samples(latency_values, i)
        hs = get_max_likelyhood_pareto_index(xs)
        hill_estimators.append(hs)

    return k, hill_estimators


def trapezoidal_cdf(ag, a, b, n):
    h = np.float(b - a) / n
    s = 0.0
    s += ag(a)[0]/2.0
    for i in range(1, n):
        s += ag(a + i*h)[0]
    s += ag(b)[0]/2.0
    return s * h


def get_cdf(latency_values):
    a = np.array(latency_values)
    ag = st.gaussian_kde(a, bw_method=1e-3)

    cdf = [0]
    x = []
    k = 0

    max_data = max(latency_values)

    while k < max_data:
        x.append(k)
        k = k + 1

    sum_integral = 0

    for i in range(1, len(x)):
        sum_integral = sum_integral + (trapezoidal_cdf(ag, x[i - 1], x[i], 10))
        cdf.append(sum_integral)

    return x, cdf


def trapezoidal_mass(ag, a, b, n):
    h = np.float(b - a) / n
    s = 0.0
    s += a*ag(a)[0]/2.0
    for i in range(1, n):
        s += (a + i*h)*ag(a + i*h)[0]
    s += b*ag(b)[0]/2.0
    return s * h


def get_mass_distribution(latency_values):
    a = np.array(latency_values)
    ag = st.gaussian_kde(a, bw_method=1e-3)

    Fm = [0]
    x = []
    k = 0

    max_data = max(latency_values)

    while k < max_data:
        x.append(k)
        k = k+1

    sum_integral = 0

    for i in range(1, len(x)):
        sum_integral = sum_integral + (trapezoidal_mass(ag, x[i-1], x[i], 10))
        Fm.append(sum_integral)

    sum_integral = sum_integral + (trapezoidal_mass(ag, k - 1, int(max(latency_values)) * 3, (int(max(latency_values)) * 3 - k - 1) * 2))

    Fm_n = [i/sum_integral for i in Fm]

    return x, Fm_n


def get_joint_ratio(xm, Fm, xc, yc):

    best_i = 0
    best_diff = sys.maxsize

    for i in range(len(xc)):
        if abs(yc[i]-(1-Fm[i]))<best_diff:
            best_i = i
            best_diff = abs(yc[i]-(1-Fm[i]))

    p = 100*Fm[best_i]
    return p


def get_n_half(xm, Fm, xc, yc):

    xc = [math.ceil(i) for i in xc]

    i_half = 0
    best_ = sys.maxsize
    for i in range(len(xc)):
        if abs(Fm[i]-0.5) < best_:
            i_half  =i
            best_ = abs(Fm[i]-0.5)

    return 100*(1-yc[i_half])


def get_w_half(xm, Fm, xc, yc):

    xc = [math.ceil(i) for i in xc]

    i_half = 0
    best_ = sys.maxsize
    for i in range(len(xc)):
        if abs(yc[i] - 0.5) < best_:
            i_half = i
            best_ = abs(yc[i] - 0.5)

    return 100*Fm[i_half]


def get_lorenze_curve(y_c, y_m):
    y_c = [100*i for i in y_c]
    y_m = [100 * i for i in y_m]

    return y_c, y_m


def get_lorenze_straight_line():
    x= []
    for i in range(100):
        x.append(i)

    return x, x


def get_survival_function(x, y):

    sf = []
    for i in y:
        sf.append(abs(1-i))
    return x, sf


def get_log_log_complementary_graph(x_cdf, y_cdf):
    xs, ys = get_survival_function(x_cdf, y_cdf)

    xs = [math.log10(i+1/sys.maxsize) for i in xs]
    ys = [math.log10(i+1/sys.maxsize) for i in ys]

    return xs[1:len(xs)-1], ys[1:len(ys)-1]


def f(x, A, B):
    return A*x + B


def get_max_n_samples(x, n):
    x.sort()
    return x[len(x)-n:]


def get_tail_index(xs, ys, latency_values, k = 0.01):

    start_item= get_max_n_samples(latency_values, int(k * len(latency_values)))[0]
    start_item_log = math.log10(start_item)
    tail_x = []
    tail_y = []
    for i in range(len(xs)):
        if xs[i] >= start_item_log:
            tail_x.append(xs[i])
            tail_y.append(ys[i])

    return curve_fit(f, tail_x, tail_y)[0]






