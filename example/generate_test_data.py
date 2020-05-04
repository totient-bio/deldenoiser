import numpy as np
import pandas as pd
from scipy.stats import beta, norm, poisson
from itertools import product


def _create_lane_index(lmax, include_zero):
    lmin = 0 if include_zero else 1
    lanes = np.array(
        list(
            product(*[list(range(lmin, lmax_c + 1, 1)) for lmax_c in lmax])
        )
    )
    return lanes


def _create_lane_index_table(lmax, include_zero):
    """ Creates a pandas dataframe containing only the lane index combinations.

    Args:
        lmax: cmax-long vector. number of lanes in each cycle
        include_zero (bool): if True, lane indexes start from 0, else from 1

    Returns:
        pandas.DataFrame: len(lmax) + 1 columns:
            "cycle_<c>_lane": extended lane id (0,1,2,... lmax[c]),
                for c = 1,2,... len(lmax)
    """

    lanes = _create_lane_index(lmax, include_zero)
    lane_col_names = [f'cycle_{c}_lane' for c in range(1, len(lmax) + 1, 1)]
    df = pd.DataFrame()
    for col_idx, col_name in enumerate(lane_col_names):
        df[col_name] = lanes[:, col_idx]
    return df


def generate_design_table(lmax):
    """Creates a table containing the sizes of the design for each cycle.

    Args:
        lmax: cmax-long vector. number of lanes in each cycle

    Returns:
        pandas.DataFrame: 2 columns
            "cycle": cycle index
            "lanes": number of lanes in each cycle
    """
    df_design = pd.DataFrame({
        'cycle': list(range(1, len(lmax) + 1, 1)),
        'lanes': lmax
    })
    return df_design


def generate_yield_table(lmax, low=0.4, high=0.8):
    """ Generate yield data, y_{c, l_c} as a pandas dataframe

    Args:
        lmax: cmax-long vector. number of lanes in each cycle
        low: typical low yield value (mid - 2 sigma)
        high: typical high yield value (mid + 2 sigma)

    Returns:
        pandas.DataFrame: 3 columns
            "cycle": cycle index
            "lane": lane index
            "yield": associated yield
    """

    cycles_list = []
    lane_list = []
    for cycle, lmax_c in enumerate(lmax):
        cycles_list.extend([cycle + 1] * lmax_c)
        lane_list.extend(list(range(1, lmax_c + 1, 1)))

    m = 0.5 * (low + high)
    s = 0.25 * (high - low)  # assuming that [low, high] interval is 4 sigma
    a = m * ((1 - m) / s ** 2 - 1)  # first parameter of beta distribution
    b = (1 - m) * a  # second parameter of beta distribution
    yield_arr = beta.rvs(a=a, b=b, size=len(cycles_list))

    df = pd.DataFrame({
        'cycle': cycles_list,
        'lane': lane_list,
        'yield': yield_arr
    })
    return df


def generate_log10_association_constant_table(lmax, min=2.0, high=5.0):
    """ Generate log10 association constants logK_{D} as a pandas dataframe.

    A half-normal distribution is used, starting at `min`, and a width
    set to achive `high` being the 2-sigma point. K is measured in 1 / molar.

    Args:
        lmax: cmax-long vector. number of lanes in each cycle
        min: minimal logK value
        high: min + 2-sigma logK value

    Returns:
        pandas.DataFrame: len(lmax) + 1 columns:
            "cycle_<c>_lane": extended lane id (0,1,2,... lmax[c]),
                for c = 1,2,... len(lmax)
            "log10_K": log10(K[L]), K measured in 1 / molar.
    """

    mu = min
    sigma = 0.5 * (high - min)
    logK = mu + sigma * np.abs(norm.rvs(size=np.prod(np.array(lmax) + 1)))

    df = _create_lane_index_table(lmax, include_zero=True)
    df['log10_K'] = logK

    index_cols = [f'cycle_{c+1}_lane' for c in range(len(lmax))]
    q_arr = df[index_cols].values
    truncates = (q_arr == 0).any(axis=1)
    df_fullcycles = df[~truncates].copy()
    df_truncates = df[truncates].copy()
    return df, df_fullcycles, df_truncates


def generate_sequencing_imbalance_table(lmax, low=0.5, high=2.0):
    """ Generate sequencing imbalance factors from a lognormal distribution.

    Assuming the each cycle have an independent multiplicative effect.

    Args:
        lmax: cmax-long vector. number of lanes in each cycle
        low: typical low value (~5th percentile)
        high: typical low value (~95th percentile)

    Returns:
        pandas.DataFrame: len(lmax) + 1 columns:
            "cycle_<c>_lane": proper lane id (1,2,... lmax[c]),
                for c = 1,2,... len(lmax)
            "sequencing_imbalance": unnormalized imbalance factors
        pandas.DataFrame: 3 columns:
            "cycle", "lane", "imbalance_factor"
    """

    mu_one = 0.5 * (np.log(low) + np.log(high)) / len(lmax)
    sigma_one = 0.25 * (np.log(high) - np.log(low)) / np.sqrt(len(lmax))
    log_factors = []
    for c, lmaxc in enumerate(lmax):
        log_factors.append(mu_one + sigma_one * norm.rvs(size=lmaxc))

    log_factors_for_all_L = np.array(list(product(*log_factors)))
    imbalance = np.exp(np.sum(log_factors_for_all_L, axis=1))

    df = _create_lane_index_table(lmax, include_zero=False)
    df['sequencing_imbalance'] = imbalance

    b = []
    for c in range(len(lmax)):
        bc = np.exp(log_factors[c])
        bc /= np.sum(bc)
        b.append(bc)

    cycles_list = []
    lane_list = []
    for cycle, lmax_c in enumerate(lmax):
        cycles_list.extend([cycle + 1] * lmax_c)
        lane_list.extend(list(range(1, lmax_c + 1, 1)))

    df_b = pd.DataFrame({
        'cycle': cycles_list,
        'lane': lane_list,
        'imbalance_factor': np.concatenate(b)
    })

    return df, df_b


def sample_reads(lamb_total, imbalance):
    """ Generate read counts from sequencing

    Args:
        lamb_total: expected total read count
        imbalance: relative factors between the different entries

    Returns:
        numpy.ndarray: sequencing readcount
    """

    lambs = lamb_total * imbalance / np.sum(imbalance)
    readcounts = poisson.rvs(lambs)

    return readcounts


def generate_data(lmax,
                  sequencing_depth=10,
                  protein_conc=1e-6,
                  selection_cycles=3,
                  sequencing_imbalance_range=(0.5, 2.0),
                  log10_association_constant_range=(2.0, 5.0),
                  yield_range=(0.4, 0.8)
                  ):
    """ Generates test data.

    Args:
        lmax: cmax-long vector. number of lanes in each cycle
        sequencing_depth: expected reads per DNA tag in each of both pre- and
            post-selection sequencing
        protein_conc: protein concentration measured in molar
        selection_cycles: number of selection cycles
        sequencing_imbalance_range: tuple of low and high imbalance values,
            representing 5th and 95th percentiles (from lognormal distribution)
        log10_association_constant_range: tuple of minimum and high (95th percentile)
            log10(association constant) values (from half-normal distribution)
        yield_range: tuple of low and high yield values,
            representing 5th adn 95th percentiles (from beta distribution)

    Returns:
        dict of dataframes:
            "design": DataFrame of the design (i.e. number of lanes in each cycle)
            "sequencing_imbalance": DataFrame of imbalance factors
            "yield": DataFrame of yields
            "log10_association_constant":  DataFrame of log association constants
            "readcount": DataFrame of read counts
            "pre_reacount": DataFrame of only the pre-selection read counts
            "post_readcount": DataFrame of only the post-selection read counts
    """

    df_design = generate_design_table(lmax)

    imb_low, imb_high = sequencing_imbalance_range
    df_imb, df_b = generate_sequencing_imbalance_table(lmax, low=imb_low,
                                                 high=imb_high)

    logK_min, logK_high = log10_association_constant_range
    df_logK, df_logK_fullcycles, df_logK_truncates = \
        generate_log10_association_constant_table(lmax, min=logK_min,
                                                  high=logK_high)
    df_logK['survival_rate'] = (1 + 1.0 / (protein_conc * 10 ** (
        df_logK['log10_K']))) ** (-selection_cycles)
    df_logK_fullcycles['survival_rate'] = (1 + 1.0 / (protein_conc * 10 ** (
        df_logK_fullcycles['log10_K']))) ** (-selection_cycles)
    df_logK_truncates['survival_rate'] = (1 + 1.0 / (protein_conc * 10 ** (
        df_logK_truncates['log10_K']))) ** (-selection_cycles)

    yield_low, yield_high = yield_range
    df_yield = generate_yield_table(lmax, low=yield_low, high=yield_high)

    imb = df_imb['sequencing_imbalance'].values
    y = [np.array(df_yield['yield'][df_yield['cycle'] == c]) for c in
         range(1, len(lmax) + 1, 1)]
    logK = df_logK['log10_K'].values.reshape(
        np.array(lmax) + 1)

    L = _create_lane_index(lmax, include_zero=False)  # proper lane indexes
    s_list = list(product(*[[0, 1] for _ in range(len(lmax))]))
    mu = np.zeros((len(L), len(s_list)))
    for s_idx, s in enumerate(s_list):
        sL = np.array(s) * L
        logKs = logK[tuple(sL.T)]  # picks out logK entries for s*L
        rs = (1 + 1.0 / (protein_conc * 10 ** (logKs))) ** (-selection_cycles)

        ys = []
        for c in range(len(lmax)):
            ys.append(s[c] * y[c] + (1 - s[c]) * (1 - y[c]))
        Y = np.array(list(product(*ys)))  # spread out the ys entries across L
        J = np.prod(Y, axis=1)

        mu[:, s_idx] = imb * J * rs

    total_reads = sequencing_depth * np.prod(lmax)
    N_pre = sample_reads(total_reads, imb)  # vector of size L
    N_post = sample_reads(total_reads, mu)  # matrix of size (L, len(s_list)

    df_N = _create_lane_index_table(lmax, include_zero=False)
    df_N['pre_readcount'] = N_pre
    df_N[f'post_readcount'] = np.sum(N_post, axis=1)
    for s_idx, s in enumerate(s_list):
        s_label = ''.join(map(str, s))
        df_N[f'post_readcount_{s_label}'] = N_post[:, s_idx]

    lane_col_names = [f'cycle_{c}_lane' for c in range(1, len(lmax) + 1, 1)]
    df_N_pre = df_N[lane_col_names + ['pre_readcount']] \
        .rename(columns={'pre_readcount': 'readcount'})
    df_N_post = df_N[lane_col_names + ['post_readcount']] \
        .rename(columns={'post_readcount': 'readcount'})

    return {
        "design": df_design,
        "imbalance_factors": df_b,
        "yields": df_yield,
        "fullcycles": df_logK_fullcycles,
        "truncates": df_logK_truncates,
        "readcount": df_N,
        "preselection_readcount": df_N_pre,
        "postselection_readcount": df_N_post
    }


if __name__ == '__main__':
    import os

    os.makedirs('./data/', exist_ok=True)

    size = 20
    design = [size, size, size]
    prefix = f'size{size}x{size}x{size}'
    data_tables = generate_data(design)

    for key in ['design', 'yields', 'preselection_readcount',
                'postselection_readcount']:
        data_tables[key].to_csv(f'./input/{prefix}_{key}.tsv',
                                sep='\t', index=False)

    for key in ['imbalance_factors', 'fullcycles', 'truncates',
                'readcount']:
        data_tables[key].to_csv(f'./true/{prefix}_{key}.tsv',
                                sep='\t', index=False)
