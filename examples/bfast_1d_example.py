import datetime
import numpy as np
np.set_printoptions(precision=2, linewidth=120)

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

import bfast
#from datasets import *


def interp_nans(x):
    x_nn = np.array(pd.DataFrame(x).interpolate().values.ravel().tolist())
    nans = x_nn[np.isnan(x)]
    return x_nn, nans

def plot(name, y, x, f, season, level=0.05, h=0.15, max_iter=10, nan_clr="crimson", nan_value=0):
    def segmented_plot(arr, bp):
        prev = 0
        vals = np.concatenate((bp, [arr.shape[0] - 1]))
        for i, s in enumerate(vals + 1):
            ind = max(0, prev-1)
            ax.plot(x[ind:s-1], arr[ind:s-1], label="seg {}".format(i+1))
            ax.legend()
            prev = s

    def add_nans(y_n):
        if nans:
            ax.scatter(x_n, y_n, color=nan_clr, label="missing values", marker="x")
            ax.legend()

    print("Plotting {}".format(name))
    y = y.reshape((y.shape[0], 1, 1))
    vo = bfast.BFAST(frequency=f, season_type=season, level=level, h=h, max_iter=max_iter)
    vo.fit(y, x, nan_value=nan_value)
    nans = np.isnan(y).any()

    x_n = x[np.isnan(y[:,0,0])]
    y, y_n = interp_nans(y[:,0,0])
    Tt, Tt_n = interp_nans(vo.trend[:,0,0])
    St, St_n = interp_nans(vo.season[:,0,0])
    Rt, Rt_n = interp_nans(vo.remainder[:,0,0])

    n_Tt_bp = vo.n_trend_breakpoints[0,0]
    n_St_bp = vo.n_season_breakpoints[0,0]

    Tt_bp = vo.trend_breakpoints[:n_Tt_bp,0,0]
    St_bp = vo.season_breakpoints[:n_St_bp,0,0]

    figsz = (16, 10) if season != "none" else (16, 6)

    fig = plt.figure(figsize=figsz)
    # fig.suptitle(name, fontsize=24)
    if season != "none":
        ax = fig.add_subplot(4, 1, 1)
        # plot Y
        ax.set_title("observations")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(x, y)
        add_nans(y_n)

        # plot trend
        ax = fig.add_subplot(4, 1, 2)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("trend")
        if Tt_bp is not None:
            segmented_plot(Tt, Tt_bp)
        else:
            ax.plot(x, Tt)
        add_nans(Tt_n)

        # plot season
        ax = fig.add_subplot(4, 1, 3)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title("season")
        if St_bp is not None:
            segmented_plot(St, St_bp)
        else:
            ax.plot(x, St)
        add_nans(St_n)

        # plot remainder
        ax = fig.add_subplot(4, 1, 4)
        ax.set_title("remainder")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.plot(x, Rt)
        add_nans(Rt_n)

    else:
        ax = fig.add_subplot(1, 1, 1)
        ax.set_title("observations")
        if Tt_bp is not None:
            prev = 0
            vals = np.concatenate((Tt_bp, [y.shape[0] - 1]))
            for i, s in enumerate(vals + 1):
                ind = max(0, prev-1)
                ax.plot(x[ind:s], y[ind:s], label="seg {}".format(i+1))
                ax.legend()
                prev = s
        else:
            ax.plot(x, Tt)

    plt.subplots_adjust(hspace=0.6)
    # plt.savefig(name.lower() + ".png", bbox_inches ="tight")
    plt.show()

def year_fraction(date):
    start = datetime.date(date.year, 1, 1).toordinal()
    year_length = datetime.date(date.year+1, 1, 1).toordinal() - start
    return date.year + float(date.toordinal() - start) / year_length


if __name__ == "__main__":
    
    nan_value = -32768
    df = pd.read_csv(r'data/GroundCover.csv')
    data = 100-np.array(df['bare50percentile'])
    
    dates = pd.to_datetime(df['date'])
    dates_frac = np.array(dates.dt.year + dates.dt.dayofyear/365)
    
    dates_frac = dates_frac[~np.isnan(data)]
    data = data[~np.isnan(data)]
    
    #data_ints = data
    #data = np.copy(data_ints).astype(np.float32)

    # set NaN values
    #data[data_ints==nan_value] = np.nan
    
    # with open(r'data/dates.txt') as f:
    #     dates = f.read().split('\n')
    #     dates = np.array([year_fraction(datetime.datetime.strptime(d, '%Y-%m-%d')) for d in dates if len(d) > 0])
        
    freq = 16
    plot("Ground cover", data, dates_frac, freq, "harmonic", nan_value=nan_value)
    # plot("nile", nile, nile_dates, None, "none")
    # plot("SIMTS", simts_sum, simts_dates, simts_freq, "harmonic", level=0.35, h=0.3, max_iter=2)
    # plot("NDVI", ndvi, ndvi_dates, ndvi_freq, "dummy")
