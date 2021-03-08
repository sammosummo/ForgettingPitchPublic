"""Run this.

"""
import json
import os
import shutil
from itertools import chain, combinations, product
from os import makedirs
from os.path import exists, join as pj

import arviz as az
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import pymc3 as pm
import seaborn as sns

from decimal import Decimal
from matplotlib import rcParams
from patsy import dmatrix
from theano import tensor as tt
from scipy.stats import norm, skewnorm

az.Numba.enable_numba()  # unsure if this is necessary


def set_fig_defaults(figsize=(3.5, 3.5 * 0.6)):
    """Makes figures look nice for JASA."""
    rcParams["figure.figsize"] = figsize
    rcParams["font.size"] = 10
    rcParams["font.sans-serif"] = "Arial"
    rcParams["font.family"] = "sans-serif"
    rcParams["xtick.direction"] = "in"
    rcParams["xtick.top"] = True
    rcParams["ytick.direction"] = "in"
    rcParams["ytick.right"] = True
    rcParams["figure.subplot.wspace"] = 0.025
    rcParams["figure.subplot.hspace"] = 0.025 * 2
    rcParams["legend.borderpad"] = 0
    rcParams["legend.handletextpad"] = 0.2
    rcParams["legend.columnspacing"] = 0.5
    rcParams["legend.frameon"] = False
    rcParams["legend.fancybox"] = False
    # sns.set_palette("deep")


def reset_fig():
    """Reset matplotlib's rcParams."""
    mpl.rcParams.update(mpl.rcParamsDefault)


def cartoon():
    """Constructs a cartoon plot of psychometric functions."""
    l = sorted(f for f in os.listdir("../../Desktop/ForgettingPitchPublic/manuscript") if "." not in f)[-1]
    f = pj("../../Desktop/ForgettingPitchPublic/manuscript", l, "fig1.eps")
    if not exists(f):
        print("making figure 1")
        set_fig_defaults()
        rcParams["ytick.right"] = False
        fig, ax0 = plt.subplots(1, 1, constrained_layout=True)
        fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0)
        x = np.linspace(-8, 8)
        ax0.plot(x, 0.05 + (1 - 0.1) * norm.cdf(x), c="k")
        ax0.plot(
            x,
            0.05 + (1 - 0.1) * norm.cdf(x, scale=3),
            "--",
            c="k",
            label="Gradual decay",
        )
        ax0.plot(x, 0.15 + (1 - 0.3) * norm.cdf(x), ":", c="k", label="Sudden death")
        ax0.set_xticks([0])
        ax0.tick_params(top=False)
        ax0.set_xlabel("$\Delta$")
        ax0.set_yticks([0, 1])
        ax0.set_ylabel('Prop. "2nd"')
        ax0.legend()
        sns.despine(top=True, right=True)
        plt.savefig(f, pad_inches=0)
        reset_fig()


def cartoon2():
    """Constructs a cartoon plot of psychometric functions."""
    l = sorted(f for f in os.listdir("../../Desktop/ForgettingPitchPublic/manuscript") if "." not in f)[-1]
    f = pj("../../Desktop/ForgettingPitchPublic/manuscript", l, "fig1b.png")
    if not exists(f):
        print("making figure 1b")
        set_fig_defaults((3.5, 3.5))
        rcParams["ytick.right"] = False
        fig, (ax0, ax1) = plt.subplots(2, 1, constrained_layout=True, sharex=True)
        fig.set_constrained_layout_pads(w_pad=0.0, h_pad=0.0)
        x = np.linspace(-8, 8)
        y0 = 0.05 + (1 - 0.1) * norm.cdf(x)
        ax0.plot(x, y0, c="k")
        y1 = 0.05 + (1 - 0.1) * norm.cdf(x, scale=3)
        ax0.plot(x, y1, "--", c="k", label="Gradual decay")
        y2 = 0.15 + (1 - 0.3) * norm.cdf(x)
        ax0.plot(x, y2, ":", c="k", label="Sudden death")
        ax0.legend()
        y0 = np.gradient(np.gradient(y0))
        ax1.plot(x, y0, c="k")
        ax1.plot(x[[y0.argmin(), y0.argmax()]], [y0.min(), y0.max()], "ko")
        y1 = np.gradient(np.gradient(y1))
        ax1.plot(x, y1, "--", c="k", label="Gradual decay")
        ax1.plot(x[[y1.argmin(), y1.argmax()]], [y1.min(), y1.max()], "ko")
        y2 = np.gradient(np.gradient(y2))
        ax1.plot(x[[y2.argmin(), y2.argmax()]], [y2.min(), y2.max()], "ko")
        ax1.plot(x, y2, ":", c="k", label="Sudden death")
        ax1.set_ylim(y0.min() - y0.max() * 0.1, y0.max() * 1.1)
        ax1.set_xticks([0])
        ax1.tick_params(top=False)
        ax1.set_xlabel("$\Delta$")
        ax1.set_ylabel('Prop. "2nd"')
        sns.despine(fig=fig, top=True, right=True)
        plt.savefig(f, pad_inches=0, dpi=300)
        reset_fig()


def load_data(exp, drop):
    """Returns the data stored in the file `data.csv`, formatted for modelling.

    Args:
        exp (str): Name of the experiment, either "exp1", "exp2", or "both".

    Returns:
        data (pd.DataFrame): Data formatted ready for modeling.

    """
    _data = pd.read_csv("data.csv")
    kwargs = dict(values="response", index=("exp", "listener", "isi", "delta"))
    data = pd.pivot_table(_data, aggfunc=len, **kwargs).reset_index()
    cols = data.columns.tolist()
    cols[4] = "trials"
    data.columns = cols
    _s = pd.pivot_table(_data, aggfunc=sum, **kwargs).reset_index().response
    data["num"] = _s
    data["prop"] = data.num / data.trials
    data.sort_values(["exp", "listener", "isi"], inplace=True)
    if exp in ("exp1", "exp2"):
        data = data[data["exp"] == exp].reset_index(drop=True)
    elif exp == "both":
        data = data[data["exp"].isin(("exp1", "exp2"))]
    else:
        raise ValueError(f"{exp} not recognized")
    data = data[~data.listener.isin(drop)]
    return data


def agnostic_model(exp, restrict, drop):
    """Creates an agnostic model.

    An "agnostic" model is one that allows a fixed effect of ISI on some subset of the
    listener-level free parameters, `{a, b, g, s}`. Note that for coding convenience,
    the missing fixed-effect coefficients are in the model, but aren't connected to the
    design matrix and therefore have no bearing.

    Args:
        exp (str): Either `"exp1"`, `"exp2"`, or `"both"`. If `"both"`, data are
            combined for listeners who participated in both experiments.
        restrict (str): Some iterable of listener-level parameters not to vary across
            ISIs.
        drop (list): Listener IDs to exclude from the model.

    """
    print("constructing agnostic model with these args:", exp, restrict, drop)
    data = load_data(exp, drop)
    dm = dmatrix("0 + C(listener) + C(isi)", data, return_type="dataframe")
    params = {
        "a": {"pre": "α", "link": lambda x: pm.math.sigmoid(np.sqrt(2) * x)},
        "b": {"pre": "β", "link": lambda x: x / 5},
        "g": {"pre": "γ", "link": lambda x: pm.math.sigmoid(tt.sqrt(2) * (x - 1))},
        "n": {"pre": "ν", "link": lambda x: tt.exp(x - 2)},
    }
    coords = {
        "trials": data.index.tolist(),
        "post": list(params.keys()),
        "pre": [params[k]["pre"] for k in params],
        "levels": dm.design_info.column_names,
    }
    dims = {
        "pre_Θ": ["levels", "pre"],
        "Θ": ["levels", "post"],
        "expanded": ["trials", "post"],
        "num_trials": ["trials"],
        "num_rsp": ["trials"],
        "y": ["trials"],
    }

    with pm.Model(coords=coords) as model:

        pre_Θ = pm.Normal("_pre_Θ", 0, 1, dims=dims["pre_Θ"])

        for x in restrict:

            _blank = pre_Θ[-3:, list(params).index(x)]
            pre_Θ = tt.set_subtensor(_blank, tt.zeros_like(_blank))

        pm.Deterministic("pre_Θ", pre_Θ, dims=dims["pre_Θ"])
        Θ = tt.zeros_like(pre_Θ)

        pre_expanded = tt.dot(dm, pre_Θ)
        expanded = tt.zeros_like(pre_expanded)

        for i, param in enumerate(params):
            Θ = tt.set_subtensor(Θ[:, i], params[param]["link"](pre_Θ[:, i]))
            expanded = tt.set_subtensor(
                expanded[:, i], params[param]["link"](pre_expanded[:, i])
            )

        pm.Deterministic("Θ", Θ, dims=dims["Θ"])

        a = expanded[:, list(params).index("a")]
        b = expanded[:, list(params).index("b")]
        g = expanded[:, list(params).index("g")]
        n = expanded[:, list(params).index("n")]

        pm.Deterministic("expanded", expanded, dims=dims["expanded"])

        p = g * a + (1 - g) * pm.invprobit((data.delta.values - b) / n)

        num_trials = pm.Data("num_trials", data.trials.values, dims=dims["num_trials"])
        num_rsp = pm.Data("num_rsp", data.num.values, dims=dims["num_rsp"])
        pm.Binomial(name="y", p=p, n=num_trials, observed=num_rsp, dims=dims["y"])
        print("done constructing model")

        dr = pj(
            "../../Desktop/ForgettingPitchPublic/models",
            "agnostic",
            exp,
            "none" if not restrict else restrict,
            "".join(drop) if drop else "none",
        )

        if not exists(dr):
            makedirs(dr)

        f = pj(dr, "model.nc")
        if not exists(f):
            target = 0.8
            results = sample(coords, dims, target)
            print("saving samples")
            results.to_netcdf(f)
            print("done saving samples")
        else:
            print("loading samples")
            results = az.from_netcdf(f)

        plot_energy(dr, results)
        plot_dist_comparison(coords, dims, dr, results, "pre_Θ")
        plot_dist_comparison(coords, dims, dr, results, "Θ")
        print_stats(dr, results)
        plot_psyfun(data, dr, params, results)
        plot_psyfun_epl(data, dr, params, results, True)
        plot_psyfun_epl(data, dr, params, results, False)
        plot_posteriors(data, dr, params, coords, results)
        plot_posteriors_epl(data, dr, params, coords, results, True)
        plot_posteriors_epl(data, dr, params, coords, results, False)
        savage_dickey(dr, results, coords)
        var_table(data, dr, results)
    print("done with model")


def agnostic_model_with_expeff():
    """Creates an agnostic model with experiment effects.

    """
    print("constructing agnostic model with experiment effects:")
    data = load_data("both", [])
    dm = dmatrix("0 + C(listener) + C(isi) * C(exp)", data, return_type="dataframe")
    params = {
        "a": {"pre": "α", "link": lambda x: pm.math.sigmoid(np.sqrt(2) * x)},
        "b": {"pre": "β", "link": lambda x: x / 5},
        "g": {"pre": "γ", "link": lambda x: pm.math.sigmoid(tt.sqrt(2) * (x - 1))},
        "n": {"pre": "ν", "link": lambda x: tt.exp(x - 2)},
    }
    coords = {
        "trials": data.index.tolist(),
        "post": list(params.keys()),
        "pre": [params[k]["pre"] for k in params],
        "levels": dm.design_info.column_names,
    }
    dims = {
        "pre_Θ": ["levels", "pre"],
        "Θ": ["levels", "post"],
        "expanded": ["trials", "post"],
        "num_trials": ["trials"],
        "num_rsp": ["trials"],
        "y": ["trials"],
    }

    with pm.Model(coords=coords) as model:

        pre_Θ = pm.Normal("_pre_Θ", 0, 1, dims=dims["pre_Θ"])

        pm.Deterministic("pre_Θ", pre_Θ, dims=dims["pre_Θ"])
        Θ = tt.zeros_like(pre_Θ)

        pre_expanded = tt.dot(dm, pre_Θ)
        expanded = tt.zeros_like(pre_expanded)

        for i, param in enumerate(params):
            Θ = tt.set_subtensor(Θ[:, i], params[param]["link"](pre_Θ[:, i]))
            expanded = tt.set_subtensor(
                expanded[:, i], params[param]["link"](pre_expanded[:, i])
            )

        pm.Deterministic("Θ", Θ, dims=dims["Θ"])

        a = expanded[:, list(params).index("a")]
        b = expanded[:, list(params).index("b")]
        g = expanded[:, list(params).index("g")]
        n = expanded[:, list(params).index("n")]

        pm.Deterministic("expanded", expanded, dims=dims["expanded"])

        p = g * a + (1 - g) * pm.invprobit((data.delta.values - b) / n)

        num_trials = pm.Data("num_trials", data.trials.values, dims=dims["num_trials"])
        num_rsp = pm.Data("num_rsp", data.num.values, dims=dims["num_rsp"])
        pm.Binomial(name="y", p=p, n=num_trials, observed=num_rsp, dims=dims["y"])
        print("done constructing model")

        dr = pj(
            "../../Desktop/ForgettingPitchPublic/models",
            "agnostic_with_exp"
        )

        if not exists(dr):
            makedirs(dr)

        f = pj(dr, "model.nc")
        if not exists(f):
            target = 0.8
            results = sample(coords, dims, target)
            print("saving samples")
            results.to_netcdf(f)
            print("done saving samples")
        else:
            print("loading samples")
            results = az.from_netcdf(f)

        plot_energy(dr, results)
        plot_dist_comparison(coords, dims, dr, results, "pre_Θ")
        plot_dist_comparison(coords, dims, dr, results, "Θ")
        print_stats(dr, results)
        plot_psyfun(data, dr, params, results)
        plot_posteriors(data, dr, params, coords, results)
        savage_dickey(dr, results, coords)
        var_table(data, dr, results)
    print("done with model")


def plot_energy(dr, results):
    """Produce an energy plot. BFMI are printed in the legend.
    Args:
        dr (str): Path to save figure.
        results (az.InferenceData): An arViZ object ready for plotting and analysis.

    """
    f = pj(dr, "energy.png")
    if not exists(f):
        print("making energy plot")
        az.plot_energy(results)
        plt.savefig(f)
        plt.close()


def plot_dist_comparison(coords, dims, dr, results, x):
    """Produce a figure that compares prior and posterior distributions for sotchastic
    random variables.

    Args:
        coords (dict): Dictionary of coordinate systems used by the model.
        dims (dict): Dictionary of dimensions used by the model.
        dr (str): Path to save figure.
        results (az.InferenceData): An arViZ object ready for plotting and analysis.
        x (str): Name of parameter that appears in all models.

    """
    f = pj(dr, "dists")
    if not exists(f):
        print("making dist comparison plots")
        makedirs(f)
        for coord in product(*[coords[dim] for dim in dims[x]]):
            f = pj(dr, "dists", "_".join([x] + list(coord)) + ".png")
            dic = dict(zip(dims[x], coord))
            if not exists(f):
                az.plot_dist_comparison(results, var_names=[x], coords=dic)
                plt.savefig(f)
                plt.close()


def plot_psyfun(data, dr, params, results):
    """Plots psychometric functions with modelling results overlain.

    Args:
        data (pd.DataFrame): Data augmented with model parameters.
        dr (str): Path to save summary stats.
        exp (str): Either `"exp1"`, `"exp2"`, or `"both"`. If `"both"`, data are
            combined for listeners who participated in both experiments.
        params (dict): Parameter dictionary.
        results (az.InferenceData): An arViZ object ready for plotting and analysis.

    """
    f = pj(dr, "psyfun.eps")
    if not exists(f):
        print("making psyfun figure")
        augment_data(data, params, results)
        nlisteners = data.listener.nunique()
        nisis = data.isi.nunique()

        set_fig_defaults((8, nlisteners * 0.8))
        fig = plt.figure(constrained_layout=False)  # takes **forever** if `True`!
        axes = []
        for i, ((listener, isi), df) in enumerate(data.groupby(["listener", "isi"]), 1):

            ax = fig.add_subplot(nlisteners, nisis, i)
            ax.set_ylim(-0.1, 1.1)
            listener = listener.replace("L0", "L")
            if i <= data.isi.nunique():
                s = f"ISI = {isi} s".replace(".0", "")
                ax.set_title(s, fontsize=rcParams["font.size"])
            if i != 1:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel('Prop. "2nd"')
            if i != (data.listener.nunique() * data.isi.nunique()):
                plt.setp(ax.get_xticklabels(), visible=False)
            else:
                ax.set_xlabel(r"$\Delta$ (semitones)")
            if i % data.isi.nunique() == 1:
                ax.text(data.delta.min(), 1, listener, verticalalignment="top")

            for _e, _c in zip(["exp1", "exp2"], ["o", "v"]):
                _df = df[df.exp == _e]
                ax.plot(_df.delta, _df.prop, _c, fillstyle="none", mec="black")
                ax.fill_between(
                    _df.delta, _df.ppc_lo, _df.ppc_hi, fc="lightgrey", zorder=-10
                )

            x = np.linspace(data.delta.min(), data.delta.max())
            row = df.iloc[0]
            try:
                a, b, g, n = row[["a", "b", "g", "n"]]
                p = a * g + (1 - g) * norm.cdf((x - b) / n)
            except KeyError:
                a, b, l, s = row[["a", "b", "l", "s"]]
                d = row["d"] if "d" in row.index else 0
                u = row["u"] if "u" in row.index else 0
                soa = isi + 0.1
                m = d * soa
                v = 2 * s ** 2 + m
                q = 1 - (1 - u) ** soa
                g = 1 - (1 - l) * (1 - q)
                p = g * a + (1 - g) * norm.cdf((x - b) / np.sqrt(v))
            ax.plot(x, p, "black")
            axes.append(ax)
        axes[0].get_shared_x_axes().join(*axes)
        fig.savefig(f, bbox_inches="tight", pad_inches=0)
        plt.close()
        reset_fig()

def plot_psyfun_epl(data, dr, params, results, dark):
    """Plots psychometric functions with modelling results overlain.

    Args:
        data (pd.DataFrame): Data augmented with model parameters.
        dr (str): Path to save summary stats.
        exp (str): Either `"exp1"`, `"exp2"`, or `"both"`. If `"both"`, data are
            combined for listeners who participated in both experiments.
        params (dict): Parameter dictionary.
        results (az.InferenceData): An arViZ object ready for plotting and analysis.

    """
    dr = "/Volumes/samuelrobertmathias/Documents/EPL"
    f = pj(dr, f"psyfun-{'dark' if dark else 'light'}.eps")
    c = "#fffff0" if dark else "#303030"
    if not exists(f):
        if dark:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        print("making psyfun figure")
        augment_data(data, params, results)
        data = data[data.listener == "L02"].copy()
        nlisteners = data.listener.nunique()
        nisis = data.isi.nunique()

        set_fig_defaults((7.4, nlisteners * 1.4))
        rcParams["font.size"] = 12
        fig = plt.figure(constrained_layout=False)  # takes **forever** if `True`!
        axes = []
        for i, ((listener, isi), df) in enumerate(data.groupby(["listener", "isi"]), 1):

            ax = fig.add_subplot(nlisteners, nisis, i)
            ax.set_ylim(-0.1, 1.1)
            listener = listener.replace("L0", "L")
            if i <= data.isi.nunique():
                s = f"ISI = {isi} s".replace(".0", "")
                ax.text(data.delta.min(), 1, s, verticalalignment="top", fontsize=10)
            if i != 1:
                plt.setp(ax.get_yticklabels(), visible=False)
            else:
                ax.set_ylabel(r'p("2nd")')
            if i != (data.listener.nunique() * data.isi.nunique()):
                plt.setp(ax.get_xticklabels(), visible=True)
            else:
                ax.set_xlabel(r"$\Delta$ (semitones)")

            for _e, _c in zip(["exp1", "exp2"], ["o", "o"]):
                _df = df[df.exp == _e]
                ax.plot(_df.delta, _df.prop, _c, fillstyle="none", mec=c)
            x = np.linspace(data.delta.min(), data.delta.max())
            row = df.iloc[0]
            try:
                a, b, g, n = row[["a", "b", "g", "n"]]
                p = a * g + (1 - g) * norm.cdf((x - b) / n)
            except KeyError:
                a, b, l, s = row[["a", "b", "l", "s"]]
                d = row["d"] if "d" in row.index else 0
                u = row["u"] if "u" in row.index else 0
                soa = isi + 0.1
                m = d * soa
                v = 2 * s ** 2 + m
                q = 1 - (1 - u) ** soa
                g = 1 - (1 - l) * (1 - q)
                p = g * a + (1 - g) * norm.cdf((x - b) / np.sqrt(v))
            ax.plot(x, p, "C3", lw=3)
            axes.append(ax)
        axes[0].get_shared_x_axes().join(*axes)
        fig.savefig(f, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close()
        reset_fig()

def augment_data(data, params, results):
    k = ["chain", "draw"]
    _samples = results.posterior["expanded"].mean(k).values
    data[list(params.keys())] = _samples
    pp = results.posterior_predictive["y"]
    data["ppc_lo"] = pp.quantile(0.01, k).values / data.trials.values
    data["ppc_hi"] = pp.quantile(0.99, k).values / data.trials.values


def plot_posteriors(data, dr, params, coords, results):
    """Plot group-average effects of n and g.

    Args:
        data (pd.DataFrame): Data augmented with model parameters.
        dr (str): Path to save summary stats.
        exp (str): Either `"exp1"`, `"exp2"`, or `"both"`. If `"both"`, data are
            combined for listeners who participated in both experiments.
        params (dict): Parameter dictionary.
        coords (dict): Coordinate dictionary.
        results (az.InferenceData): An arViZ object ready for plotting and analysis.

    """
    f = pj(dr, "posteriors.eps")
    if not exists(f):
        print("making posterior figure")
        augment_data(data, params, results)
        set_fig_defaults((7, 7 * 0.6))
        rcParams["xtick.top"] = False
        fig = plt.figure(constrained_layout=True)
        names = {
            "a": "guessing preference ($a$)",
            "b": "bias ($b$)",
            "n": "internal noise ($n$)",
            "g": "guessing probability ($g$)",
            "l": "lapse probability ($l$)",
            "d": "decay rate ($d$)",
            "s": "sensory noise ($s$)",
            "u": "sudden-death prob. ($u$)",
        }
        gs = fig.add_gridspec(2, 2 if "g" in data.columns else 4)
        isis = [2, 5, 10]

        for i, param in enumerate(params):

            ax = fig.add_subplot(gs[i // 2, i % 2], zorder=10)
            s = f"Group difference in {names[param]}\nrelative to ISI = 0.5 s"
            ax.set_xlabel(s)
            ax.set_yticks([])
            ax.patch.set_alpha(0)
            sns.despine(ax=ax, left=True, right=True, bottom=False, top=True)

        fig.canvas.draw()

        for i, param in enumerate(params):

            ax = fig.axes[i]
            pos = ax.get_position()
            height = 0.5 * pos.height
            _axes = []
            _lims = np.array([])

            for j, isi in reversed(list(enumerate(reversed(isis)))):

                y0 = pos.y0 + j * 0.25 * pos.height
                _ax = fig.add_axes((pos.x0, y0, pos.width, height))
                _axes.append(_ax)
                _ax.patch.set_alpha(0)
                _ax.set_yticks([])
                _ax.set_xticks([])
                ix = [l for l in coords["levels"] if f"T.{isi}.0" in l][0]
                p = params[param]["pre"]
                samples = results.posterior["_pre_Θ"].sel(levels=[ix], pre=[p])
                samples = samples.values.ravel()
                kwargs = dict(bins=20, histtype="step", color="lightgray", density=True)
                vals, bins, _ = _ax.hist(samples, **kwargs)
                _lims = np.concatenate([_lims, bins])
                start, stop = az.hdi(samples)
                for n, l, r in zip(vals, bins, bins[1:]):
                    if l > start:
                        if r < stop:
                            _ax.fill_between([l, r], 0, [n, n], color="lightgray")
                        elif l < stop < r:
                            _ax.fill_between([l, stop], 0, [n, n], color="lightgray")
                    elif l < start < r:
                        _ax.fill_between([start, r], 0, [n, n], color="lightgray")
                _ax.set_ylim(-vals.max() * 0.2, vals.max() * 1.1)
                sns.despine(ax=_ax, left=True, bottom=True)
                x = np.linspace(bins[0], bins[-1])
                _ax.plot(x, skewnorm.pdf(x, *skewnorm.fit(samples)), "black")
                side = "right" if param == "b" and isi == 2 else "left"
                _ax.text(
                    -0.1
                    if param == "g" and isi == 10
                    else (bins[0] if side == "left" else bins[-1]),
                    vals.max() * 0.2,
                    f"ISI = {isi} s",
                    va="bottom",
                    ha=side,
                )

            ax.set_xlim(np.array(_lims).min(), np.array(_lims).max())
            [_ax.set_xlim(*ax.get_xlim()) for _ax in _axes]

        fig.savefig(f, bbox_inches="tight", pad_inches=0)
        plt.close()


def plot_posteriors_epl(data, dr, params, coords, results, dark):
    """Plot group-average effects of n and g.

    Args:
        data (pd.DataFrame): Data augmented with model parameters.
        dr (str): Path to save summary stats.
        exp (str): Either `"exp1"`, `"exp2"`, or `"both"`. If `"both"`, data are
            combined for listeners who participated in both experiments.
        params (dict): Parameter dictionary.
        coords (dict): Coordinate dictionary.
        results (az.InferenceData): An arViZ object ready for plotting and analysis.

    """
    dr = "/Volumes/samuelrobertmathias/Documents/EPL"
    f = pj(dr, f"posteriors-{'dark' if dark else 'light'}.eps")
    if not exists(f):
        if dark:
            plt.style.use("dark_background")
        else:
            plt.style.use("default")
        print("making posterior figure")
        augment_data(data, params, results)
        set_fig_defaults((7.4, 3))
        rcParams["font.size"] = 12
        rcParams["xtick.top"] = False
        fig = plt.figure(constrained_layout=True)
        names = {
            "a": "guessing preference ($a$)",
            "b": "bias ($b$)",
            "n": "internal noise ($n$)",
            "g": "guessing probability ($g$)",
            "l": "lapse probability ($l$)",
            "d": "decay rate ($d$)",
            "s": "sensory noise ($s$)",
            "u": "sudden-death prob. ($u$)",
        }
        gs = fig.add_gridspec(2, 2 if "g" in data.columns else 4)
        isis = [2, 5, 10]

        for i, param in enumerate("ng"):

            ax = fig.add_subplot(gs[i // 2, i % 2], zorder=10)
            s = f"Group difference in {names[param]}\nrel. to ISI = 0.5 s"
            ax.set_xlabel(s)
            ax.set_yticks([])
            ax.patch.set_alpha(0)
            sns.despine(ax=ax, left=True, right=True, bottom=False, top=True)

        fig.canvas.draw()

        for i, param in enumerate("ng"):

            ax = fig.axes[i]
            pos = ax.get_position()
            height = 0.5 * pos.height
            _axes = []
            _lims = np.array([])
            c = {"n": "C4", "g": "C5"}[param]

            for j, isi in reversed(list(enumerate(reversed(isis)))):

                y0 = pos.y0 + j * 0.25 * pos.height
                _ax = fig.add_axes((pos.x0, y0, pos.width, height))
                _axes.append(_ax)
                _ax.patch.set_alpha(0)
                _ax.set_yticks([])
                _ax.set_xticks([])
                ix = [l for l in coords["levels"] if f"T.{isi}.0" in l][0]
                p = params[param]["pre"]
                samples = results.posterior["_pre_Θ"].sel(levels=[ix], pre=[p])
                samples = samples.values.ravel()
                x = np.linspace(samples.min(), samples.max())
                _c = "w" if dark else "k"
                _ax.plot(x, skewnorm.pdf(x, *skewnorm.fit(samples)), _c, lw=2)
                _lims = np.concatenate([_lims, [x.min(), x.max()]])
                sns.despine(ax=_ax, left=True, bottom=True)
                x = np.linspace(*az.hdi(samples))
                y = skewnorm.pdf(x, *skewnorm.fit(samples))
                _ax.fill_between(x, 0, y, color=c)

            ax.set_xlim(np.array(_lims).min(), np.array(_lims).max())
            [_ax.set_xlim(*ax.get_xlim()) for _ax in _axes]

        fig.savefig(f, transparent=True, bbox_inches="tight", pad_inches=0)
        plt.close()

def savage_dickey(dr, results, coords):
    """Compute Bayes factors using the Savage-Dickey approximation."""
    f = pj(dr, "bfs.json")
    if not exists(f):
        print("saving bayes factors")
        dic = {}
        params = ("νγ", "ng")
        for p, q in zip(*params):
            for i, isi in enumerate([2, 5, 10]):
                lvl = [l for l in coords["levels"] if f"T.{isi}.0" in l][0]
                k = {"levels": [lvl], "pre": [p]}
                a = results.prior["_pre_Θ"].sel(**k).values.ravel()
                num = skewnorm.pdf(0, *skewnorm.fit(a))
                b = results.posterior["_pre_Θ"].sel(**k).values.ravel()
                den = skewnorm.pdf(0, *skewnorm.fit(b))
                dic[f"{q}_{isi}"] = num / den
        json.dump(dic, open(f, "w"))


def print_stats(dr, results):
    """Calculate diagnostic stats for a given model.

    Args:
        dr (str): Path to save summary stats.
        results (az.InferenceData): An arViZ object ready for plotting and analysis.

    """
    f = pj(dr, "stats.json")
    if not exists(f):
        print("saving diagnostic statistics")
        dic = {
            "bfmi_min": az.bfmi(results).min(),
            "bfmi_max": az.bfmi(results).max(),
        }
        x = az.rhat(results, var_names=["pre_Θ"]).to_array().values
        x = x[~np.isnan(x)]
        dic["rhat_max"] = x.max()
        x = az.ess(results, var_names=["pre_Θ"]).to_array().values
        x = x[~np.isnan(x)]
        dic["ess_min"] = x.min()
        y_true = results.observed_data["y"].values
        y_pred = results.posterior_predictive.stack(sample=("chain", "draw"))
        y_pred = y_pred["y"].values.T
        dic["r2_score"], dic["r2_sd"] = az.r2_score(y_true, y_pred).values
        json.dump(dic, open(f, "w"))


def var_table(data, dr, results):
    f = pj(dr, "vars.tex")
    if not exists(f):
        print("saving table of posterior means")
        k = ["chain", "draw"]
        df = results.posterior["expanded"].mean(k).to_dataframe().reset_index()
        df = pd.pivot_table(df, "expanded", index="trials", columns="post")
        s = ["listener", "isi"]
        _drops = ["exp", "delta", "trials", "num", "prop"]
        try:
            df = data.join(df.reset_index(drop=True)).drop_duplicates(subset=s)
        except ValueError:
            df = data
            _drops += ["ppc_lo", "ppc_hi"]
        df.drop(_drops, inplace=True, axis=1)
        # df["listener"] = df.listener.str.replace("L0", "L")
        # df["isi"] = df.isi.astype(str).str.replace(".0", "")
        # df.set_index(["listener", "isi"], inplace=True)
        df = pd.pivot_table(
            df, values=["a", "b", "g", "n"], index="listener", columns="isi"
        )
        df.index = [s.replace("L0", "L") for s in df.index.tolist()]
        a = "Posterior means of listener-level variables from the agnostic model."
        s = df.to_latex(
            sparsify=True,
            float_format="${:0.3f}$".format,
            escape=False,
            caption=a,
            label="vars",
        )
        s = s.replace("isi", "ISI (s)")
        s = s.replace("listener", "Listener")
        s = " ".join(s.split())
        open(f, "w").write(s)


def fit_agnostic_models():
    """Fit the agnostic model to each experiment separately, then to both experiments
    together.

    """
    print("constructing agnostic models")
    exps = ("both", "exp1", "exp2")
    rs = ("", "ab", "g")
    drops = ([], ["L06", "L07"])
    for args in product(exps, rs, drops):
        agnostic_model(*args)
    agnostic_model_with_expeff()


def sample(coords, dims, target):
    """Sample from the prior, prior predictive, posterior, and posterior predictive
    distributions.

    Args:
        coords (dict): Dictionary of coordinate systems used by the model.
        dims (dict): Dictionary of dimensions used by the model.
        target (int): Initial lower guess at `target_accept` used during sampling. If
            sampling produces any divergences, this value is increased by `0.05` and
            sampling is performed again.

    Returns:
        results (az.InferenceData): An arViZ object ready for plotting and analysis.

    """
    while target < 1:
        print(f"sampling with target_accept={target}")
        trace = pm.sample(10000, tune=5000, target_accept=target)
        if trace["diverging"].sum() == 0:
            break
        target += 0.005
    print(f"final target_accept={target}")
    results = az.from_pymc3(
        trace=trace,
        prior=pm.sample_prior_predictive(3000),
        posterior_predictive=pm.sample_posterior_predictive(trace),
        coords=coords,
        dims=dims,
    )
    return results


def prescriptive_model(exp, decay, death, prior_type, drop):
    """Creates a prescriptive model.

    A "prescriptive" model is one where internal noise is assumed to be a combination of
    sensory noise and memory noise, the latter following Wiener diffusion, and guessing
    probability is the union of lapses and sudden death, where the latter has a constant
    probability per unit time.

    Args:
        exp (str): Either `"exp1"`, `"exp2"`, or `"both"`. If `"both"`, data are
            combined for listeners who participated in both experiments.
        decay (bool): Whether or not to allow gradual decay.
        death (bool): Whether or not to allow sudden death.
        prior_type (str): If `"non-hierarchical"`, each listener-level parameter is
            assigned an independent informative prior. If `"univariate hierarchical"`,
            listener-level parameters of the same kind (e.g., `a`) are assumed to be
            drawn from the same distribution and are assigned the same prior.
            `"multivariate hierarchical"`, listener-level parameters are assumed to be
            drawn from the same distribution and all are assigned a multivariate prior,
            which allows covariance between different parameters within the same
            listener.

    """
    data = load_data(exp, drop)
    dm = dmatrix("0 + C(listener)", data, return_type="dataframe")
    params = {
        "a": {"pre": "α", "link": lambda x: pm.math.sigmoid(np.sqrt(2) * x)},
        "b": {"pre": "β", "link": lambda x: x / 5},
        "l": {"pre": "λ", "link": lambda x: pm.math.sigmoid(tt.sqrt(2) * (x - 1))},
        "s": {"pre": "ς", "link": lambda x: tt.exp(x - 2)},
    }
    if decay:
        params["d"] = {"pre": "δ", "link": lambda x: tt.exp(x - 5)}
    if death:
        params["u"] = {
            "pre": "υ",
            "link": lambda x: pm.math.sigmoid(tt.sqrt(2) * (x - 2)),
        }
    coords = {
        "trials": data.index.tolist(),
        "post": list(params.keys()),
        "pre": [params[k]["pre"] for k in params],
        "listeners": [x.split("[")[1].strip("]") for x in dm.design_info.column_names],
    }
    dims = {
        "pre_Θ": ["listeners", "pre"],
        "pre_μ": ["pre"],
        "Θ": ["listeners", "post"],
        "μ": ["post"],
        "p": ["trials"],
        "soa": ["trials"],
        "num_trials": ["trials"],
        "num_rsp": ["trials"],
        "y": ["trials"],
    }

    with pm.Model(coords=coords) as model:

        if prior_type == "non-hierarchical":
            pre_Θ = pm.Normal("pre_Θ", 0, 1, dims=dims["pre_Θ"])
            pre_μ = pm.Deterministic(
                "pre_μ", tt.mean(pre_Θ, axis=0), dims=dims["pre_μ"]
            )

        if prior_type == "univariate":
            pre_μ = pm.Normal("pre_μ", 0, 1, dims=dims["pre_μ"])
            dims["σ"] = ["pre"]
            σ = pm.Exponential("σ", 1, dims=dims["pre_μ"])
            dims["ζ"] = ["listeners", "pre"]
            ζ = pm.Normal("ζ", 0, 1, dims=dims["ζ"])
            pre_Θ = pm.Deterministic("pre_Θ", pre_μ + ζ * σ, dims=dims["pre_Θ"])

        if prior_type == "multivariate":
            pre_μ = pm.Normal("pre_μ", 0, 1, dims=dims["pre_μ"])
            chol, *_ = pm.LKJCholeskyCov(
                "chol",
                eta=2,
                n=len(params),
                sd_dist=pm.Exponential.dist(1),
                compute_corr=True,
            )
            dims["ζ_tr"] = ["pre", "listeners"]
            ζ = pm.Normal("ζ_tr", 0, 1, dims=dims["ζ_tr"])
            pre_Θ = pm.Deterministic(
                "pre_Θ", pre_μ + tt.dot(chol, ζ).T, dims=dims["pre_Θ"]
            )

        Θ = tt.zeros_like(pre_Θ)
        μ = tt.zeros_like(pre_μ)

        for i, param in enumerate(params):
            Θ = tt.set_subtensor(Θ[:, i], params[param]["link"](pre_Θ[:, i]))
            μ = tt.set_subtensor(μ[i], params[param]["link"](pre_μ[i]))

        pm.Deterministic("Θ", Θ, dims=dims["Θ"])
        pm.Deterministic("μ", μ, dims=dims["μ"])
        expanded = pm.Deterministic("expanded", tt.dot(dm, Θ))

        a = expanded[:, list(params).index("a")]
        b = expanded[:, list(params).index("b")]
        d = expanded[:, list(params).index("d")] if decay else 0
        l = expanded[:, list(params).index("l")]
        s = expanded[:, list(params).index("s")]
        u = expanded[:, list(params).index("u")] if death else 0

        soa = pm.Data("soa", data.isi + 0.1, dims=dims["soa"])
        m = d * soa
        v = 2 * s ** 2 + m
        q = 1 - (1 - u) ** soa
        g = 1 - (1 - l) * (1 - q)
        p = g * a + (1 - g) * pm.invprobit((data.delta.values - b) / tt.sqrt(v))
        pm.Deterministic("p", p, dims=dims["p"])

        num_trials = pm.Data("num_trials", data.trials.values, dims=dims["num_trials"])
        num_rsp = pm.Data("num_rsp", data.num.values, dims=dims["num_rsp"])
        pm.Binomial(name="y", p=p, n=num_trials, observed=num_rsp, dims=dims["y"])

        dr = pj(
            "../../Desktop/ForgettingPitchPublic/models",
            "prescriptive",
            exp,
            f"decay={decay}",
            f"death={death}",
            prior_type,
            "".join(drop) if drop else "none",
        )
        if not exists(dr):
            makedirs(dr)

        f = pj(dr, "model.nc")
        if not exists(f):
            if "multi" in prior_type:
                target = 0.94
            elif "uni" in prior_type:
                target = 0.94
            else:
                target = 0.87
            results = sample(coords, dims, target)
            results.to_netcdf(f)
        else:
            results = az.from_netcdf(f)

        plot_energy(dr, results)
        print_stats(dr, results)
        plot_psyfun(data, dr, params, results)

        # print("computing LOO")
        # x = az.loo(results, pointwise=True).pareto_k
        # print(x[x > 0.7])

    return results


def get_worst_stats(exp, prior_type, drop):

    dics = []
    for decay, death in [(True, True), (True, False), (False, True)]:
        dr = pj(
            "../../Desktop/ForgettingPitchPublic/models",
            "prescriptive",
            exp,
            f"decay={decay}",
            f"death={death}",
            prior_type,
            "".join(drop) if drop else "none",
        )
        dics.append(json.load(open(pj(dr, "stats.json"))))
    df = pd.DataFrame(dics)
    dic = {}
    for c in df.columns:
        dic["pres_" + c] = df[c].max() if "max" in c else df[c].min()
    dr = pj(
        "../../Desktop/ForgettingPitchPublic/models",
        "prescriptive",
        exp,
        prior_type,
        "".join(drop) if drop else "none",
    )
    json.dump(dic, open(pj(dr, "stats.json"), "w"))


def fit_prescriptive_models():
    print("fitting prescriptive models")
    exps = ("both", "exp1", "exp2")
    priors = ("multivariate", "univariate", "non-hierarchical")
    drops = ([], ["L06", "L07"])
    for exp, prior, drop in product(exps, priors, drops):
        models = {
            "Decay and death": prescriptive_model(exp, True, True, prior, drop),
            "Decay only": prescriptive_model(exp, True, False, prior, drop),
            "Death only": prescriptive_model(exp, False, True, prior, drop),
        }
        dr = pj(
            "../../Desktop/ForgettingPitchPublic/models",
            "prescriptive",
            exp,
            prior,
            "".join(drop) if drop else "none",
        )
        if not exists(dr):
            makedirs(dr)
        get_worst_stats(exp, prior, drop)
        f = pj(dr, "comparison.tex")
        if not exists(f):
            df = az.compare(models)
            df.drop(["rank", "warning", "loo_scale", "weight"], axis=1, inplace=True)
            df.columns = ["IC", "$N_p$", r"$\Delta$IC", "se(IC)", r"se($\Delta$IC)"]
            df.index.name = "Model"
            df.to_csv(f + ".csv")
            c = "Results of LOOCV model comparison. Higher information criterion (IC) indicates a better model. $N_p$ is effective number of parameters, se() is standard error of estimate, $\Delta$ is difference from best model."
            s = df.to_latex(
                sparsify=True,
                float_format="${:0.3f}$".format,
                escape=False,
                caption=c,
                label="loo_table",
            )
            s = " ".join(s.split()).replace("\midrule", "\hline\hline")
            open(f, "w").write(s)


def populate_manuscript(exp, restrict, drop, prior_type):
    dst = sorted(f for f in os.listdir("../../Desktop/ForgettingPitchPublic/manuscript") if "." not in f)[-1]
    dst = pj("../../Desktop/ForgettingPitchPublic/manuscript", dst)
    s = open(pj(dst, "manuscript.tex")).read()

    src = pj(
        "../../Desktop/ForgettingPitchPublic/models",
        "prescriptive",
        exp,
        prior_type,
        "".join(drop) if drop else "none",
    )
    s = s.replace("loostable", open(pj(src, "comparison.tex")).read())
    stats = json.load(open(pj(src, "stats.json")))
    for k, v in stats.items():
        if v > 10000:
            v = ("%.2E" % Decimal(v)).replace("E", r"\times10^{")
            v = v.replace("+", "").replace("{0", "{") + "}"
        else:
            v = f"{v:0.3f}"
        print(k, v)
        s = s.replace(k, v)

    src = pj(
        "../../Desktop/ForgettingPitchPublic/models",
        "agnostic",
        exp,
        "none" if not restrict else restrict,
        "".join(drop) if drop else "none",
    )
    shutil.copy(pj(src, "psyfun.eps"), pj(dst, "fig2.eps"))
    shutil.copy(pj(src, "posteriors.eps"), pj(dst, "fig3.eps"))
    stats = json.load(open(pj(src, "stats.json")))
    for k, v in stats.items():
        if v > 10000:
            v = ("%.2E" % Decimal(v)).replace("E", r"\times10^{")
            v = v.replace("+", "").replace("{0", "{") + "}"
        else:
            v = f"{v:0.3f}"
        print(k, v)
        s = s.replace(k, v)
    stats = json.load(open(pj(src, "bfs.json")))
    for k, v in stats.items():
        if v > 10000:
            v = ("%.2E" % Decimal(v)).replace("E", r"\times10^{")
            v = v.replace("+", "").replace("{0", "{") + "}"
        else:
            v = f"{v:0.3f}"
        print(k, v)
        s = s.replace(f"${k}$", f"${v}$")

    s = s.replace("varstable", open(pj(src, "vars.tex")).read())
    s = s.replace(r"%\begin{figure*}[ht]", r"\begin{figure*}[ht]")
    s = s.replace(r"\toprule", "\hline\hline").replace(r"\bottomrule", "\hline\hline")
    s = s.replace("\midrule", "\hline")

    open(pj(dst, "manuscript2.tex"), "w").write(s)

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def fatigue():
    data = pd.read_csv("data.csv")
    data["block"] = 0
    block = 0
    for ix, row in data.iterrows():
        if ix > 0:
            previous = data.loc[ix - 1]
            if row.listener != previous.listener or row.isi != previous.isi:
                block += 1
            data.loc[ix, "block"] = block
    for _, df in data.groupby("block"):
        for chunk in chunks(df.index, 65):
            block += 1
            data.loc[chunk, "block"] = block
    data["last"] = False
    for _, df in data.groupby("block"):
        data.loc[df.iloc[32:].index, "last"] = True
    data.response.replace({0: -1}, inplace=True)
    data = data[data.delta != 0]
    data.delta = data.delta.clip(-0.0001, 0.0001) * 10000
    data["correct"] = data.delta == data.response
    data.reset_index(inplace=True, drop=True)
    data = pd.pivot_table(data, index=["listener", "isi", "last"], values="correct", aggfunc=np.mean)
    data.reset_index(inplace=True)
    sns.catplot(
        data=data, kind="bar",
        x="isi", y="correct", hue="last",
    )
    plt.savefig("plot.png")


def main():
    """Runs everything."""
    # cartoon()
    fit_agnostic_models()
    # fit_prescriptive_models()
    # populate_manuscript("both", [], [], "multivariate")
    # cartoon2()
    # fatigue()

if __name__ == "__main__":
    main()
