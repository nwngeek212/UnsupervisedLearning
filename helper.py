import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import sklearn
import pickle

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_random_state
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.neighbors import KernelDensity

from scipy import stats, integrate

def plot_pairgrid(df):
    """
    Uses seaborn.PairGrid to visualize the attributes related to the six physical characteristics.
    Diagonal plots are histograms. The off-diagonal plots are scatter plots.

    Parameters
    ----------
    df: A pandas.DataFrame. Comes from importing delta.csv.

    Returns
    -------
    A seaborn.axisgrid.PairGrid instance.
    """

    sns.set(style="white")

    # Make pair plot
    g = sns.PairGrid(
        df[['Cruising Speed (mph)', 'Range (miles)', 'Engines', 'Wingspan (ft)', 'Tail Height (ft)', 'Length (ft)']])
    g = g.map_diag(plt.hist)
    g = g.map_offdiag(plt.scatter)

    return g


def fit_pca(df, n_components):
    """
    Uses sklearn.decomposition.PCA to fit a PCA model on "df".

    Parameters
    ----------
    df: A pandas.DataFrame. Comes from delta.csv.
    n_components: An int. Number of principal components to keep.

    Returns
    -------
    An sklearn.decomposition.pca.PCA instance.
    """

    pca = PCA(n_components)
    pca.fit(df)

    return pca


def plot_naive_variance(pca):
    """
    Plots the variance explained by each of the principal components.
    Attributes are not scaled, hence a naive approach.

    Parameters
    ----------
    pca: An sklearn.decomposition.pca.PCA instance.

    Returns
    -------
    A matplotlib.Axes instance.
    """

    # Make the plot
    sns.set(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Decorate the plot
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Variance vs Component')

    ax.set_xlim(0, 3)

    ax.plot(pca.explained_variance_ratio_)

    return ax


def standardize(df):
    """
    Uses sklearn.preprocessing.StandardScaler to make each features look like
    a Gaussian with zero mean and unit variance.

    Parameters
    ----------
    df: A pandas.DataFrame

    Returns
    -------
    A numpy array.
    """

    scaled = StandardScaler(copy=True, with_mean=True, with_std=True).fit_transform(df)

    return scaled


def plot_scaled_variance(pca):
    """
    Plots the variance explained by each of the principal components.
    Features are scaled with sklearn.StandardScaler.

    Parameters
    ----------
    pca: An sklearn.decomposition.pca.PCA instance.

    Returns
    -------
    A matplotlib.Axes instance.
    """

    # Make the plot
    sns.set(style="ticks", font_scale=2.0)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Decorate the plot
    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Explained Variance')
    ax.set_title('Variance vs Component')

    ax.set_xlim(0, 10)

    ax.plot(pca.explained_variance_ratio_)

    return ax


def reduce(pca, array):
    """
    Applies the `pca` model on array.

    Parameters
    ----------
    pca: An sklearn.decomposition.PCA instance.

    Returns
    -------
    A Numpy array
    """

    reduced = pca.fit_transform(array)

    return reduced


def cluster(array, random_state, n_clusters=4):
    """
    Fits and predicts k-means clustering on "array"

    Parameters
    ----------
    array: A numpy array
    random_state: Random seed, e.g. check_random_state(0)
    n_clusters: The number of clusters. Default: 4

    Returns
    -------
    A tuple (sklearn.KMeans, np.ndarray)
    """

    k_means = KMeans(n_clusters, random_state=random_state)

    # We fit our data to assign classes
    model = k_means.fit(array)
    clusters = k_means.predict(array)

    return model, clusters


def plot_inertia(array, start=1, end=10):
    """
    Increase the number of clusters from "start" to "end" (inclusive).
    Finds the inertia of k-means clustering for different k.
    Plots inertia as a function of the number of clusters.

    Parameters
    ----------
    array: A numpy array.
    start: An int. Default: 1
    end: An int. Default: 10

    Returns
    -------
    A matplotlib.Axes instance.
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel('Clusters')
    ax.set_ylabel('Inertia')
    ax.set_title('Inertia v No. Clusters')

    ax.set_xlim(start, end)

    x = list()
    y = list()

    for i in range(start, end + 1):
        k = KMeans(n_clusters=i, random_state=check_random_state(0))
        model = k.fit(array)
        x.append(i)
        y.append(model.inertia_)

    ax.plot(x, y)
    return ax


def plot_pair(reduced, clusters):
    """
    Uses seaborn.PairGrid to visualize the data distribution
    when axes are the first four principal components.
    Diagonal plots are histograms. The off-diagonal plots are scatter plots.

    Parameters
    ----------
    reduced: A numpy array. Comes from importing delta_reduced.npy

    Returns
    -------
    A seaborn.axisgrid.PairGrid instance.
    """

    sns.set(style="white")

    # Make pair plot

    df = pd.DataFrame(reduced)

    df = df.ix[:, 0:3]
    df['Clusters'] = clusters

    g = sns.pairplot(df, hue='Clusters', hue_order=[0, 1, 2, 3], vars=[0, 1, 2, 3])
    g.map_diag(plt.hist)
    g.map_offdiag(plt.scatter)

    return g


def plot_rugplot(df, column='AirTime', jitter=0.0, seed=0):
    """
    Plots a rug plot.

    Parameters
    ----------
    df: A pandas.DataFrame
    column: The column to use in "df"
    jitter: An int or float. Default: 0.
            If jitter > 0, uses numpy.random.normal() to draw
            random samples from a normal distribution with zero mean
            and standard deviatation equal to "jitter".
    seed: An int. Used by numpy.random.seed().

    Returns
    -------
    A matplotlib.axes.Axes
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(column)
    ax.set_ylim(0, 1)
    rand = 0
    if (jitter > 0):
        np.random.seed(seed)
        rand = np.random.normal(0, jitter, len(df[column]))
    x = df[column] + rand

    sns.rugplot(x, height=0.5, ax=ax)

    return ax


def plot_histogram(df, bins, column='AirTime', normed=False):
    """
    Plots a histogram.

    Parameters
    ----------
    df: A pandas.DataFrame
    column: The column to use in "df"
    normed: If true, the integral of the histogram will sum to 1
            (i.e. normalized) to form a probability density.

    Returns
    -------
    A matplotlib.axes.Axes
    """

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.set_xlabel(column, fontsize=15)
    ax.hist(np.ravel(df[column]), bins=bins, normed=normed, alpha=0.5, color=sns.xkcd_rgb["denim blue"])
    sns.despine(ax=ax, offset=5, trim=True)

    return ax


def plot_distplot(df, bins, column='AirTime'):
    """
    Plots a "distplot".

    Parameters
    ----------
    df: A pandas.DataFrame
    bins: The number of bins
    column: The column to use in "df"

    Returns
    -------
    A matplotlib.axes.Axes
    """

    ax = sns.distplot(np.ravel(df[column]), kde=True, rug=True, bins=bins)
    ax.set_title("Distplot")
    ax.set_xlabel(column)
    ax.set_ylabel("counts")
    sns.despine(ax=ax, offset=5, trim=True)

    return ax


def get_silverman_bandwidth(df, column='AirTime'):
    """
    Calculates bandwidth for KDE using Silverman's rule of thumb.

    Parameters
    ----------
    df: A pandas.DataFrame
    column: The column to use in "df"

    Returns
    -------
    A float
    """

    bw = 1.06 * np.std(df[column]) * len(df[column]) ** (-1.0 / 5.0)

    return bw


def get_kernels(df, support, column='AirTime'):
    """
    Generates Gaussian kernels.

    Parameters
    ----------
    df: A pandas.DataFrame.
    support: Input data points for the probabilit density function.
    column: The column that will be used in "df"

    Returns
    -------
    A 2-d numpy array
    """

    bw = get_silverman_bandwidth(df, column)

    kernel = stats.norm(df, bw).pdf(support)

    return kernel


def normalize_kernels(support, kernels):
    """
    Sums up the individual kernels and normalizes by total area.

    Parameters
    ----------
    support: A 1-d numpy array.
             Input data points for the probabilit density function.
    kernels: A 2-d numpy array.
             Kernels generated from "get_kernels()"

    Returns
    -------
    A 1-d numpy array
    """

    density = np.sum(kernels, axis=0)
    density /= integrate.trapz(density, support)

    return density


def plot_scipy_kde(df, support, bins=50):
    """
    Plots a KDE (using scipy functions) over a histogram.

    Parameters
    ----------
    df: A pandas.DataFrame
    support: A 1-d numpy array.
             Input data points for the probabilit density function.

    Returns
    -------
    A matplotlib.axes.Axes instance.
    """

    density = normalize_kernels(support, get_kernels(df, support))

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(support, density)
    ax.set_ylim(0.0, 0.15)
    ax.hist(np.ravel(df['AirTime']), bins=bins, normed=True, alpha=0.5, color=sns.xkcd_rgb["denim blue"])
    ax.set_title('Histogram and KDE')
    ax.set_ylabel('Density')
    ax.set_xlabel('Air Time')
    sns.despine(ax=ax, offset=5, trim=True)

    return ax


def plot_sklearn_kde(df, support, column='AirTime', bins=50):
    """
    Plots a KDE and a histogram using sklearn.KernelDensity.
    Uses Gaussian kernels.
    The optimal bandwidth is calculated according to Silverman's rule of thumb.

    Parameters
    ----------
    df: A pandas.DataFrame
    support: A 1-d numpy array.
             Input data points for the probabilit density function.

    Returns
    -------
    A matplotlib.axes.Axes instance.
    """

    bw = get_silverman_bandwidth(df, column)

    kde = KernelDensity(kernel='gaussian', bandwidth=bw)

    x = df[column]

    kde.fit(x[:, np.newaxis])
    y = kde.score_samples(support[:, np.newaxis])

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(np.ravel(x), bins=bins, alpha=0.5, color=sns.xkcd_rgb["denim blue"], normed=True)
    ax.plot(support, np.exp(y))
    ax.set_xlabel(column, fontsize=14)
    ax.set_ylabel('Density', fontsize=14)
    ax.set_title('Kernel Density Plot', fontsize=14)
    sns.despine(ax=ax, offset=5, trim=True)

    return ax