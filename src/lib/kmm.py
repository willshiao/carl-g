import copy
import warnings

import numpy as np
import scipy.sparse as sp
import sklearn.cluster
from sklearn.cluster._kmeans import KMeans, _BaseKMeans
from sklearn.cluster._kmeans import _labels_inertia, _check_sample_weight
from sklearn.cluster._k_means_minibatch import _minibatch_update_sparse
from sklearn.cluster._kmeans import _tolerance, row_norms, effective_n_jobs
from sklearn.exceptions import ConvergenceWarning
from sklearn.utils import check_random_state, check_array
from sklearn.utils.validation import _num_samples
from sklearn.utils._joblib import Parallel
from sklearn.utils._joblib import delayed


def _check_normalize_sample_weight(sample_weight, X):
    """Set sample_weight if None, and check for correct dtype"""

    sample_weight_was_none = sample_weight is None

    sample_weight = _check_sample_weight(X, sample_weight)

    if not sample_weight_was_none:
        # normalize the weights to sum up to n_samples
        # an array of 1 (i.e. samples_weight is None) is already normalized
        n_samples = len(sample_weight)
        scale = n_samples / sample_weight.sum()
        sample_weight *= scale
    return sample_weight

def _validate_center_shape(X, n_centers, centers):
    """Check if centers is compatible with X and n_centers"""
    if len(centers) != n_centers:
        raise ValueError('The shape of the initial centers (%s) '
                         'does not match the number of clusters %i'
                         % (centers.shape, n_centers))
    if centers.shape[1] != X.shape[1]:
        raise ValueError(
            "The number of features of the initial centers %s "
            "does not match the number of features of the data %s."
            % (centers.shape[1], X.shape[1]))


def _k_means_minus_minus(
    X,
    sample_weight,
    n_clusters,
    prop_outliers,
    max_iter=300,
    init="k-means++",
    verbose=False,
    x_squared_norms=None,
    random_state=None,
    tol=1e-4,
    precompute_distances=True,
):
    """A single run of k-means, assumes preparation completed prior.

    Parameters
    ----------
    X : array-like of floats, shape (n_samples, n_features)
        The observations to cluster.

    sample_weight : array-like, shape (n_samples,)
        The weights for each observation in X.

    n_clusters : int
        The number of clusters to form as well as the number of
        centroids to generate.

    prop_outliers : float
        What proportion of the training dataset X to treat as outliers, and
        to exclude in each iteration of Lloyd's algorithm.

    max_iter : int, optional, default 300
        Maximum number of iterations of the k-means algorithm to run.

    init : {'k-means++', 'random', or ndarray, or a callable}, optional
        Method for initialization, default to 'k-means++':

        'k-means++' : selects initial cluster centers for k-mean
        clustering in a smart way to speed up convergence. See section
        Notes in k_init for more details.

        'random': choose k observations (rows) at random from data for
        the initial centroids.

        If an ndarray is passed, it should be of shape (k, p) and gives
        the initial centers.

        If a callable is passed, it should take arguments X, k and
        and a random state and return an initialization.

    tol : float, optional
        The relative increment in the results before declaring convergence.

    verbose : boolean, optional
        Verbosity mode

    x_squared_norms : array
        Precomputed x_squared_norms.

    precompute_distances : boolean, default: True
        Precompute distances (faster but takes more memory).

    random_state : int, RandomState instance or None (default)
        Determines random number generation for centroid initialization. Use
        an int to make the randomness deterministic.
        See :term:`Glossary <random_state>`.

    Returns
    -------
    centroid : float ndarray with shape (k, n_features)
        Centroids found at the last iteration of k-means.

    label : integer ndarray with shape (n_samples,)
        label[i] is the code or index of the centroid the
        i'th observation is closest to.

    inertia : float
        The final value of the inertia criterion (sum of squared distances to
        the closest centroid for all observations in the training set).

    n_iter : int
        Number of iterations run.
    """

    n_outliers = int(X.shape[0] * prop_outliers)
    random_state = check_random_state(random_state)

    sample_weight = _check_normalize_sample_weight(sample_weight, X)

    best_labels, best_inertia, best_centers = None, None, None
    base = KMeans(n_clusters=n_clusters, init=init, tol=tol, n_init=1, max_iter=max_iter, verbose=verbose, random_state=random_state)
    # init
    centers = base._init_centroids(
        X, n_clusters, init, random_state=random_state, x_squared_norms=x_squared_norms
    )
    if verbose:
        print("Initialization complete")

    # Allocate memory to store the distances for each sample to its
    # closer center for reallocation in case of ties
    distances = np.zeros(shape=(X.shape[0],), dtype=X.dtype)

    # iterations
    for i in range(max_iter):
        centers_old = centers.copy()

        # labels assignment is also called the E-step of EM
        labels, inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            centers,
            precompute_distances=precompute_distances,
            distances=distances,
        )

        # the "minus-minus" modification step - filter out n_outliers # of
        # datapoints that are farthest from their assigned cluster centers
        X_subset, sample_weight_subset, labels_subset, distances_subset = (
            X,
            sample_weight,
            labels,
            distances,
        )
        if n_outliers > 0:
            outlier_indices = np.argpartition(distances, -n_outliers)[
                -n_outliers:
            ]  # ~20x faster than np.argsort()

            X_subset, sample_weight_subset, labels_subset, distances_subset = (
                np.delete(X, outlier_indices, axis=0),
                np.delete(sample_weight, outlier_indices, axis=0),
                np.delete(labels, outlier_indices, axis=0),
                np.delete(distances, outlier_indices, axis=0),
            )

            # indices_to_refit = np.argsort(distances) < (X.shape[0] - n_outliers)
        # X_subset, sample_weight_subset = X[indices_to_refit], sample_weight[indices_to_refit]

        # computation of the means is also called the M-step of EM
        if sp.issparse(X):
            centers = _minibatch_update_sparse(
                X_subset, sample_weight_subset, labels_subset, n_clusters, distances_subset
            )
        else:
            centers = _minibatch_update_dense(
                X_subset, sample_weight_subset, labels_subset, n_clusters, distances_subset
            )

        if verbose:
            print("Iteration %2d, inertia %.3f" % (i, inertia))

        if best_inertia is None or inertia < best_inertia:
            best_labels = labels.copy()
            best_centers = centers.copy()
            best_inertia = inertia

        center_shift_total = ((centers_old - centers)**2).sum()
        if center_shift_total <= tol:
            if verbose:
                print(
                    "Converged at iteration %d: "
                    "center shift %e within tolerance %e" % (i, center_shift_total, tol)
                )
            break

    if center_shift_total > 0:
        # rerun E-step in case of non-convergence so that predicted labels
        # match cluster centers
        best_labels, best_inertia = _labels_inertia(
            X,
            sample_weight,
            x_squared_norms,
            best_centers,
            precompute_distances=precompute_distances,
            distances=distances,
        )

    return best_labels, best_inertia, best_centers, i + 1


class KMeansMM(KMeans):
    """
    An sklearn compatible version of K-means-- (read "minus minus"), as described in
    this paper: http://pmg.it.usyd.edu.au/outliers.pdf

    Because this class extends the sklearn KMeans function, it inherits any tunable properties of that
    class. This class can also be integrated into any other sklearn infrastructure, such as pipelines
    and hyperparameter search tools.

    Parameters
    ----------

    prop_outliers : float
        What proportion of the training dataset X to treat as outliers, and
        to exclude in each iteration of Lloyd's algorithm.

    **kwargs : optional keyword parameters
        Keyword arguments to the sklearn KMeans class. See documentation at
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html.

    """

    def __init__(self, prop_outliers=0.1, **kwargs):
        self.prop_outliers = prop_outliers
        super().__init__(**kwargs)

    def fit(self, X, y, sample_weight=None):
        """Compute k-means-- clustering.

        Parameters
        ----------
        X : array-like or sparse matrix, shape=(n_samples, n_features)
            Training instances to cluster. It must be noted that the data
            will be converted to C ordering, which will cause a memory
            copy if the given data is not C-contiguous.

        y : Ignored
            Not used, present here for API consistency by convention.

        sample_weight : array-like, shape (n_samples,), optional
            The weights for each observation in X. If None, all observations
            are assigned equal weight (default: None).

        Returns
        -------
        self
            Fitted estimator.
        """
        random_state = check_random_state(self.random_state)

        n_init = self.n_init
        if n_init <= 0:
            raise ValueError(
                "Invalid number of initializations." " n_init=%d must be bigger than zero." % n_init
            )

        if self.max_iter <= 0:
            raise ValueError(
                "Number of iterations should be a positive number,"
                " got %d instead" % self.max_iter
            )

        # avoid forcing order when copy_x=False
        order = "C" if self.copy_x else None
        X = check_array(
            X, accept_sparse="csr", dtype=[np.float64, np.float32], order=order, copy=self.copy_x
        )
        # verify that the number of samples given is larger than k
        if _num_samples(X) < self.n_clusters:
            raise ValueError(
                "n_samples=%d should be >= n_clusters=%d" % (_num_samples(X), self.n_clusters)
            )

        tol = _tolerance(X, self.tol)

        # If the distances are precomputed every job will create a matrix of
        # shape (n_clusters, n_samples). To stop KMeans from eating up memory
        # we only activate this if the created matrix is guaranteed to be
        # under 100MB. 12 million entries consume a little under 100MB if they
        # are of type double.
        precompute_distances = self.precompute_distances
        if precompute_distances == "auto":
            n_samples = X.shape[0]
            precompute_distances = (self.n_clusters * n_samples) < 12e6
        elif isinstance(precompute_distances, bool):
            pass
        else:
            raise ValueError(
                "precompute_distances should be 'auto' or True/False"
                ", but a value of %r was passed" % precompute_distances
            )

        # Validate init array
        init = self.init
        if hasattr(init, "__array__"):
            init = check_array(init, dtype=X.dtype.type, copy=True)
            _validate_center_shape(X, self.n_clusters, init)

            if n_init != 1:
                warnings.warn(
                    "Explicit initial center position passed: "
                    "performing only one init in k-means instead of n_init=%d" % n_init,
                    RuntimeWarning,
                    stacklevel=2,
                )
                n_init = 1

        # subtract of mean of x for more accurate distance computations
        if not sp.issparse(X):
            X_mean = X.mean(axis=0)
            # The copy was already done above
            X -= X_mean

            if hasattr(init, "__array__"):
                init -= X_mean

        # precompute squared norms of data points
        x_squared_norms = row_norms(X, squared=True)

        best_labels, best_inertia, best_centers = None, None, None

        kmeans_single = _k_means_minus_minus

        seeds = random_state.randint(np.iinfo(np.int32).max, size=n_init)
        if effective_n_jobs(self.n_jobs) == 1:
            # For a single thread, less memory is needed if we just store one
            # set of the best results (as opposed to one set per run per
            # thread).
            for seed in seeds:
                # run a k-means once
                labels, inertia, centers, n_iter_ = kmeans_single(
                    X,
                    sample_weight,
                    self.n_clusters,
                    self.prop_outliers,
                    max_iter=self.max_iter,
                    init=init,
                    verbose=self.verbose,
                    precompute_distances=precompute_distances,
                    tol=tol,
                    x_squared_norms=x_squared_norms,
                    random_state=seed,
                )
                # determine if these results are the best so far
                if best_inertia is None or inertia < best_inertia:
                    best_labels = labels.copy()
                    best_centers = centers.copy()
                    best_inertia = inertia
                    best_n_iter = n_iter_
        else:
            # parallelisation of k-means runs
            results = Parallel(n_jobs=self.n_jobs, verbose=0)(
                delayed(kmeans_single)(
                    X,
                    sample_weight,
                    self.n_clusters,
                    self.prop_outliers,
                    max_iter=self.max_iter,
                    init=init,
                    verbose=self.verbose,
                    tol=tol,
                    precompute_distances=precompute_distances,
                    x_squared_norms=x_squared_norms,
                    # Change seed to ensure variety
                    random_state=seed,
                )
                for seed in seeds
            )
            # Get results with the lowest inertia
            labels, inertia, centers, n_iters = zip(*results)
            best = np.argmin(inertia)
            best_labels = labels[best]
            best_inertia = inertia[best]
            best_centers = centers[best]
            best_n_iter = n_iters[best]

        if not sp.issparse(X):
            if not self.copy_x:
                X += X_mean
            best_centers += X_mean

        distinct_clusters = len(set(best_labels))
        if distinct_clusters < self.n_clusters:
            warnings.warn(
                "Number of distinct clusters ({}) found smaller than "
                "n_clusters ({}). Possibly due to duplicate points "
                "in X.".format(distinct_clusters, self.n_clusters),
                ConvergenceWarning,
                stacklevel=2,
            )

        self.cluster_centers_ = best_centers
        self.labels_ = best_labels
        self.inertia_ = best_inertia
        self.n_iter_ = best_n_iter
        return self

    def fit_transform(self, X, y=None, **fit_params):
        """Fit to data, then transform it.

        This is a modified version of the fit_transform function from sklearn.Base.TransformerMixin.
        It is explicitly written out here rather than inherted from KMeans, because the KMeans
        fit_transform() function was not meant to be used within supervised contexts.
        Inheriting from TransformerMixin and forcing a call to that class' fit_transform function
        also did not solve the problem.

        Fits transformer to X and y with optional parameters fit_params
        and returns a transformed version of X.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            Training set.

        y : numpy array of shape [n_samples]
            Target values.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        """
        # fit method of arity 2 (supervised transformation)
        return self.fit(X, y, **fit_params).transform(X)

    def get_params(self, deep=True):
        """Overrides superclass get_params() to allow superclass hyperparameters
          to be returned as well.

          This allows for tuning any parameters from the parent KMeans class
          without having to list each parameter in the KMeansMM __init__() function.
          Taken from https://stackoverflow.com/questions/51430484/how-to-subclass-a-vectorizer-in-scikit-learn-without-repeating-all-parameters-in.

        Parameters
        ----------
        deep : boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """

        params = super().get_params(deep)
        cp = copy.copy(self)
        cp.__class__ = KMeans
        params.update(KMeans.get_params(cp, deep))
        return params

    def predict(self, X, mark_outliers=True):
        """Labels input with closest cluster each sample in X belongs to, unless a sample is
        marked as an outlier. Each time .predict() is run on a dataset X, the closest integer to
        X.shape[0] * self.prop_outliers are marked as outliers. In these cases, the samples are
        assigned a label of -1 rather than a cluster label.

        This function runs the predict() function of sklearn.cluster.KMeans internally.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            New data to predict.

        Returns
        -------
        labels : array, shape [n_samples,]
            Index of the cluster each sample belongs to, or -1 if a sample does not
            belong in any of the clusters
        """
        labels = super().predict(X)

        if mark_outliers:
            n_outliers = int(X.shape[0] * self.prop_outliers)
            cluster_center_matrix = self.cluster_centers_[labels, :]
            outliers = np.argpartition(np.sum((X - cluster_center_matrix) ** 2, axis=1), -n_outliers)[-n_outliers:]
            labels[outliers] = -1

        return labels.reshape(-1, 1)
