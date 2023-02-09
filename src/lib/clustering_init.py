import numpy as np
from sklearn.utils import check_random_state
from sklearn.utils.extmath import stable_cumsum

def kpp_init(D, n_clusters, random_state_=None, n_local_trials=None):
    """Init n_clusters seeds with a method similar to k-means++

    Parameters
    -----------
    D : array, shape (n_samples, n_samples)
        The distance matrix we will use to select medoid indices.

    n_clusters : integer
        The number of seeds to choose

    random_state : RandomState
        The generator used to initialize the centers.

    n_local_trials : integer, optional
        The number of seeding trials for each center (except the first),
        of which the one reducing inertia the most is greedily chosen.
        Set to None to make the number of trials depend logarithmically
        on the number of seeds (2+log(k)); this is the default.

    Notes
    -----
    Selects initial cluster centers for k-medoid clustering in a smart way
    to speed up convergence. see: Arthur, D. and Vassilvitskii, S.
    "k-means++: the advantages of careful seeding". ACM-SIAM symposium
    on Discrete algorithms. 2007

    Version ported from http://www.stanford.edu/~darthur/kMeansppTest.zip,
    which is the implementation used in the aforementioned paper.
    """
    n_samples, _ = D.shape
    random_state_ = check_random_state(random_state_)

    centers = np.empty(n_clusters, dtype=int)

    # Set the number of local seeding trials if none is given
    if n_local_trials is None:
        # This is what Arthur/Vassilvitskii tried, but did not report
        # specific results for other than mentioning in the conclusion
        # that it helped.
        n_local_trials = 2 + int(np.log(n_clusters))

    center_id = random_state_.randint(n_samples)
    centers[0] = center_id

    # Initialize list of closest distances and calculate current potential
    closest_dist_sq = D[centers[0], :]**2
    current_pot = closest_dist_sq.sum()

    # pick the remaining n_clusters-1 points
    for cluster_index in range(1, n_clusters):
        rand_vals = (random_state_.random_sample(n_local_trials) * current_pot)
        candidate_ids = np.searchsorted(stable_cumsum(closest_dist_sq), rand_vals)

        # Compute distances to center candidates
        distance_to_candidates = D[candidate_ids, :]**2

        # Decide which candidate is the best
        best_candidate = None
        best_pot = None
        best_dist_sq = None
        for trial in range(n_local_trials):
            # Compute potential when including center candidate
            new_dist_sq = np.minimum(closest_dist_sq, distance_to_candidates[trial])
            new_pot = new_dist_sq.sum()

            # Store result if it is the best local trial so far
            if (best_candidate is None) or (new_pot < best_pot):
                best_candidate = candidate_ids[trial]
                best_pot = new_pot
                best_dist_sq = new_dist_sq

        centers[cluster_index] = best_candidate
        current_pot = best_pot
        closest_dist_sq = best_dist_sq

    return centers
