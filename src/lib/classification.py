import numpy as np
from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, ShuffleSplit, train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import OneHotEncoder, normalize

def fit_ind_logistic_regression(X, y, train_mask, data_random_seed=1, repeat=1, return_mistakes=False):
    # transform targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # in the inductive setting, we have a fixed test mask
    train_mask = train_mask.cpu().numpy()
    test_mask = ~train_mask
    X_base, y_base = X[train_mask], y[train_mask]
    X_test, y_test = X[test_mask], y[test_mask]

    # normalize x
    X = normalize(X, norm='l2')
    # set random state to ensure a consistent split
    rng = np.random.RandomState(data_random_seed)

    test_accs = []
    val_accs = []

    for _ in range(repeat):
        X_train, X_val, y_train, y_val = train_test_split(
            X_base, y_base, test_size=0.1, random_state=rng
        )

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear', max_iter=200)
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg),
            param_grid=dict(estimator__C=c),
            n_jobs=5,
            cv=cv,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        y_pred_val = clf.predict_proba(X_val)
        y_pred_val = np.argmax(y_pred_val, axis=1)
        y_pred_val = one_hot_encoder.transform(y_pred_val.reshape(-1, 1)).astype(bool)

        y_pred_test = clf.predict_proba(X_test)
        y_pred_test = np.argmax(y_pred_test, axis=1)
        y_pred_test = one_hot_encoder.transform(y_pred_test.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred_test)
        val_acc = metrics.accuracy_score(y_val, y_pred_val)

        test_accs.append(test_acc)
        val_accs.append(val_acc)

    if return_mistakes:
        return test_accs, val_accs, mistakes
    return test_accs, val_accs


def fit_logistic_regression(X, y, data_random_seed=1, repeat=1, return_mistakes=False):
    """Fit a logistic regression model to the data for node classification.
    This is based off of the official BGRL implementation:
    https://github.com/nerdslab/bgrl/blob/dec99f8c605e3c4ae2ece57f3fa1d41f350d11a9/bgrl/logistic_regression_eval.py#L9
    """
    # transform targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)

    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # normalize x
    X = normalize(X, norm='l2')
    # set random state to ensure a consistent split


    rng = np.random.RandomState(data_random_seed)

    test_accs = []
    val_accs = []
    mistakes = []

    for _ in range(repeat):
        # different random split after each repeat
        idxs = np.arange(y.shape[0])
        X_train, X2, y_train, y2, idx_train, idx2 = train_test_split(
            X, y, idxs, test_size=0.8, random_state=rng
        )
        # we follow AFGRL and split into a validation set as well
        X_val, X_test, y_val, y_test, idx_val, idx_test = train_test_split(
            X2, y2, idx2, test_size=0.5, random_state=rng
        )

        # grid search with one-vs-rest classifiers
        logreg = LogisticRegression(solver='liblinear', max_iter=200)
        c = 2.0 ** np.arange(-10, 11)
        cv = ShuffleSplit(n_splits=5, test_size=0.5)
        clf = GridSearchCV(
            estimator=OneVsRestClassifier(logreg),
            param_grid=dict(estimator__C=c),
            n_jobs=5,
            cv=cv,
            verbose=0,
        )
        clf.fit(X_train, y_train)

        y_pred_val = clf.predict_proba(X_val)
        y_pred_val = np.argmax(y_pred_val, axis=1)
        y_pred_val = one_hot_encoder.transform(y_pred_val.reshape(-1, 1)).astype(bool)

        y_pred_test = clf.predict_proba(X_test)
        y_pred_test = np.argmax(y_pred_test, axis=1)
        y_pred_test = one_hot_encoder.transform(y_pred_test.reshape(-1, 1)).astype(bool)

        test_acc = metrics.accuracy_score(y_test, y_pred_test)
        val_acc = metrics.accuracy_score(y_val, y_pred_val)

        test_wrong_idxs = (y_test != y_pred_test).any(axis=1)
        # print(test_wrong_idxs)
        wrong_idxs = idx_test[test_wrong_idxs]

        test_accs.append(test_acc)
        val_accs.append(val_acc)
        mistakes.append(wrong_idxs)
    if return_mistakes:
        return test_accs, val_accs, mistakes
    return test_accs, val_accs


def do_classification_eval(dataset, embeddings):
    data = dataset[0]
    X = embeddings.weight.cpu().numpy()
    y = data.y.cpu().numpy()
    accs = fit_logistic_regression(X, y)
    return np.mean(accs)


def fit_logistic_regression_preset_splits(X, y, train_masks, val_masks, test_mask):
    # transfrom targets to one-hot vector
    one_hot_encoder = OneHotEncoder(categories='auto', sparse=False)
    y = one_hot_encoder.fit_transform(y.reshape(-1, 1)).astype(bool)

    # normalize x
    X = normalize(X, norm='l2')

    accuracies = []
    val_accuracies = []
    for split_id in range(train_masks.shape[1]):
        # get train/val/test masks
        train_mask, val_mask = train_masks[:, split_id], val_masks[:, split_id]

        # make custom cv
        X_train, y_train = X[train_mask], y[train_mask]
        X_val, y_val = X[val_mask], y[val_mask]
        X_test, y_test = X[test_mask], y[test_mask]

        # grid search with one-vs-rest classifiers
        best_test_acc, best_acc = 0, 0
        for c in 2.0 ** np.arange(-10, 11):
            clf = OneVsRestClassifier(LogisticRegression(solver='liblinear', C=c))
            clf.fit(X_train, y_train)

            y_pred = clf.predict_proba(X_val)
            y_pred = np.argmax(y_pred, axis=1)
            y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
            val_acc = metrics.accuracy_score(y_val, y_pred)
            if val_acc > best_acc:
                best_acc = val_acc
                y_pred = clf.predict_proba(X_test)
                y_pred = np.argmax(y_pred, axis=1)
                y_pred = one_hot_encoder.transform(y_pred.reshape(-1, 1)).astype(bool)
                best_test_acc = metrics.accuracy_score(y_test, y_pred)

        accuracies.append(best_test_acc)
        val_accuracies.append(best_acc)
    # print(np.mean(accuracies))
    return accuracies, val_accuracies