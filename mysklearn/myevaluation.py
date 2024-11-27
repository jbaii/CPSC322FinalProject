from mysklearn import myutils

import numpy as np

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
            Use random_state to seed your random number generator
                you can use the math module or use numpy for your generator
                choose one and consistently use that generator throughout your code
        shuffle(bool): whether or not to randomize the order of the instances before splitting
            Shuffle the rows in X and y before splitting and be sure to maintain the parallel order of X and y!!

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)

    Note:
        Loosely based on sklearn's train_test_split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        np.random.seed(random_state)

    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if shuffle:
        np.random.shuffle(indices)
        indices=indices[::-1]

    if isinstance(test_size, float):
        test_size = int(np.ceil(n_samples * test_size))

    train_indices = indices[:-test_size]
    test_indices = indices[-test_size:]
    # print("train_indices:", train_indices)
    # print("test_indices:", test_indices)
    # print("test_size:", test_size)
    X_train = [X[i] for i in train_indices]
    X_test = [X[i] for i in test_indices]
    y_train = [y[i] for i in train_indices]
    y_test = [y[i] for i in test_indices]
    return X_train, X_test, y_train, y_test

def kfold_split(X, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        The first n_samples % n_splits folds have size n_samples // n_splits + 1,
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    n_samples = len(X)
    indices = np.arange(n_samples)
    
    if random_state is not None:
        np.random.seed(random_state)

    if shuffle:
        np.random.shuffle(indices)

    folds = []
    fold_sizes = [n_samples // n_splits + 1 if i < n_samples % n_splits else n_samples // n_splits for i in range(n_splits)]
    current = 0
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        test_indices = indices[start:stop]
        train_indices = np.concatenate([indices[:start], indices[stop:]])
        folds.append((train_indices.tolist(), test_indices.tolist()))
        current = stop

    return folds

# BONUS function
def stratified_kfold_split(X, y, n_splits=5, random_state=None, shuffle=False):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples).
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X).
            The shape of y is n_samples
        n_splits(int): Number of folds.
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before creating folds

    Returns:
        folds(list of 2-item tuples): The list of folds where each fold is defined as a 2-item tuple
            The first item in the tuple is the list of training set indices for the fold
            The second item in the tuple is the list of testing set indices for the fold

    Notes:
        Loosely based on sklearn's StratifiedKFold split():
            https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if isinstance(X, list):
        X = np.array(X)
    if isinstance(y, list):
        y = np.array(y)
    
    n_samples = len(X)
    indices = np.arange(n_samples)
    if random_state is not None:
        np.random.seed(random_state)
    if shuffle:
        np.random.shuffle(indices)

    y = y[indices]
    unique_classes, y_indices = np.unique(y, return_inverse=True)
    
    # Initialize fold structures
    folds = [[] for _ in range(n_splits)]
    
    # For each class, assign indices to folds
    for class_idx in unique_classes:
        class_indices = np.where(y == class_idx)[0]
        np.random.shuffle(class_indices)
        
        fold_sizes = np.full(n_splits, len(class_indices) // n_splits, dtype=int)
        fold_sizes[:len(class_indices) % n_splits] += 1
        
        start = 0
        for i in range(n_splits):
            stop = start + fold_sizes[i]
            folds[i].extend(class_indices[start:stop])
            start = stop

    # Generate train/test splits
    stratified_folds = []
    for i in range(n_splits):
        test_indices = folds[i]
        train_indices = np.hstack([folds[j] for j in range(n_splits) if j != i]).tolist()
        stratified_folds.append((train_indices, test_indices))

    return stratified_folds


def bootstrap_sample(X, y=None, n_samples=None, random_state=None):
    """Split dataset into bootstrapped training set and out of bag test set.

    Args:
        X(list of list of obj): The list of samples
        y(list of obj): The target y values (parallel to X)
            Default is None (in this case, the calling code only wants to sample X)
        n_samples(int): Number of samples to generate. If left to None (default) this is automatically
            set to the first dimension of X.
        random_state(int): integer used for seeding a random number generator for reproducible results

    Returns:
        X_sample(list of list of obj): The list of samples
        X_out_of_bag(list of list of obj): The list of "out of bag" samples (e.g. left-over samples)
        y_sample(list of obj): The list of target y values sampled (parallel to X_sample)
            None if y is None
        y_out_of_bag(list of obj): The list of target y values "out of bag" (parallel to X_out_of_bag)
            None if y is None
    Notes:
        Loosely based on sklearn's resample():
            https://scikit-learn.org/stable/modules/generated/sklearn.utils.resample.html
        Sample indexes of X with replacement, then build X_sample and X_out_of_bag
            as lists of instances using sampled indexes (use same indexes to build
            y_sample and y_out_of_bag)
    """
    if isinstance(X, list):
        X = np.array(X)
    if y is not None and isinstance(y, list):
        y = np.array(y)

    if n_samples is None:
        n_samples = len(X)

    if random_state is not None:
        np.random.seed(random_state)

    # Sample indices with replacement
    indices = np.random.choice(len(X), size=n_samples, replace=True)
    out_of_bag_indices = np.setdiff1d(np.arange(len(X)), indices, assume_unique=False)

    # Select samples according to indices
    X_sample = X[indices]
    X_out_of_bag = X[out_of_bag_indices]

    if y is not None:
        y_sample = y[indices]
        y_out_of_bag = y[out_of_bag_indices]
        return X_sample, X_out_of_bag, y_sample, y_out_of_bag
    else:
        return X_sample, X_out_of_bag, None, None

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry
            indicates the number of samples with true label being i-th class
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
     # Create a mapping from label to index
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    
    # print(label_to_index)

    # Initialize the confusion matrix with zeros
    matrix = [[0 for _ in labels] for _ in labels]

    # Populate the confusion matrix
    for true, pred in zip(y_true, y_pred):
        # print(label_to_index)
        # print(true)
        true_index = label_to_index[true]
        pred_index = label_to_index[pred]
        matrix[true_index][pred_index] += 1

    return matrix

def accuracy_score(y_true, y_pred, normalize=True):
    """Compute the classification prediction accuracy score.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        normalize(bool): If False, return the number of correctly classified samples.
            Otherwise, return the fraction of correctly classified samples.

    Returns:
        score(float): If normalize == True, return the fraction of correctly classified samples (float),
            else returns the number of correctly classified samples (int).

    Notes:
        Loosely based on sklearn's accuracy_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.accuracy_score.html#sklearn.metrics.accuracy_score
    """
    res = []
    for true, pred in zip(y_true, y_pred):
        if pred == true:
            res.append(1)
        else:
            res.append(0)
    if normalize:
        return sum(res) / len(res)
    else:
        return sum(res)

def binary_precision_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the precision (for binary classification). The precision is the ratio tp / (tp + fp)
        where tp is the number of true positives and fp the number of false positives.
        The precision is intuitively the ability of the classifier not to label as
        positive a sample that is negative. The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        precision(float): Precision of the positive class

    Notes:
        Loosely based on sklearn's precision_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.precision_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == pos_label)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != pos_label and yp == pos_label)

    if tp + fp == 0:
        return 0.0  # To avoid division by zero
    precision = tp / (tp + fp)
    return precision

def binary_recall_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the recall (for binary classification). The recall is the ratio tp / (tp + fn) where tp is
        the number of true positives and fn the number of false negatives.
        The recall is intuitively the ability of the classifier to find all the positive samples.
        The best value is 1 and the worst value is 0.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        recall(float): Recall of the positive class

    Notes:
        Loosely based on sklearn's recall_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.recall_score.html
    """
    if labels is None:
        labels = list(set(y_true))
    if pos_label is None:
        pos_label = labels[0]

    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == yp == pos_label)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == pos_label and yp != pos_label)

    if tp + fn == 0:
        return 0.0  # To avoid division by zero
    recall = tp / (tp + fn)
    return recall

def binary_f1_score(y_true, y_pred, labels=None, pos_label=None):
    """Compute the F1 score (for binary classification), also known as balanced F-score or F-measure.
        The F1 score can be interpreted as a harmonic mean of the precision and recall,
        where an F1 score reaches its best value at 1 and worst score at 0.
        The relative contribution of precision and recall to the F1 score are equal.
        The formula for the F1 score is: F1 = 2 * (precision * recall) / (precision + recall)

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of obj): The list of possible class labels. If None, defaults to
            the unique values in y_true
        pos_label(obj): The class label to report as the "positive" class. If None, defaults
            to the first label in labels

    Returns:
        f1(float): F1 score of the positive class

    Notes:
        Loosely based on sklearn's f1_score():
            https://scikit-learn.org/stable/modules/generated/sklearn.metrics.f1_score.html
    """
    precision = binary_precision_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    recall = binary_recall_score(y_true, y_pred, labels=labels, pos_label=pos_label)
    
    if precision + recall == 0:
        return 0.0  # To avoid division by zero in the F1 calculation
    f1 = 2 * (precision * recall) / (precision + recall)
    return f1

from tabulate import tabulate

def classification_report(y_true, y_pred, labels=None, output_dict=False):
    if labels is None:
        labels = sorted(set(y_true))

    metrics = {}
    for label in labels:
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp == label)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != label and yp == label)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == label and yp != label)
        support = sum(1 for yt in y_true if yt == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1_score,
            'support': support
        }

    total_support = sum(metrics[label]['support'] for label in labels)
    avg_metrics = {
        'precision': sum(metrics[label]['precision'] for label in labels) / len(labels),
        'recall': sum(metrics[label]['recall'] for label in labels) / len(labels),
        'f1-score': sum(metrics[label]['f1-score'] for label in labels) / len(labels),
        'support': total_support
    }
    metrics['macro avg'] = avg_metrics

    weighted_avg_metrics = {
        'precision': sum(metrics[label]['precision'] * metrics[label]['support'] for label in labels) / total_support,
        'recall': sum(metrics[label]['recall'] * metrics[label]['support'] for label in labels) / total_support,
        'f1-score': sum(metrics[label]['f1-score'] * metrics[label]['support'] for label in labels) / total_support,
        'support': total_support
    }
    metrics['weighted avg'] = weighted_avg_metrics

    if output_dict:
        return metrics

    table = [[label, metrics[label]['precision'], metrics[label]['recall'],
              metrics[label]['f1-score'], metrics[label]['support']] for label in labels]
    table.append(['macro avg', avg_metrics['precision'], avg_metrics['recall'], avg_metrics['f1-score'], avg_metrics['support']])
    table.append(['weighted avg', weighted_avg_metrics['precision'], weighted_avg_metrics['recall'], weighted_avg_metrics['f1-score'], weighted_avg_metrics['support']])

    return tabulate(table, headers=['Label', 'Precision', 'Recall', 'F1-Score', 'Support'], floatfmt=".2f")