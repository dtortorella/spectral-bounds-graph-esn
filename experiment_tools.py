import itertools
import random


def cross_split(samples, folds, shuffle=True):
    """
    K-fold split

    :param samples: Number of samples or list of indices
    :param folds: Number of folds
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of pairs training set, validation set
    """
    if type(samples) is int:
        samples = list(range(samples))
    fold_size = int(len(samples) / folds)
    if shuffle:
        random.shuffle(samples)
    splits = [samples[i * fold_size:(i + 1) * fold_size] for i in range(folds - 1)]
    splits.append(samples[(folds - 1) * fold_size:])
    return [(_cat(_all_but(splits, fold)), splits[fold]) for fold in range(folds)]


def double_cross_split(samples, outer_folds, inner_folds, shuffle=True):
    """
    Double k-fold split

    :param samples: Number of samples or list of indices
    :param outer_folds: Number of outer folds
    :param inner_folds: Number of inner folds
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of triples of cross-validation splits, training set, test set
    """
    outer_splits = cross_split(samples, outer_folds, shuffle)
    return [(cross_split(selection_set, inner_folds, shuffle), selection_set, test_set) for selection_set, test_set in outer_splits]


def holdout_split(samples, ratio, shuffle=True):
    """
    Hold-out split

    :param samples: Number of samples or list of indices
    :param ratio: Ratio of samples in held-out set
    :param shuffle: Whether to shuffle samples (default true)
    :return: Training set, validation (held out) set
    """
    if type(samples) is int:
        samples = list(range(samples))
    if shuffle:
        random.shuffle(samples)
    return samples[int(ratio * len(samples)):], samples[:int(ratio * len(samples))]


def cross_holdout_split(samples, outer_folds, inner_ratio, shuffle=True):
    """
    K-fold outer split with inner hold-out

    :param samples: Number of samples or list of indices
    :param outer_folds: Number of outer folds
    :param inner_ratio: Ratio of samples in inner held-out sets
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of triples training set, validation set, test set
    """
    outer_splits = cross_split(samples, outer_folds, shuffle)
    return [holdout_split(selection_set, inner_ratio, shuffle) + (test_set,) for selection_set, test_set in outer_splits]


def cross_split_stratified(samples, folds, shuffle=True):
    """
    K-fold split with stratification

    :param samples: List of samples classes
    :param folds: Number of folds
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of pairs training set, validation set
    """
    classes = max(samples) + 1
    samples = list(enumerate(samples))
    if shuffle:
        random.shuffle(samples)
    classified_samples = [[index for index, label in samples if label == c] for c in range(classes)]
    classified_splits = [cross_split(class_samples, folds, shuffle) for class_samples in classified_samples]
    splits = [tuple(_cat(class_split[i][j] for class_split in classified_splits) for j in range(2)) for i in range(folds)]
    return splits


def double_cross_split_stratified(samples, outer_folds, inner_folds, shuffle=True):
    """
    Double k-fold split with stratification

    :param samples: List of samples classes
    :param outer_folds: Number of outer folds
    :param inner_folds: Number of inner folds
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of triples of cross-validation splits, training set, test set
    """
    classes = max(samples) + 1
    samples = list(enumerate(samples))
    if shuffle:
        random.shuffle(samples)
    classified_samples = [[index for index, label in samples if label == c] for c in range(classes)]
    classified_splits = [double_cross_split(class_samples, outer_folds, inner_folds, shuffle) for class_samples in classified_samples]
    splits = [([tuple(_cat(class_split[outer][0][i][j] for class_split in classified_splits) for j in range(2)) for i in range(inner_folds)], _cat(class_split[outer][1] for class_split in classified_splits), _cat(class_split[outer][2] for class_split in classified_splits)) for outer in range(outer_folds)]
    return splits


def double_cross_split_stratified_with_selection_dic(samples, selection_dic, inner_folds, shuffle=True):
    """
    Double k-fold split with stratification, by selection splits dictionary

    :param samples: List of samples classes
    :param selection_dic: List of train/test outer splits indexes
    :param inner_folds: Number of inner folds
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of triples of cross-validation splits, training set, test set
    """
    classes = max(samples) + 1
    test_splits = [split['test'] for split in selection_dic]
    selection_splits = [split['train'] for split in selection_dic]
    cross_selection_splits = []
    for outer_split in selection_splits:
        inner_samples = [(index, samples[index]) for index in outer_split]
        classified_samples = [[index for index, label in inner_samples if label == c] for c in range(classes)]
        classified_splits = [cross_split(class_samples, inner_folds, shuffle) for class_samples in classified_samples]
        inner_splits = [tuple(_cat(class_split[i][j] for class_split in classified_splits) for j in range(2)) for i in range(inner_folds)]
        cross_selection_splits.append(inner_splits)
    return list(zip(cross_selection_splits, selection_splits, test_splits))


def holdout_split_stratified(samples, ratio, shuffle=True):
    """
    Hold-out split with stratification

    :param samples: List of samples classes
    :param ratio: Ratio of samples in held-out set
    :param shuffle: Whether to shuffle samples (default true)
    :return: Training set, validation (held out) set
    """
    classes = max(samples) + 1
    samples = list(enumerate(samples))
    if shuffle:
        random.shuffle(samples)
    classified_samples = [[index for index, label in samples if label == c] for c in range(classes)]
    classified_splits = [holdout_split(class_samples, ratio, shuffle) for class_samples in classified_samples]
    splits = tuple(_cat(class_split[i] for class_split in classified_splits) for i in range(2))
    return splits


def cross_holdout_split_stratified(samples, outer_folds, inner_ratio, shuffle=True):
    """
    K-fold outer split with inner hold-out

    :param samples: List of samples classes
    :param outer_folds: Number of outer folds
    :param inner_ratio: Ratio of samples in inner held-out sets
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of triples training set, validation set, test set
    """
    classes = max(samples) + 1
    samples = list(enumerate(samples))
    if shuffle:
        random.shuffle(samples)
    classified_samples = [[index for index, label in samples if label == c] for c in range(classes)]
    classified_splits = [cross_holdout_split(class_samples, outer_folds, inner_ratio, shuffle) for class_samples in classified_samples]
    splits = [tuple(_cat(class_split[outer][inner] for class_split in classified_splits) for inner in range(3)) for outer in range(outer_folds)]
    return splits


def cross_holdout_split_stratified_with_selection_dic(samples, selection_dic, inner_ratio, shuffle=True):
    """
    K-fold outer split with inner hold-out with stratification, by selection splits dictionary

    :param samples: List of samples classes
    :param selection_dic: List of train/test outer splits indexes
    :param inner_ratio: Ratio of samples in inner held-out sets
    :param shuffle: Whether to shuffle samples (default true)
    :return: List of triples of cross-validation splits, training set, test set
    """
    classes = max(samples) + 1
    test_splits = [split['test'] for split in selection_dic]
    selection_splits = [split['train'] for split in selection_dic]
    final_splits = []
    for outer_split, test_split in zip(selection_splits, test_splits):
        inner_samples = [(index, samples[index]) for index in outer_split]
        classified_samples = [[index for index, label in inner_samples if label == c] for c in range(classes)]
        classified_splits = [holdout_split(class_samples, inner_ratio, shuffle) for class_samples in classified_samples]
        inner_splits = tuple(_cat(class_split[j] for class_split in classified_splits) for j in range(2))
        final_splits.append(inner_splits + (test_split,))
    return final_splits


def split_dataset(dataset, *splits):
    """
    Split dataset according to partitions

    :param dataset: Dataset, indexed
    :param splits: List of partitions
    :return: List of sub-datasets
    """
    return [[dataset[i] for i in split] for split in splits]


def _cat(lists):
    """
    Concatenate a list of lists together

    :param lists: List of lists
    :return: Concatenated list
    """
    return list(itertools.chain(*lists))


def _all_but(list, but):
    """
    Exclude an element in a list

    :param list: List
    :param but: Excluded element index
    :return: Purged list
    """
    return [elem for index, elem in enumerate(list) if index != but]
