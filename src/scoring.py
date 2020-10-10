"""
A library of functions for scoring models.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import logging

import numpy as np
import sklearn

# Possible scoring methods that may be used for hyperparameter tuning.
SCORING_METHODS = ('accuracy',
                   'precision',
                   'hmean_precision',
                   'hmean_recall',
                   'sensitivity',
                   'specificity',
                   'informedness')


def create_scorer(scoring):
    """
    Create a scoring function for hyperparameter tuning.

    Args
      scoring: The scoring method that the function should use. Possible
               values are enumerated by `SCORING_METHODS`.

    Returns
      A function that can be passed to the `scoring` parameter of
      `sklearn.model_selection.GridSearchCV`.

    """

    if scoring not in SCORING_METHODS:
        raise ValueError(f'`{scoring}` is not a valid scoring method.')

    def scorer(model, inputs, targets):
        model_scores = score_model(model, inputs, targets)
        try:
            score = model_scores[scoring]

        except KeyError:
            raise ValueError(f'`{scoring}` can not be used with this type of classification.')

        return score

    return scorer


def score_model(model, input_data, target_data):
    """
    Score the given model on a set of data. The scoring metrics used are
    accuracy, precision, sensitivity, specificity, and informedness.

    Args:
      model: A trained instance of a scikit-learn estimator.
      input_data: A 2D numpy array of inputs to the model where the rows
                  are samples and the columns are features.
      target_data: A 1D numpy array of expected model outputs.

    Returns:
      A dict with keys for all the scoring metrics that were performed. For
      binary classifiction, scoring metrics include accuracy, precision,
      sensitivity, specificity, recall, informedness, likelihood_ratio, mcc,
      f1_score, ami, dor, lr_plus, lr_minus, and roc_auc. For the multiclass
      situation, this includes accuracy, precision, recall, informedness, mcc,
      f1_score, and ami. Precision, recall, and f1_score in this situation are
      averaged by class weight.

    """

    if len(input_data) != len(target_data):
        raise ValueError('input_data and target_data must be the same length.')

    if np.ndim(target_data) != 1:
        raise ValueError('target_data must have dimensions N x 1.')

    predictions = model.predict(input_data)
    assert len(predictions) == len(target_data)
    assert np.ndim(predictions) == 1

    scores = dict()
    scores['accuracy'] = sklearn.metrics.accuracy_score(target_data, predictions)
    scores['informedness'] = informedness_score(target_data, predictions)
    scores['mcc'] = sklearn.metrics.matthews_corrcoef(target_data, predictions)
    scores['precision'] = precision_score(target_data, predictions)
    scores['recall'] = recall_score(target_data, predictions)
    scores['f1_score'] = f1_score(target_data, predictions)
    scores['ami'] = sklearn.metrics.adjusted_mutual_info_score(target_data, predictions)

    classes = np.unique(np.concatenate([target_data, predictions]))
    if len(classes) == 2:
        scores['sensitivity'] = sensitivity_score(target_data, predictions)
        scores['specificity'] = specificity_score(target_data, predictions)
        scores['dor'] = diagnostic_odds_ratio_score(target_data, predictions)
        scores['lr_plus'] = positive_likelihood_ratio_score(target_data, predictions)
        scores['lr_minus'] = negative_likelihood_ratio_score(target_data, predictions)
        try:
            scores['roc_auc'] = sklearn.metrics.roc_auc_score(target_data, predictions)

        except ValueError as value_error:
            logger = logging.getLogger(__name__)
            logger.warning(str(value_error))

    return scores


def diagnostic_odds_ratio_score(y_true, y_pred):
    """
    Compute the diagnostic odds ratio given by the formula:

    DOR = tp * tn / (fp * fn)

    Note that this function is only defined for binary classification and is
    also undefined when fp or fn equal 0.
    Source: https://www.sciencedirect.com/science/article/pii/S2210832718301546

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.

    Returns:
      The diagnostic odds ratio as a float or NaN if it is undefined. If both fp
      and fn are 0 (the model classifies perfectly), return Inf.

    """

    logger = logging.getLogger(__name__)
    confusion_matrix = sklearn.metrics.confusion_matrix(y_true, y_pred)
    if len(confusion_matrix[0]) != 2:
        raise ValueError('diagnostic odds ratio is undefined for the multiclass situation.')

    tp = confusion_matrix[0][0]
    fp = confusion_matrix[0][1]
    fn = confusion_matrix[1][0]
    tn = confusion_matrix[1][1]

    assert 0 <= tp <= len(y_true)
    assert 0 <= fp <= len(y_true)
    assert 0 <= fn <= len(y_true)
    assert 0 <= tn <= len(y_true)

    if fp == 0 and fn == 0:
        return np.inf

    if fp == 0 or fn == 0:
        logger.warning('diagnostic odds ratio undefined when fp or fn equal 0.')
        return np.nan

    return tp * tn / (fp * fn)


def positive_likelihood_ratio_score(y_true, y_pred, warn=True):
    """
    Compute the likelihood ratio for positive test results given by the
    formula:

    LR+ = sensitivity / (1 - specificity)

    Note that this function is only defined for binary classification.
    Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4975285/

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.
      warn: Whether to log a warning and return 0, nan, or inf when the
            function is undefined or raise an exception. Defaults to True.

    Returns:
      The likelihood ratio for positive test results as a float. If the
      specificity is 1 and warn is True, return inf.

    Raises:
      ValueError when `y_true` contains more than two classes and `warn`
      is False or specificity is 1.

    """

    logger = logging.getLogger(__name__)
    try:
        sensitivity = sensitivity_score(y_true, y_pred)
        specificity = specificity_score(y_true, y_pred)

    except ValueError:
        msg = 'likelihood ratio is undefined for the multiclass situation.'
        raise ValueError(msg)

    if specificity == 1:
        msg = 'positive likelihood ratio is undefined when specificity is 1.'
        if warn:
            logger.warning(msg)
            return np.inf

        raise ValueError(msg)

    return sensitivity / (1 - specificity)


def negative_likelihood_ratio_score(y_true, y_pred, warn=True):
    """
    Compute the likelihood ratio for negative test results given by the
    formula:

    LR- = (1 - sensitivity) / specificity

    Note that this function is only defined for binary classification.
    Source: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4975285/

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.
      warn: Whether to log a warning and return 0, nan, or inf when the
            function is undefined or raise an exception. Defaults to True.

    Returns:
      The likelihood ratio for negative test results as a float. If the
      specificity is 0 and warn is True, return inf.

    Raises:
      ValueError when `y_true` contains more than two classes or `warn`
      is True and specificity is 0.

    """

    logger = logging.getLogger(__name__)
    try:
        sensitivity = sensitivity_score(y_true, y_pred)
        specificity = specificity_score(y_true, y_pred)

    except ValueError:
        msg = 'likelihood ratio is undefined for the multiclass situation.'
        raise ValueError(msg)

    if specificity == 0:
        msg = 'negative likelihood ratio is undefined when specificity is 0.'
        if warn:
            logger.warning(msg)
            return np.inf

        raise ValueError(msg)

    return (1 - sensitivity) / specificity


def sensitivity_score(y_true, y_pred):
    """
    Compute the sensitivity, i.e. recall of the positive class.

    Note that this function is only defined for binary classification.
    Source: https://www.sciencedirect.com/science/article/pii/S2210832718301546

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.

    Returns:
      The sensitivity as a float.

    Raises:
      ValueError when `y_true` contains more than two classes.

    """

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) != 2:
        msg = 'sensitivity is not defined for the multiclass situation.'
        raise ValueError(msg)

    positive_class = np.max(classes)
    return sklearn.metrics.recall_score(y_true, y_pred, pos_label=positive_class)


def specificity_score(y_true, y_pred):
    """
    Compute the specificity, i.e. recall of the negative class.

    Note that this function is only defined for binary classification.
    Source: https://www.sciencedirect.com/science/article/pii/S2210832718301546

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.

    Returns:
      The specificity as a float.

    Raises:
      ValueError when `y_true` contains more than two classes.

    """

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) != 2:
        msg = 'specificity is not defined for the multiclass situation.'
        raise ValueError(msg)

    negative_class = np.min(classes)
    return sklearn.metrics.recall_score(y_true, y_pred, pos_label=negative_class)


def informedness_score(y_true, y_pred):
    """
    Compute the informedness, also known as Youden's J statistic.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.

    Returns:
      The informedness as a float.

    """

    if len(np.unique(np.concatenate([y_true, y_pred]))) == 2:
        return sensitivity_score(y_true, y_pred) + specificity_score(y_true, y_pred) - 1

    return sklearn.metrics.balanced_accuracy_score(y_true, y_pred, adjusted=True)


def precision_score(y_true, y_pred):
    """
    Compute the precision. For the multiclass situation, use the weighted
    average across all classes.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.

    Returns:
      The precision as a float.

    """

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 2:
        positive_class = np.max(classes)
        return sklearn.metrics.precision_score(y_true, y_pred,
                                               pos_label=positive_class)

    return sklearn.metrics.precision_score(y_true, y_pred, average='weighted')


def recall_score(y_true, y_pred):
    """
    Compute the recall. For the multiclass situation, use the weighted
    average across all classes.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.

    Returns:
      The recall as a float.

    """

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 2:
        positive_class = np.max(classes)
        return sklearn.metrics.recall_score(y_true, y_pred,
                                            pos_label=positive_class)

    return sklearn.metrics.recall_score(y_true, y_pred, average='weighted')


def f1_score(y_true, y_pred):
    """
    Compute the F1 score. For the multiclass situation, use the weighted
    average across all classes.

    Args:
      y_true: Ground truth (correct) target values.
      y_pred: Estimated targets as returned by a classifier.

    Returns:
      The F1 score as a float.

    """

    classes = np.unique(np.concatenate([y_true, y_pred]))
    if len(classes) == 2:
        return sklearn.metrics.f1_score(y_true, y_pred)

    return sklearn.metrics.f1_score(y_true, y_pred, average='weighted')
