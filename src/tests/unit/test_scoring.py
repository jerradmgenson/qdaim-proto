"""
Unit tests for scoring.py

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import unittest
from unittest.mock import Mock

import numpy as np

import scoring


class CreateScorerTests(unittest.TestCase):
    """
    Tests for scoring.create_scorer

    """

    def test_informedness_metric(self):
        """
        Test create_scorer() with informedness scoring metric.

        """

        inputs = np.array([0, 0, 0, 0])
        targets = np.array([0, 1, 0, 1])
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 1, 0]))
        scorer = scoring.create_scorer('informedness')
        score = scorer(model, inputs, targets)
        self.assertEqual(score, 0.0)

    def test_accuracy_metric(self):
        """
        Test create_scorer() with accuracy scoring metric.

        """

        inputs = np.array([0, 0, 0, 0])
        targets = np.array([0, 1, 0, 1])
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 1, 0]))
        scorer = scoring.create_scorer('accuracy')
        score = scorer(model, inputs, targets)
        self.assertEqual(score, 0.5)

    def test_scorer_with_invalid_metric1(self):
        """
        Test create_scorer() with a scoring metric that is invalid for
        the given type of classification.

        """

        inputs = np.array([1, 2, 3, 4])
        targets = np.array([1, 2, 3, 4])
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 2, 3, 4]))
        scorer = scoring.create_scorer('sensitivity')
        with self.assertRaises(ValueError):
            scorer(model, inputs, targets)

    def test_scorer_with_invalid_metric2(self):
        """
        Test create_scorer() with a scoring metric that is invalid for
        any type of classification.

        """

        with self.assertRaises(ValueError):
            scoring.create_scorer('invalid')


class CalculateInformednessTests(unittest.TestCase):
    """
    Tests for scoring.calculate_informedness

    """

    def test_100_percent_correct_binary_classification(self):
        """
        Test calculate_informedness() with 100% correct binary classification.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 1.0)

    def test_50_percent_correct_binary_classification(self):
        """
        Test calculate_informedness() with 50% correct binary classification.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 0)

    def test_random_binary_classifications(self):
        """
        Test calculate_informedness() with random binary classifications.

        """

        y_true = np.array([1, 0, 1, 1, 1, 1, 0, 0, 1, 0])
        y_pred = np.array([0, 1, 1, 1, 1, 1, 0, 1, 0, 1])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertAlmostEqual(informedness, -0.08333333333333337)

    def test_100_percent_correct_ternary_classification(self):
        """
        Test calculate_informedness() with 100% correct ternary classification.

        """

        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        y_pred = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 1.0)

    def test_50_percent_correct_ternary_classification(self):
        """
        Test calculate_informedness() with 50% correct ternary classification.

        """

        y_true = np.array([0, 0, 0, 1, 1, 1, 2, 2, 2, 2])
        y_pred = np.array([0, 0, 0, 1, 1, 0, 0, 0, 0, 0])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 0.3333333333333332)

    def test_random_ternary_classifications(self):
        """
        Test calculate_informedness() with random ternary classifications.

        """

        y_true = np.array([2, 0, 1, 2, 0, 0, 0, 1, 0, 1])
        y_pred = np.array([0, 1, 0, 0, 2, 2, 0, 2, 1, 1])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertAlmostEqual(informedness, -0.23333333333333328)

    def test_100_percent_correct_quaternary_classification(self):
        """
        Test calculate_informedness() with 100% correct quaternary classification.

        """

        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
        y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 1.0)

    def test_50_percent_correct_quaternary_classification(self):
        """
        Test calculate_informedness() with 50% correct quaternary classification.

        """

        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 3, 3])
        y_pred = np.array([0, 0, 1, 1, 2, 0, 0, 0, 0, 0])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 0.5)

    def test_random_quaternary_classifications(self):
        """
        Test calculate_informedness() with random quaternary classifications.

        """

        y_true = np.array([3, 2, 3, 0, 1, 3, 1, 1, 4, 4])
        y_pred = np.array([0, 3, 4, 2, 0, 0, 2, 3, 0, 3])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertAlmostEqual(informedness, -0.25)

    def test_100_percent_correct_quinary_classification(self):
        """
        Test calculate_informedness() with 100% correct quinary classification.

        """

        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 1.0)

    def test_50_percent_correct_quinary_classification(self):
        """
        Test calculate_informedness() with 50% correct quinary classification.

        """

        y_true = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        y_pred = np.array([0, 1, 1, 0, 2, 0, 3, 0, 4, 0])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertEqual(informedness, 0.37499999999999994)

    def test_random_quinary_classifications(self):
        """
        Test calculate_informedness() with random quinary classifications.

        """

        y_true = np.array([4, 2, 0, 5, 4, 0, 2, 0, 3, 1])
        y_pred = np.array([5, 1, 1, 1, 2, 0, 3, 4, 5, 1])
        informedness = scoring.informedness_score(y_true, y_pred)
        self.assertAlmostEqual(informedness, 0.06666666666666667)


class ScoreModelTests(unittest.TestCase):
    """
    Tests for scoring.score_model

    """

    def test_100_percent_binary_classification(self):
        """
        Test score_model() with 100% correct binary classification.

        """

        input_data = np.array([0, 0, 0, 0])
        target_data = np.array([0, 1, 1, 0])
        model = Mock()
        model.predict = Mock(return_value=target_data)
        scores = scoring.score_model(model, input_data, target_data)
        self.assertEqual(model.predict.call_count, 1)
        self.assertTrue((model.predict.call_args[0][0] == input_data).all())
        self.assertEqual(scores['accuracy'], 1.0)
        self.assertEqual(scores['precision'], 1.0)
        self.assertEqual(scores['recall'], 1.0)
        self.assertEqual(scores['sensitivity'], 1.0)
        self.assertEqual(scores['specificity'], 1.0)
        self.assertEqual(scores['informedness'], 1.0)
        self.assertEqual(scores['mcc'], 1.0)
        self.assertEqual(scores['f1_score'], 1.0)
        self.assertEqual(scores['ami'], 1.0)
        self.assertEqual(scores['dor'], np.inf)
        self.assertEqual(scores['lr_plus'], np.inf)
        self.assertEqual(scores['lr_minus'], 0.0)
        self.assertEqual(scores['roc_auc'], 1.0)

    def test_50_percent_binary_classification(self):
        """
        Test score_model() with 50% correct binary classification.

        """

        input_data = np.array([0, 1, 0, 1])
        target_data = input_data
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 1, 0]))
        scores = scoring.score_model(model, input_data, target_data)
        self.assertAlmostEqual(scores['accuracy'], 0.5)
        self.assertAlmostEqual(scores['precision'], 0.5)
        self.assertAlmostEqual(scores['recall'], 0.5)
        self.assertAlmostEqual(scores['f1_score'], 0.5)
        self.assertAlmostEqual(scores['ami'], -0.49999999999999956)
        self.assertAlmostEqual(scores['sensitivity'], 0.5)
        self.assertAlmostEqual(scores['specificity'], 0.5)
        self.assertAlmostEqual(scores['informedness'], 0)
        self.assertAlmostEqual(scores['dor'], 1.0)
        self.assertAlmostEqual(scores['lr_plus'], 1.0)
        self.assertAlmostEqual(scores['lr_minus'], 1.0)
        self.assertAlmostEqual(scores['roc_auc'], 0.5)

    def test_random_binary_classifications(self):
        """
        Test score_model() with random binary classifications.

        """

        input_data = np.array([0, 1, 0, 1])
        target_data = input_data
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 0, 1, 0]))
        scores = scoring.score_model(model, input_data, target_data)
        self.assertAlmostEqual(scores['accuracy'], 0.25)
        self.assertEqual(scores['precision'], 0)
        self.assertAlmostEqual(scores['sensitivity'], 0)
        self.assertAlmostEqual(scores['specificity'], 0.5)
        self.assertAlmostEqual(scores['informedness'], -0.5)
        self.assertAlmostEqual(scores['mcc'], -0.5773502691896258)
        self.assertAlmostEqual(scores['recall'], 0.0)
        self.assertAlmostEqual(scores['f1_score'], 0.0)
        self.assertAlmostEqual(scores['ami'], 2.6948494595149616e-16)
        self.assertAlmostEqual(scores['dor'], 0.0)
        self.assertAlmostEqual(scores['lr_plus'], 0.0)
        self.assertAlmostEqual(scores['lr_minus'], 2.0)
        self.assertAlmostEqual(scores['roc_auc'], 0.25)

    def test_100_percent_ternary_classification(self):
        """
        Test score_model() with 100% correct ternary classification.

        """

        input_data = np.array([0, 1, 2, 0, 1, 2])
        target_data = input_data
        model = Mock()
        model.predict = Mock(return_value=target_data)
        scores = scoring.score_model(model, input_data, target_data)
        self.assertEqual(scores['accuracy'], 1.0)
        self.assertEqual(scores['precision'], 1.0)
        self.assertEqual(scores['recall'], 1.0)
        self.assertEqual(scores['informedness'], 1.0)
        self.assertEqual(scores['mcc'], 1.0)
        self.assertEqual(scores['f1_score'], 1.0)
        self.assertEqual(scores['ami'], 1.0)

    def test_50_percent_ternary_classification(self):
        """
        Test score_model() with 50% correct ternary classification.

        """

        input_data = np.array([0, 1, 2, 0, 1, 2])
        target_data = input_data
        model = Mock()
        model.predict = Mock(return_value=np.array([0, 1, 2, 2, 0, 1]))
        scores = scoring.score_model(model, input_data, target_data)
        self.assertAlmostEqual(scores['accuracy'], 0.5)
        self.assertAlmostEqual(scores['informedness'], 0.25)
        self.assertAlmostEqual(scores['mcc'], 0.25)
        self.assertAlmostEqual(scores['precision'], 0.5)
        self.assertAlmostEqual(scores['recall'], 0.5)
        self.assertAlmostEqual(scores['f1_score'], 0.5)
        self.assertAlmostEqual(scores['ami'], -0.2499999999999995)

    def test_random_ternary_classifications(self):
        """
        Test score_model() with random ternary classifications.

        """

        input_data = np.array([0, 1, 2, 0, 1, 2])
        target_data = input_data
        model = Mock()
        model.predict = Mock(return_value=np.array([1, 1, 1, 2, 2, 0]))
        scores = scoring.score_model(model, input_data, target_data)
        self.assertAlmostEqual(scores['accuracy'], 0.1666667)
        self.assertAlmostEqual(scores['informedness'], -0.24999999999999997)
        self.assertAlmostEqual(scores['mcc'], -0.26111648393354675)
        self.assertAlmostEqual(scores['precision'], 0.1111111111111111)
        self.assertAlmostEqual(scores['recall'], 0.16666666666666666)
        self.assertAlmostEqual(scores['f1_score'], 0.13333333333333333)
        self.assertAlmostEqual(scores['ami'], -0.3349071351468493)

    def test_different_length_arrays(self):
        """
        Test score_model() with input_data and target_data of different
        lengths.

        """

        input_data = np.array([0, 1, 2, 0, 1, 2])
        target_data = np.array([])
        model = Mock()
        with self.assertRaises(ValueError):
            scoring.score_model(model, input_data, target_data)

    def test_target_data_wrong_dimensions(self):
        """
        Test score_model() with target_data with the wrong dimensions.

        """

        input_data = np.array([0, 1, 2, 0, 1, 2])
        target_data = np.array([[0, 0], [1, 1], [2, 2], [0, 0], [1, 1], [2, 2]])
        model = Mock()
        with self.assertRaises(ValueError):
            scoring.score_model(model, input_data, target_data)


class DiagnosticOddsRatioScoreTest(unittest.TestCase):
    """
    Tests for scoring.diagnostic_odds_ratio_score()

    """

    def test_perfect_binary_classifier(self):
        """
        Test a binary model that always predicts correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        dor = scoring.diagnostic_odds_ratio_score(y_true, y_pred)
        self.assertEqual(dor, np.inf)

    def test_useless_binary_classifier(self):
        """
        Test a binary model that always predicts incorrectly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        dor = scoring.diagnostic_odds_ratio_score(y_true, y_pred)
        self.assertEqual(dor, 0)

    def test_half_correct_binary_classifier(self):
        """
        Test a binary model that predicts 50% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        dor = scoring.diagnostic_odds_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(dor, 2.25)

    def test_mostly_correct_binary_classifier(self):
        """
        Test a binary model that predicts 80% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        dor = scoring.diagnostic_odds_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(dor, 16)

    def test_mostly_incorrect_binary_classifier(self):
        """
        Test a binary model that predicts 20% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        dor = scoring.diagnostic_odds_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(dor, 0.0625)

    def test_binary_classifier_with_no_false_negatives(self):
        """
        Test a binary model that predicts with no false negatives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        dor = scoring.diagnostic_odds_ratio_score(y_true, y_pred)
        self.assertTrue(np.isnan(dor))

    def test_binary_classifier_with_no_false_positives(self):
        """
        Test a binary model that predicts with no false positives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        dor = scoring.diagnostic_odds_ratio_score(y_true, y_pred)
        self.assertTrue(np.isnan(dor))

    def test_multiclass_classifier_raises_value_error(self):
        """
        Test that multiclass models correctly raise ValueError.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        with self.assertRaises(ValueError):
            scoring.diagnostic_odds_ratio_score(y_true, y_pred)


class PositiveLikelihoodRatioScoreTest(unittest.TestCase):
    """
    Tests for scoring.positive_likelihood_ratio_score()

    """

    def test_perfect_binary_classifier_warn_true(self):
        """
        Test a binary model that always predicts correctly with warnings.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        lr_plus = scoring.positive_likelihood_ratio_score(y_true, y_pred,
                                                          warn=True)

        self.assertEqual(lr_plus, np.inf)

    def test_perfect_binary_classifier_warn_false(self):
        """
        Test a binary model that always predicts correctly without warnings.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            scoring.positive_likelihood_ratio_score(y_true, y_pred,
                                                    warn=False)

    def test_useless_binary_classifier(self):
        """
        Test a binary model that always predicts incorrectly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        lr_plus = scoring.positive_likelihood_ratio_score(y_true, y_pred)
        self.assertEqual(lr_plus, 0)

    def test_half_correct_binary_classifier(self):
        """
        Test a binary model that predicts 50% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        lr_plus = scoring.positive_likelihood_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(lr_plus, 1.4999999999999998)

    def test_mostly_correct_binary_classifier(self):
        """
        Test a binary model that predicts 80% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        lr_plus = scoring.positive_likelihood_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(lr_plus, 4)

    def test_mostly_incorrect_binary_classifier(self):
        """
        Test a binary model that predicts 20% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        lr_plus = scoring.positive_likelihood_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(lr_plus, 0.25)

    def test_binary_classifier_with_no_false_positives(self):
        """
        Test a binary model that predicts with no false positives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        lr_plus = scoring.positive_likelihood_ratio_score(y_true, y_pred)
        self.assertEqual(lr_plus, np.inf)

    def test_binary_classifier_with_no_false_negatives(self):
        """
        Test a binary model that predicts with no false negatives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        lr_plus = scoring.positive_likelihood_ratio_score(y_true, y_pred)
        self.assertEqual(lr_plus, 1)

    def test_multiclass_classifier_raises_value_error(self):
        """
        Test that multiclass models correctly raise ValueError.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        with self.assertRaises(ValueError):
            scoring.positive_likelihood_ratio_score(y_true, y_pred)


class NegativeLikelihoodRatioScoreTest(unittest.TestCase):
    """
    Tests for scoring.negative_likelihood_ratio_score()

    """

    def test_perfect_binary_classifier(self):
        """
        Test a binary model that always predicts correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        lr_minus = scoring.negative_likelihood_ratio_score(y_true, y_pred)
        self.assertEqual(lr_minus, 0)

    def test_useless_binary_classifier(self):
        """
        Test a binary model that always predicts incorrectly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        lr_minus = scoring.negative_likelihood_ratio_score(y_true, y_pred)
        self.assertEqual(lr_minus, np.inf)

    def test_half_correct_binary_classifier(self):
        """
        Test a binary model that predicts 50% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        lr_minus = scoring.negative_likelihood_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(lr_minus, 0.6666666666666667)

    def test_mostly_correct_binary_classifier(self):
        """
        Test a binary model that predicts 80% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        lr_minus = scoring.negative_likelihood_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(lr_minus, 0.24999999999999994)

    def test_mostly_incorrect_binary_classifier(self):
        """
        Test a binary model that predicts 20% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        lr_minus = scoring.negative_likelihood_ratio_score(y_true, y_pred)
        self.assertAlmostEqual(lr_minus, 4.0)

    def test_binary_classifier_with_no_false_positives(self):
        """
        Test a binary model that predicts with no false positives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        lr_minus = scoring.negative_likelihood_ratio_score(y_true, y_pred)
        self.assertEqual(lr_minus, 1)

    def test_binary_classifier_with_no_false_negatives_with_warnings(self):
        """
        Test a binary model that predicts with no false negatives with warnings.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        lr_minus = scoring.negative_likelihood_ratio_score(y_true, y_pred,
                                                           warn=True)

        self.assertEqual(lr_minus, np.inf)

    def test_binary_classifier_with_no_false_negatives_without_warnings(self):
        """
        Test a binary model that predicts with no false negatives without warnings.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        with self.assertRaises(ValueError):
            scoring.negative_likelihood_ratio_score(y_true, y_pred,
                                                    warn=False)

    def test_multiclass_classifier_raises_value_error(self):
        """
        Test that multiclass models correctly raise ValueError.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        with self.assertRaises(ValueError):
            scoring.negative_likelihood_ratio_score(y_true, y_pred)


class SensitivityScoreTest(unittest.TestCase):
    """
    Tests for scoring.sensitivity_score()

    """

    def test_perfect_binary_classifier(self):
        """
        Test a binary model that always predicts correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        sensitivity = scoring.sensitivity_score(y_true, y_pred)
        self.assertEqual(sensitivity, 1)

    def test_useless_binary_classifier(self):
        """
        Test a binary model that always predicts incorrectly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        sensitivity = scoring.sensitivity_score(y_true, y_pred)
        self.assertEqual(sensitivity, 0)

    def test_half_correct_binary_classifier(self):
        """
        Test a binary model that predicts 50% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        sensitivity = scoring.sensitivity_score(y_true, y_pred)
        self.assertAlmostEqual(sensitivity, 0.6)

    def test_mostly_correct_binary_classifier(self):
        """
        Test a binary model that predicts 80% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        sensitivity = scoring.sensitivity_score(y_true, y_pred)
        self.assertAlmostEqual(sensitivity, 0.8)

    def test_mostly_incorrect_binary_classifier(self):
        """
        Test a binary model that predicts 20% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        sensitivity = scoring.sensitivity_score(y_true, y_pred)
        self.assertAlmostEqual(sensitivity, 0.2)

    def test_binary_classifier_with_no_false_positives(self):
        """
        Test a binary model that predicts with no false positives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        sensitivity = scoring.sensitivity_score(y_true, y_pred)
        self.assertEqual(sensitivity, 0)

    def test_binary_classifier_with_no_false_negatives(self):
        """
        Test a binary model that predicts with no false negatives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        sensitivity = scoring.sensitivity_score(y_true, y_pred)
        self.assertEqual(sensitivity, 1)

    def test_multiclass_classifier_raises_value_error(self):
        """
        Test that multiclass models correctly raise ValueError.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        with self.assertRaises(ValueError):
            scoring.sensitivity_score(y_true, y_pred)


class SpecificityScoreTest(unittest.TestCase):
    """
    Tests for scoring.specificity_score()

    """

    def test_perfect_binary_classifier(self):
        """
        Test a binary model that always predicts correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        specificity = scoring.specificity_score(y_true, y_pred)
        self.assertEqual(specificity, 1)

    def test_useless_binary_classifier(self):
        """
        Test a binary model that always predicts incorrectly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0])
        specificity = scoring.specificity_score(y_true, y_pred)
        self.assertEqual(specificity, 0)

    def test_half_correct_binary_classifier(self):
        """
        Test a binary model that predicts 50% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
        specificity = scoring.specificity_score(y_true, y_pred)
        self.assertAlmostEqual(specificity, 0.6)

    def test_mostly_correct_binary_classifier(self):
        """
        Test a binary model that predicts 80% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 0])
        specificity = scoring.specificity_score(y_true, y_pred)
        self.assertAlmostEqual(specificity, 0.8)

    def test_mostly_incorrect_binary_classifier(self):
        """
        Test a binary model that predicts 20% correctly.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0, 0, 1])
        specificity = scoring.specificity_score(y_true, y_pred)
        self.assertAlmostEqual(specificity, 0.2)

    def test_binary_classifier_with_no_false_positives(self):
        """
        Test a binary model that predicts with no false positives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        specificity = scoring.specificity_score(y_true, y_pred)
        self.assertEqual(specificity, 1)

    def test_binary_classifier_with_no_false_negatives(self):
        """
        Test a binary model that predicts with no false negatives.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        specificity = scoring.specificity_score(y_true, y_pred)
        self.assertEqual(specificity, 0)

    def test_multiclass_classifier_raises_value_error(self):
        """
        Test that multiclass models correctly raise ValueError.

        """

        y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 2])
        y_pred = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 2])
        with self.assertRaises(ValueError):
            scoring.specificity_score(y_true, y_pred)


if __name__ == '__main__':
    unittest.main()
