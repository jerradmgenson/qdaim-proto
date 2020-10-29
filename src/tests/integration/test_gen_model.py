"""
Integration testcases for gen_model.py.

Copyright 2020 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import os
import json
import unittest
import tempfile
import subprocess
import pickle
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.datasets import load_iris

import gen_model
from tests.integration import test_preprocess_stage1
from tests.integration import test_preprocess_stage2

GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
TEST_DATA = GIT_ROOT / Path('src/tests/data')


class GenModelTestCase(unittest.TestCase):
    """
    Base class for all gen_model.py testcases. Defines a common set of
    setUp() and tearDown() methods.

    """

    QDA_PCA_CONFIG = TEST_DATA / Path('gen_model_config_qda_pca.json')

    def setUp(self):
        tempfile_descriptor = tempfile.mkstemp()
        os.close(tempfile_descriptor[0])
        self.output_path = Path(tempfile_descriptor[1])
        self.validation_path = (Path(self.output_path)
                                .with_name(Path(self.output_path).name + '.csv'))

        self.logfile_path = (Path(self.output_path)
                             .with_name(Path(self.output_path).name + '.log'))

    def tearDown(self):
        if self.output_path.exists():
            self.output_path.unlink()

        if self.validation_path.exists():
            self.validation_path.unlink()

        if self.logfile_path.exists():
            self.logfile_path.unlink()


class ModelConfigTestCase(GenModelTestCase):
    """
    Test cases for gen_model.py configuration file

    """

    LDA_STANDARD_CONFIG = TEST_DATA / Path('gen_model_config_lda_standard.json')
    SVM_ROBUST_CONFIG = TEST_DATA / Path('gen_model_config_svm_robust.json')
    RFC_CONFIG = TEST_DATA / Path('gen_model_config_rfc.json')
    RRC_CONFIG = TEST_DATA / Path('gen_model_config_rrc.json')
    LRC_CONFIG = TEST_DATA / Path('gen_model_config_lrc.json')
    ETC_CONFIG = TEST_DATA / Path('gen_model_config_etc.json')
    SGD_CONFIG = TEST_DATA / Path('gen_model_config_sgd.json')
    DTC_CONFIG = TEST_DATA / Path('gen_model_config_dtc.json')

    def test_lda_with_standard_scaling(self):
        """
        Test generation of a linear discriminant analysis model with
        standard scaling of the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.LDA_STANDARD_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.StandardScaler)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.discriminant_analysis.LinearDiscriminantAnalysis)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.95)

    def test_qda_with_pca(self):
        """
        Test generation of a quadratic discriminant analysis model with
        principal component analysis of the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.QDA_PCA_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.decomposition.PCA)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.95)

    def test_svm_with_robust_scaling(self):
        """
        Test generation of a support vector machine model with robust
        scaling of the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.SVM_ROBUST_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.RobustScaler)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1], sklearn.svm.SVC)
        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.95)

    def test_rfc_with_no_preprocessing(self):
        """
        Test generation of a random forest model with no preprocessing
        of the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.RFC_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 1)
        self.assertEqual(model.steps[0][0], 'model')
        self.assertIsInstance(model.steps[0][1], sklearn.ensemble.RandomForestClassifier)
        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.95)

    def test_rrc_with_quantile_transformer(self):
        """
        Test generation of a ridge regression model with quantile
        transformation of the input data.

        """

        gen_model.CONFIG_FILE_PATH = self.RRC_CONFIG
        exit_code = gen_model.main([str(self.output_path),
                                    str(self.RRC_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.QuantileTransformer)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.linear_model.RidgeClassifier)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.88)

    def test_lrc_with_power_transformer(self):
        """
        Test generation of a logistic regression model with power
        transformation of the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.LRC_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.PowerTransformer)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.linear_model.LogisticRegression)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.88)

    def test_etc_with_normalization(self):
        """
        Test generation of an extra trees model with normalization of
        the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.ETC_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.Normalizer)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.ensemble.ExtraTreesClassifier)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.95)

    def test_sgd_with_standard_scaling(self):
        """
        Test generation of a stochastic gradient descent model with
        standard scaling of the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.SGD_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.StandardScaler)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.linear_model.SGDClassifier)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.95)

    def test_dtc_with_robust_scaling(self):
        """
        Test generation of a stochastic gradient descent model with
        standard scaling of the input data.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.DTC_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.RobustScaler)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.tree.DecisionTreeClassifier)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.validation['scores']['accuracy'])
        self.assertGreater(accuracy, 0.95)


class GenModelIntegrationTestCase(GenModelTestCase):
    """
    Testcase for integration of gen_model.py with preprocess_stage2.py

    """

    GEN_MODEL_CONFIG = TEST_DATA / Path('gen_model_config_dtc.json')

    def test_run_gen_model_with_preprocess_stage2(self):
        """
        Test running gen_model.py on the output of preprocess_stage2.R

        """

        test_preprocess_stage2.setUp(self)
        with self.GEN_MODEL_CONFIG.open() as config_template_fp:
            gen_model_config = json.load(config_template_fp)

        gen_model_config['training_dataset'] = str(self.training_path)
        gen_model_config['validation_dataset'] = str(self.validation_path)
        config_tempfile_descriptor = tempfile.mkstemp()
        os.close(config_tempfile_descriptor[0])
        with open(config_tempfile_descriptor[1], 'w') as tmp_config_fp:
            json.dump(gen_model_config, tmp_config_fp)

        subprocess.check_output([test_preprocess_stage2.PreprocessStage2Test.PREPROCESS_STAGE2,
                                 str(self.output_directory),
                                 str(test_preprocess_stage1.EXPECTED_OUTPUT_DEFAULT_PARAMETERS)])

        gen_model.CONFIG_FILE_PATH = Path(config_tempfile_descriptor[1])
        exit_code = gen_model.main([str(self.output_path),
                                    config_tempfile_descriptor[1]])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'preprocessing')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.RobustScaler)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.tree.DecisionTreeClassifier)

        testing_dataset = pd.read_csv(self.testing_path)
        testing_inputs = testing_dataset.to_numpy()[:, 0:-1]
        testing_targets = testing_dataset.to_numpy()[:, -1]
        predictions = model.predict(testing_inputs)
        accuracy = sklearn.metrics.accuracy_score(testing_targets, predictions)
        self.assertGreaterEqual(accuracy, 0.5)
        test_preprocess_stage2.tearDown(self)


class ValidationCSVTestCase(GenModelTestCase):
    """
    Testcase for the *_validation.csv output file of gen_model.py

    """

    DTC_MODEL_CONFIG = TEST_DATA / Path('gen_model_config_dtc.json')
    QDA_MODEL_CONFIG = TEST_DATA / Path('gen_model_config_qda_pca.json')

    def test_dtc_validation(self):
        """
        Test *_validation.csv when generating a decision tree classifier.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.DTC_MODEL_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        validation = pd.read_csv(self.validation_path)
        iris_dataset = load_iris()
        self.assertEqual(len(validation.columns),
                         len(iris_dataset['data'][0]) + 2)

        self.assertEqual(set(validation.columns[:-2]),
                         set(iris_dataset['feature_names']))

        self.assertTrue((validation.columns[-2:] == ['target', 'prediction']).all())
        predictions = model.predict(iris_dataset['data'])
        self.assertTrue((set(predictions) == set(validation['prediction'])))

    def test_qda_validation(self):
        """
        Test *_validation.csv when generating a quadratic discriminant
        analysis model.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.QDA_MODEL_CONFIG)])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        validation = pd.read_csv(self.validation_path)
        iris_dataset = load_iris()
        self.assertEqual(len(validation.columns),
                         len(iris_dataset['data'][0]) + 2)

        self.assertEqual(set(validation.columns[:-2]),
                         set(iris_dataset['feature_names']))

        self.assertTrue((validation.columns[-2:] == ['target', 'prediction']).all())
        predictions = model.predict(iris_dataset['data'])
        self.assertTrue((set(predictions) == set(validation['prediction'])))


class CrossValidationTestCase(GenModelTestCase):
    """
    Test gen_model.cross_validate() integration with other functions.

    """

    def test_main_multiclass(self):
        """
        Test that gen_model.main() calls cross_validate() correctly with
        multiclass classification.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.QDA_PCA_CONFIG),
                                    '--cross-validate',
                                    '5'])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertAlmostEqual(model.validation['cross_validation_median']['accuracy'], 0.9666666666666667)
        self.assertAlmostEqual(model.validation['cross_validation_mad']['accuracy'], 0.0)
        self.assertAlmostEqual(model.validation['cross_validation_median']['informedness'], 0.9555335968379446)
        self.assertAlmostEqual(model.validation['cross_validation_mad']['informedness'], 0.006849386311628791)

        with self.assertRaises(AttributeError):
            model.median_sensitivity

        with self.assertRaises(AttributeError):
            model.mad_sensitivity

        with self.assertRaises(AttributeError):
            model.median_specificity

        with self.assertRaises(AttributeError):
            model.mad_specificity


class OutliersTestCase(GenModelTestCase):
    """
    Test that gen_model.py integrates correctly with outliers.py

    """

    def test_main_outliers_score(self):
        """
        Test that gen_model.main() calls outliers.score() correctly.

        """

        exit_code = gen_model.main([str(self.output_path),
                                    str(self.QDA_PCA_CONFIG),
                                    '--outlier-scores'])

        self.assertEqual(exit_code, 0)
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertAlmostEqual(model.validation['outlier_scores']['accuracy'], 1.0)
        self.assertAlmostEqual(model.validation['outlier_scores']['precision'], 1.0)
        self.assertAlmostEqual(model.validation['outlier_scores']['recall'], 1.0)
        self.assertAlmostEqual(model.validation['outlier_scores']['informedness'], 1.0)
