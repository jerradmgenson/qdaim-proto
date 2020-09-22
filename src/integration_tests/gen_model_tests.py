import os
import unittest
import tempfile
import subprocess
import pickle
from pathlib import Path

import sklearn
from sklearn.datasets import load_iris

import gen_model


class PreprocessStage2Test(unittest.TestCase):
    """
    Test cases for preprocess_stage2.py

    """

    GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
    GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
    TEST_DATA = GIT_ROOT / Path('src/test_data')
    LDA_STANDARD_CONFIG = TEST_DATA / Path('gen_model_config_lda_standard.json')
    QDA_PCA_CONFIG = TEST_DATA / Path('gen_model_config_qda_pca.json')
    SVM_ROBUST_CONFIG = TEST_DATA / Path('gen_model_config_svm_robust.json')
    RFC_CONFIG = TEST_DATA / Path('gen_model_config_rfc.json')
    RRC_CONFIG = TEST_DATA / Path('gen_model_config_rrc.json')
    LRC_CONFIG = TEST_DATA / Path('gen_model_config_lrc.json')

    def setUp(self):
        tempfile_descriptor = tempfile.mkstemp()
        os.close(tempfile_descriptor[0])
        self.output_path = Path(tempfile_descriptor[1])
        self.prev_config_file_path = gen_model.CONFIG_FILE_PATH
        self.prev_default_output_path = gen_model.DEFAULT_OUTPUT_PATH
        gen_model.DEFAULT_OUTPUT_PATH = self.output_path

    def tearDown(self):
        gen_model.DEFAULT_OUTPUT_PATH = self.prev_default_output_path
        self.output_path.unlink()

    def test_lda_with_standard_scaling(self):
        """
        Test generation of a linear discriminant analysis model with
        standard scaling of the input data.

        """

        gen_model.CONFIG_FILE_PATH = self.LDA_STANDARD_CONFIG
        gen_model.main([])
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'standard scaling')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.StandardScaler)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.discriminant_analysis.LinearDiscriminantAnalysis)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.accuracy)
        self.assertGreater(accuracy, 0.95)

    def test_qda_with_pca(self):
        """
        Test generation of a quadratic discriminant analysis model with
        principal component analysis of the input data.

        """

        gen_model.CONFIG_FILE_PATH = self.QDA_PCA_CONFIG
        gen_model.main([])
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'pca')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.decomposition.PCA)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.discriminant_analysis.QuadraticDiscriminantAnalysis)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.accuracy)
        self.assertGreater(accuracy, 0.95)

    def test_svm_with_robust_scaling(self):
        """
        Test generation of a support vector machine model with robust
        scaling of the input data.

        """

        gen_model.CONFIG_FILE_PATH = self.SVM_ROBUST_CONFIG
        gen_model.main([])
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'robust scaling')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.RobustScaler)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1], sklearn.svm.SVC)
        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.accuracy)
        self.assertGreater(accuracy, 0.95)

    def test_rfc_with_no_preprocessing(self):
        """
        Test generation of a random forest model with no preprocessing
        of the input data.

        """

        gen_model.CONFIG_FILE_PATH = self.RFC_CONFIG
        gen_model.main([])
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

        self.assertAlmostEqual(accuracy, model.accuracy)
        self.assertGreater(accuracy, 0.95)

    def test_rrc_with_quantile_transformer(self):
        """
        Test generation of a ridge regression model with quantile
        transformation of the input data.

        """

        gen_model.CONFIG_FILE_PATH = self.RRC_CONFIG
        gen_model.main([])
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'quantile transformer')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.QuantileTransformer)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.linear_model.RidgeClassifier)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.accuracy)
        self.assertGreater(accuracy, 0.88)

    def test_lrc_with_power_transformer(self):
        """
        Test generation of a logistic regression model with power
        transformation of the input data.

        """

        gen_model.CONFIG_FILE_PATH = self.LRC_CONFIG
        gen_model.main([])
        with open(self.output_path, 'rb') as output_fp:
            model = pickle.load(output_fp)

        self.assertIsInstance(model, sklearn.pipeline.Pipeline)
        self.assertEqual(len(model.steps), 2)
        self.assertEqual(model.steps[0][0], 'power transformer')
        self.assertIsInstance(model.steps[0][1],
                              sklearn.preprocessing.PowerTransformer)

        self.assertEqual(model.steps[1][0], 'model')
        self.assertIsInstance(model.steps[1][1],
                              sklearn.linear_model.LogisticRegression)

        iris_dataset = load_iris()
        predictions = model.predict(iris_dataset['data'])
        accuracy = sklearn.metrics.accuracy_score(iris_dataset['target'],
                                                  predictions)

        self.assertAlmostEqual(accuracy, model.accuracy)
        self.assertGreater(accuracy, 0.88)
