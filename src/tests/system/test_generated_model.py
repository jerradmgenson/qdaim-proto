"""
System tests for the final generated model.

Copyright 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

"""

import json
import pickle
import unittest
import subprocess
from pathlib import Path

import sklearn
import pandas as pd

import util
import scoring


GIT_ROOT = subprocess.check_output(['git', 'rev-parse', '--show-toplevel'])
GIT_ROOT = Path(GIT_ROOT.decode('utf-8').strip())
BUILD_DIR = GIT_ROOT / 'build'
MODEL = BUILD_DIR / 'qdaim.dat'


class GeneratedModelTest(unittest.TestCase):
    def setUp(self):
        with MODEL.open('rb') as model_fp:
            self.model = pickle.load(model_fp)

    def test_model_is_sklearn_classifier(self):
        """
        Test that the model can be loaded and is a valid sklearn classifier.

        """

        self.assertTrue(sklearn.base.is_classifier(self.model))

    def test_correct_pipeline_steps(self):
        """
        Test that the model's pipeline steps are correct with respect to what is
        specified in model_gen.json.

        """

        with (GIT_ROOT / 'cfg/model_gen.json').open() as model_gen_fp:
            model_gen = json.load(model_gen_fp)

        for index, method in enumerate(model_gen['preprocessing'], start=1):
            step = 'preprocessing' + str(index)
            self.assertIsInstance(self.model[step],
                                  util.PREPROCESSING_METHODS[method])

        self.assertIsInstance(self.model['model'],
                              util.SUPPORTED_ALGORITHMS[model_gen['model']][1])

    def test_model_predict(self):
        """
        Test that the model is able to make predictions from the training data
        and that the predictions are not useless.

        """

        train = pd.read_csv(BUILD_DIR / 'training.csv')
        inputs = train.loc[:, train.columns != 'target']
        targets = train['target']
        scores = scoring.score_model(self.model, inputs, targets)

        self.assertGreater(scores['informedness'], 0)
