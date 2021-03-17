#!/usr/bin/Rscript

## usage: preprocess.R [--] [--help] [--impute-missing]
##        [--impute-multiple] [--opts OPTS] [--random-state RANDOM-STATE]
##        [--classification-type CLASSIFICATION-TYPE] [--test-fraction
##        TEST-FRACTION] [--validation-fraction VALIDATION-FRACTION]
##        [--features FEATURES] [--impute-methods IMPUTE-METHODS] training
##        testing validation source test-pool

## Clean, standardize, and impute missing data so that it can be modelled.

## Copyright 2020, 2021 Jerrad M. Genson

## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at https://mozilla.org/MPL/2.0/.

## Preprocessing steps performed by this script include:

## - Omit all rows where trestbps is equal to 0.

## - Omit rows containing NA or impute them using mice.

## - Convert cp to a binary class.

## - Convert restecg to a binary class.

## - Optionally convert target to a binary or ternary class.

## - Rescale binary and ternary classes to range from -1 to 1.

## - Randomize row order.

## - Split data into test, train, and validation sets.

## positional arguments:
##   training                   Path to write the training dataset to.
##   testing                    Path to write the test dataset to.
##   validation                 Path to write the validation dataset to.
##   source                     Input directory of CSV data files.
##   test-pool                  Name of the dataset to draw the test data
##                              from.

## flags:
##   -h, --help                 show this help message and exit
##   -i, --impute-missing       Impute rows with single NAs in the
##                              training and validation datasets.
##   --impute-multiple          Impute rows with multiple NAs in the
##                              training and validation datasets.
##                              --impute-missing has no effect when
##                              --impute-multiple is present.

## optional arguments:
##   -x, --opts                 RDS file containing argument values
##   -r, --random-state         State to initialize random number
##                              generators with. [default: 0]
##   -c, --classification-type  Classification type. Possible values:
##                              'binary', 'ternary', 'multiclass'
##                              [default: binary]
##   -t, --test-fraction        Fraction of data to use for testing as a
##                              real number between 0 and 1. [default:
##                              0.2]
##   -v, --validation-fraction  Fraction of data to use for validation as
##                              a real number between 0 and 1. [default:
##                              0.2]
##   -f, --features             Features to select from the input
##                              datasets.
##   --impute-methods           Methods to use for imputation. Methods
##                              must correspond to --features (if given)
##                              or columns of the input datasets.

git_root <-
    system2("git", args = c("rev-parse", "--show-toplevel"), stdout = TRUE)

source(file.path(git_root, "src/libpreprocess.R"))
main(commandArgs(trailingOnly = TRUE))
