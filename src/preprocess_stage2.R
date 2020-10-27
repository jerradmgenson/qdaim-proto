#!/usr/bin/Rscript

# Perform preprocessing activities that occur after feature selection.
# Ergo, this script is designed to be run after (and informed by) the
# Feature Selection notebook. The input of this script is the output
# of preprocess_stage1.py. The output of this script is the input of
# gen_model.py.
#
# Preprocessing steps performed by this script include:
# - Discard all columns except those in SUBSET_COLUMNS.
# - Discard all rows where trestbps is equal to 0.
# - Convert cp to a binary class.
# - Convert restecg to a binary class.
# - Optionally convert target to a binary or ternary class.
# - Rescale binary and ternary classes to range from -1 to 1.
# - Randomize the row order.
# - Split data into testing, training, and validation sets.
#
# Copyright 2020 Jerrad M. Genson
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

options(error=traceback)

library(argparser)
library(tidyverse)

# Path to the root of the git repository
GIT_ROOT <- system2("git", args=c("rev-parse", "--show-toplevel"), stdout=TRUE)

# Name of the testing dataset
TESTING_DATASET_NAME <- "testing.csv"

# Name of the training dataset
TRAINING_DATASET_NAME <- "training.csv"

# Name of the validation dataset
VALIDATION_DATASET_NAME <- "validation.csv"

# Columns to subset from the original input dataset
SUBSET_COLUMNS <- c("age", "sex", "cp", "thalrest", "trestbps", "restecg",
                    "fbs", "thalach", "exang", "oldpeak", "target")

# Fraction of data to use for testing as a real number between 0 and 1
TESTING_FRACTION <- 0.2

# Fraction of data to use for validation as a real number between 0 and 1
VALIDATION_FRACTION <- 0.2

# Integer to use for seeding the random number generator
RANDOM_SEED <- 667252912

# Enumerates possible values for 'CLASSIFICATION_TYPE'.
CLASSIFICATION_TYPES <- list(BINARY=0, TERNARY=1, MULTICLASS=2)

# Set what type of classification target to generate.
# Possible values are the members of ClassificationType.
CLASSIFICATION_TYPE <- CLASSIFICATION_TYPES$BINARY


parse_command_line <- function(argv) {
    # Parse the command line using argparse.
    #
    # Args
    #  argv: A list of command line arguments, excluding the program name.
    #
    # Returns
    #  The output of parse_args().
    parser <- arg_parser("Stage 2 preprocessor")
    parser <- add_argument(parser, "target",
                           help="Path of the directory to output the result of Stage 2 preprocessing.")

    parser <- add_argument(parser, "source",
                           help="CSV dataset file output by preprocess_stage1.py.")

    parse_args(parser, argv=argv)
}


set.seed(RANDOM_SEED)
command_line_arguments <- parse_command_line(commandArgs(trailingOnly=TRUE))
dataset <- read_csv(command_line_arguments$source)
data_subset <- dataset[, SUBSET_COLUMNS]
data_subset <- drop_na(data_subset)
data_subset <- data_subset[data_subset$trestbps != 0,]

# Convert chest pain to a binary class.
data_subset$cp[data_subset$cp != 4] <- 1
data_subset$cp[data_subset$cp == 4] <- -1

# Convert resting ECG to a binary class.
data_subset$restecg[data_subset$restecg != 1] <- -1

# Rescale binary/ternary classes to range from -1 to 1.
data_subset$sex[data_subset$sex == 0] <- -1
data_subset$exang[data_subset$exang == 0] <- -1
data_subset$fbs[data_subset$fbs == 0] <- -1

if (CLASSIFICATION_TYPE == CLASSIFICATION_TYPES$BINARY) {
    # Convert target (heart disease class) to a binary class.
    data_subset$target[data_subset$target != 0] <- 1
    data_subset$target[data_subset$target == 0] <- -1

} else if (CLASSIFICATION_TYPE == CLASSIFICATION_TYPES$TERNARY) {
    # Convert target to a ternary class.
    data_subset$target[data_subset$target == 0] <- -1
    data_subset$target[data_subset$target == 1] <- 0
    data_subset$target[data_subset$target > 1] <- 1

} else if (CLASSIFICATION_TYPE != CLASSIFICATION_TYPES$MULTICLASS) {
    # Invalid classification type.
    stop(sprintf("Unknown classification type `%s`.", CLASSIFICATION_TYPE))
}

# Shuffle order of rows in dataset.
data_subset <- data_subset[sample(nrow(data_subset)),]

testing_rows <- ceiling(nrow(data_subset) * TESTING_FRACTION) - 1
validation_rows <- ceiling(nrow(data_subset) * VALIDATION_FRACTION) + testing_rows - 1
testing_data <- slice(data_subset, 1:testing_rows)
validation_data <- slice(data_subset, testing_rows:validation_rows)
training_data <- slice(data_subset, validation_rows:nrow(data_subset))

testing_path <- file.path(command_line_arguments$target, TESTING_DATASET_NAME)
write_csv(testing_data, testing_path)
validation_path <- file.path(command_line_arguments$target, VALIDATION_DATASET_NAME)
write_csv(validation_data, validation_path)
training_path <- file.path(command_line_arguments$target, TRAINING_DATASET_NAME)
write_csv(training_data, training_path)
