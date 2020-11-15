#!/usr/bin/Rscript

# Perform preprocessing activities that occur after feature selection.
# Ergo, this script is designed to be run after (and informed by) the
# Feature Selection notebook. The input of this script is the output
# of preprocess_stage1.py. The output of this script is the input of
# gen_model.py.
#
# Preprocessing steps performed by this script include:
# - Discard all columns except those in subset_columns.
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

options(error = traceback)

library(argparser)

# Name of the testing dataset
testing_dataset_name <- "testing.csv"

# Name of the training dataset
training_dataset_name <- "training.csv"

# Name of the validation dataset
validation_dataset_name <- "validation.csv"

# Columns to subset from the original input dataset
subset_columns <- c("age", "sex", "cp", "trestbps", "restecg",
                    "fbs", "thalach", "exang", "oldpeak", "target")


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
                           help = "Path of the directory to output the result of Stage 2 preprocessing.")

    parser <- add_argument(parser, "source",
                           help = "CSV dataset file output by preprocess_stage1.py.")

    parser <- add_argument(parser, "--random-seed",
                           default = 1,
                           help = "Set the number number generator seed.")

    parser <- add_argument(parser, "--classification-type",
                           default = "binary",
                           help = "Classification type. Possible values: 'binary', 'ternary', 'multiclass'")

    parser <- add_argument(parser, "--testing-fraction",
                           default = 0.2,
                           help = "Fraction of data to use for testing as a real number between 0 and 1.")

    parser <- add_argument(parser, "--validation-fraction",
                           default = 0.2,
                           help = "Fraction of data to use for validation as a real number between 0 and 1.")

    parse_args(parser, argv = argv)
}


command_line_arguments <- parse_command_line(commandArgs(trailingOnly = TRUE))
set.seed(command_line_arguments$random_seed)
dataset <- read.csv(command_line_arguments$source)
data_subset <- dataset[, subset_columns]
data_subset <- na.omit(data_subset)
data_subset <- data_subset[data_subset$trestbps != 0, ]

# Convert chest pain to a binary class.
data_subset$cp[data_subset$cp != 4] <- 1
data_subset$cp[data_subset$cp == 4] <- -1

# Convert resting ECG to a binary class.
data_subset$restecg[data_subset$restecg != 1] <- -1

# Rescale binary/ternary classes to range from -1 to 1.
data_subset$sex[data_subset$sex == 0] <- -1
data_subset$exang[data_subset$exang == 0] <- -1
data_subset$fbs[data_subset$fbs == 0] <- -1

if (command_line_arguments$classification_type == "binary") {
    # Convert target (heart disease class) to a binary class.
    data_subset$target[data_subset$target != 0] <- 1
    data_subset$target[data_subset$target == 0] <- -1

} else if (command_line_arguments$classification_type == "ternary") {
    # Convert target to a ternary class.
    data_subset$target[data_subset$target == 0] <- -1
    data_subset$target[data_subset$target == 1] <- 0
    data_subset$target[data_subset$target > 1] <- 1

} else if (command_line_arguments$classification_type != "multiclass") {
    # Invalid classification type.
    stop(sprintf("Unknown classification type `%s`.",
                 command_line_arguments$classification_type))
}

# Shuffle order of rows in dataset.
data_subset <- data_subset[sample(nrow(data_subset)), ]

testing_rows <- ceiling(nrow(data_subset)
                        * command_line_arguments$testing_fraction)

validation_rows <-
    ceiling(nrow(data_subset)
            * command_line_arguments$validation_fraction) + testing_rows

testing_data <- data_subset[1:testing_rows, ]
validation_data <- data_subset[(testing_rows + 1):validation_rows, ]
training_data <- data_subset[(validation_rows + 1):nrow(data_subset), ]

testing_path <- file.path(command_line_arguments$target, testing_dataset_name)
write.csv(testing_data, file = testing_path, quote = FALSE, row.names = FALSE)
validation_path <- file.path(command_line_arguments$target,
                             validation_dataset_name)

write.csv(validation_data,
          file = validation_path,
          quote = FALSE, row.names = FALSE)

training_path <- file.path(command_line_arguments$target,
                           training_dataset_name)

write.csv(training_data,
          file = training_path,
          quote = FALSE,
          row.names = FALSE)
