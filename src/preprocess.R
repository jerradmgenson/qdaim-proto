#!/usr/bin/Rscript

# Perform preprocessing activities that occur after feature selection.
# Ergo, this script is designed to be run after (and informed by) the
# Feature Selection notebook. The input of this script is the output
# of preprocess_stage1.py. The output of this script is the input of
# gen_model.py.
#
# Preprocessing steps performed by this script include:
# - Discard all rows where trestbps is equal to 0.
# - Discard all rows containing more than one NA.
# - Impute remaining missing values using mice.
# - Convert cp to a binary class.
# - Convert restecg to a binary class.
# - Optionally convert target to a binary or ternary class.
# - Rescale binary and ternary classes to range from -1 to 1.
# - Randomize the row order.
# - Split data into test, train, and validation sets.
#
# Copyright 2020 Jerrad M. Genson
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

library(argparser)
library(mice)

git_root <- system2('git', args=c('rev-parse', '--show-toplevel'), stdout=TRUE)
source(file.path(git_root, "src/util.R"))


parse_command_line <- function(argv) {
    # Parse the command line using argparse.
    #
    # Args
    #  argv: A list of command line arguments, excluding the program name.
    #
    # Returns
    #  The output of parse_args().
    parser <- arg_parser("Stage 2 preprocessor")
    parser <- add_argument(parser, "training",
                           help = "Path to write the training dataset to.")

    parser <- add_argument(parser, "testing",
                           help = "Path to write the test dataset to.")

    parser <- add_argument(parser, "validation",
                           help = "Path to write the validation dataset to.")

    parser <- add_argument(parser, "source",
                           help = "Input directory of CSV data files.")

    parser <- add_argument(parser, "--random-state",
                           default = 0,
                           help = "State to initialize random number generators with.")

    parser <- add_argument(parser, "--classification-type",
                           default = "binary",
                           help = "Classification type. Possible values: 'binary', 'ternary', 'multiclass'")

    parser <- add_argument(parser, "--test-fraction",
                           default = 0.2,
                           help = "Fraction of data to use for testing as a real number between 0 and 1.")

    parser <- add_argument(parser, "--validation-fraction",
                           default = 0.2,
                           help = "Fraction of data to use for validation as a real number between 0 and 1.")

    parser <- add_argument(parser, "--features",
                           nargs = Inf,
                           help = "Features to select from the input datasets.")

    parser <- add_argument(parser, "--test-samples-from",
                           default = "",
                           help = "Name of the dataset to sample test data from. Defaults to all datasets.")

    parse_args(parser, argv = argv)
}


command_line_arguments <- parse_command_line(commandArgs(trailingOnly = TRUE))
set.seed(command_line_arguments$random_state)

# Convert optional parameter from NA to NULL if it wasn't given.
features  <- if (!is.na(command_line_arguments$features)) command_line_arguments$features else NULL

# Read all CSV files from the given directory into a single dataframe.
uci_dataset <- read_dir(command_line_arguments$source,
                        features = features,
                        test_samples_from = command_line_arguments$test_samples_from)

# Remove rows where trestbps is 0.
uci_dataset$df <- uci_dataset$df[uci_dataset$df$trestbps != 0, ]

# Remove rows containing more than one NA.
uci_wo_multi_na_rows <- uci_dataset$df[rowSums(is.na(uci_dataset$df)) < 2, ]

# Impute missing data using single imputation.
uci_wo_multi_na_rows$restecg <- as.factor(uci_wo_multi_na_rows$restecg)
uci_wo_multi_na_rows$fbs <- as.factor(uci_wo_multi_na_rows$fbs)
uci_mids <- mice(uci_wo_multi_na_rows,
                 seed = command_line_arguments$random_state,
                 method = c("", "", "", "", "logreg", "polyreg", "", "", "pmm", ""),
                 visit = "monotone",
                 maxit = 20,
                 m = 1,
                 print = FALSE)

imputed_dataset <- complete(uci_mids, 1)
imputed_dataset$restecg <- as.numeric(imputed_dataset$restecg)
imputed_dataset$fbs <- as.numeric(imputed_dataset$fbs)

# Convert chest pain to a binary class.
imputed_dataset$cp[imputed_dataset$cp != 4] <- 1
imputed_dataset$cp[imputed_dataset$cp == 4] <- -1

# Convert resting ECG to a binary class.
imputed_dataset$restecg[imputed_dataset$restecg != 1] <- -1

# Rescale binary/ternary classes to range from -1 to 1.
imputed_dataset$sex[imputed_dataset$sex == 0] <- -1
imputed_dataset$exang[imputed_dataset$exang == 0] <- -1
imputed_dataset$fbs[imputed_dataset$fbs == 1] <- -1
imputed_dataset$fbs[imputed_dataset$fbs == 2] <- 1

if (command_line_arguments$classification_type == "binary") {
    # Convert target (heart disease class) to a binary class.
    imputed_dataset$target[imputed_dataset$target != 0] <- 1
    imputed_dataset$target[imputed_dataset$target == 0] <- -1

} else if (command_line_arguments$classification_type == "ternary") {
    # Convert target to a ternary class.
    imputed_dataset$target[imputed_dataset$target == 0] <- -1
    imputed_dataset$target[imputed_dataset$target == 1] <- 0
    imputed_dataset$target[imputed_dataset$target > 1] <- 1

} else if (command_line_arguments$classification_type != "multiclass") {
    # Invalid classification type.
    stop(sprintf("Unknown classification type `%s`.",
                 command_line_arguments$classification_type))
}

# Split dataset into test, training, and validation sets.
test_rows <- ceiling(nrow(imputed_dataset)
                        * command_line_arguments$test_fraction)


if (command_line_arguments$test_samples_from != "") {

}

validation_rows <-
    ceiling(nrow(imputed_dataset)
            * command_line_arguments$validation_fraction)

if (command_line_arguments$test_samples_from != "") {
    # Check that the number of test rows doesn't exceed the number of rows in the
    # specified test dataset if one was given.
    original_test_set <- uci_dataset$df[1:uci_dataset$test_rows, ]
    original_test_rows <- nrow(original_test_set[(rowSums(is.na(original_test_set)) < 2), ])
    if (test_rows > original_test_rows) {
        stop(sprintf("Too few samples in %s to create test set.",
                     command_line_arguments$test_samples_from))
    }

    # Sample test data before we shuffle the source dataframe to make
    # sure we sample from the correct dataset.
    test_data <- imputed_dataset[1:test_rows, ]

    # Remove test samples from the source dataframe.
    imputed_dataset <- imputed_dataset[(test_rows + 1):nrow(imputed_dataset), ]

    # Now shuffle the source dataframe.
    imputed_dataset <- imputed_dataset[sample(nrow(imputed_dataset)), ]

} else {
    # Shuffle souce dataframe before sampling test data.
    imputed_dataset <- imputed_dataset[sample(nrow(imputed_dataset)), ]
    test_data <- imputed_dataset[1:test_rows, ]

    # Remove test samples from the source dataframe.
    imputed_dataset <- imputed_dataset[(test_rows + 1):nrow(imputed_dataset), ]
}

validation_data <- imputed_dataset[1:validation_rows, ]
training_data <- imputed_dataset[(validation_rows + 1):nrow(imputed_dataset), ]

# Write datasets to the filesystem.
write.csv(test_data,
          file = command_line_arguments$testing,
          quote = FALSE,
          row.names = FALSE)

write.csv(validation_data,
          file = command_line_arguments$validation,
          quote = FALSE, row.names = FALSE)

write.csv(training_data,
          file = command_line_arguments$training,
          quote = FALSE,
          row.names = FALSE)
