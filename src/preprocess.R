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
# - Split data into testing, training, and validation sets.
#
# Copyright 2020 Jerrad M. Genson
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

options(error = traceback)

library(argparser)
library(mice)

git_root <- system2('git', args=c('rev-parse', '--show-toplevel'), stdout=TRUE)
source(file.path(git_root, "src/util.R"))

# Name of the testing dataset
testing_dataset_name <- "testing.csv"

# Name of the training dataset
training_dataset_name <- "training.csv"

# Name of the validation dataset
validation_dataset_name <- "validation.csv"


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
                           help = "Directory of CSV dataset files.")

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

    parser <- add_argument(parser, "--columns",
                           nargs = Inf,
                           help = "Columns to select from the input datasets.")

    parse_args(parser, argv = argv)
}


command_line_arguments <- parse_command_line(commandArgs(trailingOnly = TRUE))
set.seed(command_line_arguments$random_seed)

# Convert optional parameter from NA to NULL if it wasn't given.
columns  <- if (!is.na(command_line_arguments$columns)) command_line_arguments$columns else NULL

# Read all CSV files from the given directory into a single dataframe.
uci_dataset <- read_dir(command_line_arguments$source,
                        columns = columns)

# Remove rows where trestbps is 0.
uci_dataset <- uci_dataset[uci_dataset$trestbps != 0, ]

# Remove rows containing more than one NA.
uci_dataset <- uci_dataset[rowSums(is.na(uci_dataset)) < 2, ]

# Impute missing data using single imputation.
uci_dataset$restecg <- as.factor(uci_dataset$restecg)
uci_dataset$fbs <- as.factor(uci_dataset$fbs)
uci_mids <- mice(uci_dataset,
                 seed = command_line_arguments$random_seed,
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

# Shuffle order of rows in dataset.
imputed_dataset <- imputed_dataset[sample(nrow(imputed_dataset)), ]

# Split dataset into testing, training, and validation sets.
testing_rows <- ceiling(nrow(imputed_dataset)
                        * command_line_arguments$testing_fraction)

validation_rows <-
    ceiling(nrow(imputed_dataset)
            * command_line_arguments$validation_fraction) + testing_rows

testing_data <- imputed_dataset[1:testing_rows, ]
validation_data <- imputed_dataset[(testing_rows + 1):validation_rows, ]
training_data <- imputed_dataset[(validation_rows + 1):nrow(imputed_dataset), ]

# Write datasets to the filesystem.
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
