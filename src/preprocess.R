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

library(argparser)
library(mice)
library(naniar)

help_text <- "Clean, standardize, and impute missing data so that it can be modelled.

Copyright 2020, 2021 Jerrad M. Genson

This Source Code Form is subject to the terms of the Mozilla Public
License, v. 2.0. If a copy of the MPL was not distributed with this
file, You can obtain one at https://mozilla.org/MPL/2.0/.

Preprocessing steps performed by this script include:

- Omit all rows where trestbps is equal to 0.

- Omit rows containing NA or impute them using mice.

- Convert cp to a binary class.

- Convert restecg to a binary class.

- Optionally convert target to a binary or ternary class.

- Rescale binary and ternary classes to range from -1 to 1.

- Randomize row order.

- Split data into test, train, and validation sets."

git_root <- system2("git",
                    args = c("rev-parse", "--show-toplevel"),
                    stdout = TRUE)

source(file.path(git_root, "src/read_dir.R"))


parse_command_line <- function(argv) {
    ## Parse the command line using argparse.
    ##
    ## Args
    ##  argv: A list of command line arguments, excluding the program name.
    ##
    ## Returns
    ##  The output of parse_args().
    parser <- arg_parser(help_text)
    parser <- add_argument(parser, "training",
                           help = "Path to write the training dataset to.")

    parser <- add_argument(parser, "testing",
                           help = "Path to write the test dataset to.")

    parser <- add_argument(parser, "validation",
                           help = "Path to write the validation dataset to.")

    parser <- add_argument(parser, "source",
                           help = "Input directory of CSV data files.")

    parser <- add_argument(parser, "test-pool",
                           help = "Name of the dataset to draw the test data from.")

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

    parser <- add_argument(parser, "--impute-missing",
                           flag = TRUE,
                           help = "Impute rows with single NAs in the training and validation datasets.")

    parser <- add_argument(parser, "--impute-multiple",
                           flag = TRUE,
                           help = "Impute rows with multiple NAs in the training and validation datasets. --impute-missing has no effect when --impute-multiple is present.")

    parser <- add_argument(parser, "--impute-methods",
                           nargs = Inf,
                           help = "Methods to use for imputation. Methods must correspond to --features (if given) or columns of the input datasets.")

    parse_args(parser, argv = argv)
}


multi_na.omit <- function(df) {
    ## Omit rows from the dataframe, df, that contain more than one NA.
    ## Return a copy of the dataframe without the multiple NA rows.
    df[rowSums(is.na(df)) < 2, ]
}


command_line_arguments <- parse_command_line(commandArgs(trailingOnly = TRUE))
set.seed(command_line_arguments$random_state)

## Convert optional parameter from NA to NULL if it wasn't given.
features  <- if (!is.na(command_line_arguments$features)) {
                 command_line_arguments$features
             } else { NULL }

impute_methods  <- if (!is.na(command_line_arguments$impute_methods)) {
                       command_line_arguments$impute_methods
                   } else { NULL }

if (!is.null(impute_methods)
    && !command_line_arguments$impute_missing
    && !command_line_arguments$impute_multiple) {
    cat("Warning message:\n--impute-methods has no effect if --impute-missing or --impute-multiple is not given.\n")
}

## Read all CSV files from the given directory into a single dataframe.
uci_dataset <- read_dir(command_line_arguments$source,
                        features = features,
                        test_pool = command_line_arguments$test_pool)

source_datasets_rows <- nrow(uci_dataset$df) + nrow(uci_dataset$test)
cat(sprintf("Read %d rows from source datasets\n", source_datasets_rows))
stopifnot(nrow(uci_dataset$test) > 0)

## Remove rows where trestbps is 0.
trestbps0_rows <- sum(uci_dataset$df$trestbps == 0, na.rm = TRUE)
trestbps0_rows <- trestbps0_rows + sum(uci_dataset$test$trestbps == 0,
                                       na.rm = TRUE)

uci_dataset$df <- subset(uci_dataset$df, uci_dataset$df$trestbps != 0)
uci_dataset$test <- subset(uci_dataset$test, uci_dataset$test$trestbps != 0)
cat(sprintf("Omitted %d rows where trestbps is 0\n", trestbps0_rows))

chol_present <- "chol" %in% colnames(uci_dataset$df)
if (chol_present) {
    ## Convert chol values == 0 to NA.
    chol0_rows <- sum(uci_dataset$df$chol == 0, na.rm = TRUE)
    chol0_rows <- chol0_rows + sum(uci_dataset$test$chol == 0, na.rm = TRUE)
    uci_dataset$df <- replace_with_na(uci_dataset$df, replace = list(chol = 0))
    uci_dataset$test <- replace_with_na(uci_dataset$test,
                                        replace = list(chol = 0))

    cat(sprintf("Replaced %d rows where chol is 0 with NA\n", chol0_rows))
}

## Calculate number of rows to use for testing and validation.
if (command_line_arguments$impute_multiple) {
    total_rows <- nrow(uci_dataset$df) + nrow(uci_dataset$test)

} else if (command_line_arguments$impute_missing) {
    total_rows <-
        (nrow(multi_na.omit(uci_dataset$df))
            + nrow(multi_na.omit(uci_dataset$test)))

} else {
    total_rows <-
        nrow(na.omit(uci_dataset$df)) + nrow(na.omit(uci_dataset$test))
}

test_rows <- ceiling(total_rows * command_line_arguments$test_fraction)
validation_rows <-
    ceiling(total_rows * command_line_arguments$validation_fraction)

## Check that the number of test rows doesn't exceed the number of rows in the
## specified test dataset if one was given.
original_test_rows <- nrow(na.omit(uci_dataset$test))
if (test_rows > original_test_rows) {
    stop(sprintf("Too few samples in %s to create test set. Need %d samples but only found %d.",
                 command_line_arguments$test_pool,
                 test_rows,
                 original_test_rows))
}

## Sample test data before we shuffle the source dataframe to make
## sure we sample from the correct dataset.
test_indices <- which(rowSums(is.na(uci_dataset$test)) == 0)[1:test_rows]
test_data <- uci_dataset$test[test_indices, ]
cat(sprintf("Constructed testing dataset from %s data with %d samples\n",
            command_line_arguments$test_pool,
            test_rows))

## Merge remaining samples from test pool into the main dataframe.
all_indices <- as.numeric(rownames(uci_dataset$test))
non_test_indices <- setdiff(all_indices, test_indices)
uci_dataset$df <- rbind(uci_dataset$df, uci_dataset$test[non_test_indices, ])

## Now shuffle the source dataframe.
uci_dataset$df <- uci_dataset$df[sample(nrow(uci_dataset$df)), ]
cat("Shuffled remaining data\n")

if (!command_line_arguments$impute_multiple) {
    rows_before <- nrow(uci_dataset$df)
    uci_dataset$df <- multi_na.omit(uci_dataset$df)
    rows_after <- nrow(uci_dataset$df)
    cat(sprintf("Omitted %d rows with multiple NAs\n",
                rows_before - rows_after))
}

if (command_line_arguments$impute_missing
    || command_line_arguments$impute_multiple) {
    ## Impute missing data using single imputation.
    cat("Imputing missing data...\n")
    cat(sprintf("NAs before imputation: %d\n",
                sum(!complete.cases(uci_dataset$df))))

    uci_dataset$df$restecg <- as.factor(uci_dataset$df$restecg)
    uci_dataset$df$fbs <- as.factor(uci_dataset$df$fbs)
    if (!is.null(impute_methods)) {
        uci_mids <- mice(uci_dataset$df,
                         seed = command_line_arguments$random_state,
                         method = impute_methods,
                         visit = "monotone",
                         maxit = 60,
                         m = 1,
                         print = FALSE)
    } else {
        uci_mids <- mice(uci_dataset$df,
                         seed = command_line_arguments$random_state,
                         visit = "monotone",
                         maxit = 60,
                         m = 1,
                         print = FALSE)
    }

    uci_dataset$df <- complete(uci_mids, 1)
    uci_dataset$df$restecg <- as.numeric(uci_dataset$df$restecg)
    uci_dataset$df$fbs <- as.numeric(uci_dataset$df$fbs)
    cat("Imputation complete\n")
    cat(sprintf("NAs after imputation: %d\n",
                sum(!complete.cases(uci_dataset$df))))
}

nan_count <- sum(!complete.cases(uci_dataset$df))
if (nan_count > 0) {
    uci_dataset$df <- na.omit(uci_dataset$df)
    cat(sprintf("Omitted %d remaining NAs\n", nan_count))
}

## Convert chest pain to a binary class.
uci_dataset$df$cp[uci_dataset$df$cp != 4] <- 1
uci_dataset$df$cp[uci_dataset$df$cp == 4] <- -1
test_data$cp[test_data$cp != 4] <- 1
test_data$cp[test_data$cp == 4] <- -1
cat("Converted cp to binary class\n")

## Convert resting ECG to a binary class.
uci_dataset$df$restecg[uci_dataset$df$restecg != 1] <- -1
test_data$restecg[test_data$restecg != 1] <- -1
cat("Converted restecg to binary class\n")

## Rescale binary/ternary classes to range from -1 to 1.
uci_dataset$df$sex[uci_dataset$df$sex == 0] <- -1
test_data$sex[test_data$sex == 0] <- -1
uci_dataset$df$exang[uci_dataset$df$exang == 0] <- -1
test_data$exang[test_data$exang == 0] <- -1
uci_dataset$df$fbs[uci_dataset$df$fbs == 1] <- -1
uci_dataset$df$fbs[uci_dataset$df$fbs == 2] <- 1
test_data$fbs[test_data$fbs == 1] <- -1
test_data$fbs[test_data$fbs == 2] <- 1
cat("Rescaled binary and ternary classes to have range (-1, 1)\n")

if (command_line_arguments$classification_type == "binary") {
    ## Convert target (heart disease class) to a binary class.
    uci_dataset$df$target[uci_dataset$df$target != 0] <- 1
    uci_dataset$df$target[uci_dataset$df$target == 0] <- -1
    test_data$target[test_data$target != 0] <- 1
    test_data$target[test_data$target == 0] <- -1
    cat("Converted target to binary class\n")

} else if (command_line_arguments$classification_type == "ternary") {
    ## Convert target to a ternary class.
    uci_dataset$df$target[uci_dataset$df$target == 0] <- -1
    uci_dataset$df$target[uci_dataset$df$target == 1] <- 0
    uci_dataset$df$target[uci_dataset$df$target > 1] <- 1
    test_data$target[test_data$target == 0] <- -1
    test_data$target[test_data$target == 1] <- 0
    test_data$target[test_data$target > 1] <- 1
    cat("Converted target to ternary class\n")

} else if (command_line_arguments$classification_type != "multiclass") {
    ## Invalid classification type.
    stop(sprintf("Unknown classification type `%s`.",
                 command_line_arguments$classification_type))
}

if (validation_rows > 0) {
    validation_data <- uci_dataset$df[1:validation_rows, ]
    training_data <-
        uci_dataset$df[(validation_rows + 1):nrow(uci_dataset$df), ]

} else {
    validation_data <- uci_dataset$df[FALSE, ]
    training_data <- uci_dataset$df
}

## Perform sanity checks on the datasets before writing them to disk.
## Check that each dataset is distinct from all other datasets.
distinct <- function(df1, df2) {
    intersect_df <- intersect(df1, df2)
    ## We have to check the dimensions also due to a bug in R.
    ## Sometimes intersect returns garbage on dataframes.
    (nrow(intersect_df) == 0) || (all(dim(df1) != dim(intersect_df)))
}

stopifnot(distinct(training_data, test_data))
stopifnot(distinct(training_data, validation_data))
stopifnot(distinct(validation_data, test_data))

## Check that none of the datasets contain NAs.
stopifnot(sum(!complete.cases(training_data)) == 0)
stopifnot(sum(!complete.cases(validation_data)) == 0)
stopifnot(sum(!complete.cases(test_data)) == 0)

## Check that testing data was drawn exclusively from the test pool.
stopifnot(nrow(union(test_data, uci_dataset$test)) == nrow(uci_dataset$test))

## Check that the datasets contain the number of samples expected.
stopifnot(nrow(test_data) == test_rows)
stopifnot(nrow(validation_data) == validation_rows)

## Write datasets to the filesystem.
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

cat(sprintf("Wrote testing data to %s\n", command_line_arguments$testing))
cat(sprintf("Wrote validation data to %s\n", command_line_arguments$validation))
cat(sprintf("Wrote training data to %s\n", command_line_arguments$training))
cat(sprintf("Total samples written in all datasets: %d\n",
            test_rows + validation_rows + nrow(training_data)))
