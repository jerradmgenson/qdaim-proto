## Copyright 2020, 2021 Jerrad M. Genson

## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at https://mozilla.org/MPL/2.0/.

library(argparser)
library(mice)
library(naniar)

help_text <-
    "Clean, standardize, and impute missing data so that it can be modelled.

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


main <- function(argv) {
    ## Main entry point. When run from the command line, this is the function
    ## that is called.
    cl_args <- parse_command_line(argv)
    set.seed(cl_args$random_state)

    ## Convert optional parameter from NA to NULL if it wasn't given.
    features  <-
        if (!sum(is.na(cl_args$features))) cl_args$features else c()

    impute_methods  <-
        if (!sum(is.na(cl_args$impute_methods))) cl_args$impute_methods else c()

    check_impute_methods(impute_methods,
                         cl_args$impute_missing,
                         cl_args$impute_multiple)

    ## Read all CSV files from the given directory into a single dataframe.
    uci_dataset <- read_dir(cl_args$source,
                            features = features,
                            test_pool = cl_args$test_pool)

    total_source_rows <- nrow(uci_dataset$df) + nrow(uci_dataset$test)
    cat(sprintf("Read %d rows from source datasets\n", total_source_rows))

    ## Make sure the test pool is not empty.
    stopifnot(nrow(uci_dataset$test) > 0)

    ## Remove duplicate values between the dataframes.
    uci_dataset$df <- setdiff(uci_dataset$df, uci_dataset$test)

    unique_rows <- nrow(uci_dataset$df) + nrow(uci_dataset$test)
    cat(sprintf("Omitted %d duplicate rows from source datasets\n",
                total_source_rows - unique_rows))

    ## Remove rows where trestbps is 0.
    trestbps0_rows <- sum(uci_dataset$df$trestbps == 0, na.rm = TRUE)
    trestbps0_rows <- trestbps0_rows + sum(uci_dataset$test$trestbps == 0,
                                           na.rm = TRUE)

    uci_dataset$df <- uci_dataset$df[uci_dataset$df$trestbps != 0, ]
    uci_dataset$test <- uci_dataset$test[uci_dataset$test$trestbps != 0, ]
    cat(sprintf("Omitted %d rows where trestbps is 0\n", trestbps0_rows))

    uci_dataset <- replace_chol0_with_na(uci_dataset)
    estimated_rows <- estimate_total_rows(uci_dataset,
                                          cl_args$impute_missing,
                                          cl_args$impute_multiple)

    ## Calculate number of rows to use for testing and validation.
    test_rows <- ceiling(estimated_rows * cl_args$test_fraction)
    validation_rows <- ceiling(estimated_rows * cl_args$validation_fraction)
    check_test_rows(uci_dataset$test, test_rows)

    ## Omit rows containing only NAs.
    uci_dataset$df <- complete_na_omit(uci_dataset$df)
    uci_dataset$test <- complete_na_omit(uci_dataset$test)

    ## Shuffle the test pool.
    uci_dataset$test <- uci_dataset$test[sample(nrow(uci_dataset$test)), ]

    ## Create the test set.
    packed_data <- split_test_data(uci_dataset, test_rows)
    test_data <- packed_data$test
    nontest_data <- packed_data$nontest
    uci_dataset <- NULL

    ## Now shuffle the nontest dataframe.
    nontest_data <- nontest_data[sample(nrow(nontest_data)), ]
    cat("Shuffled remaining data\n")

    ## Omit rows with multiple NAs unless the option to impute them was given.
    if (!cl_args$impute_multiple) {
        rows_before <- nrow(nontest_data)
        nontest_data <- multi_na_omit(nontest_data)
        rows_after <- nrow(nontest_data)
        cat(sprintf("Omitted %d rows with multiple NAs\n",
                    rows_before - rows_after))
    }

    ## Impute missing values if that option was given.
    if (cl_args$impute_missing || cl_args$impute_multiple) {
        nontest_data <- impute(nontest_data,
                               method = impute_methods,
                               random_state = cl_args$random_state)
    }

    ## Omit any NAs that are still present after imputation.
    nan_count <- sum(!complete.cases(nontest_data))
    if (nan_count > 0) {
        nontest_data <- na.omit(nontest_data)
        cat(sprintf("Omitted %d remaining NAs\n", nan_count))
    }

    ## Convert chest pain to a binary class.
    test_data$cp <- cp_to_binary(test_data$cp)
    nontest_data$cp <- cp_to_binary(nontest_data$cp)
    cat("Converted cp to binary class\n")

    ## Convert resting ECG to a binary class.
    test_data$restecg <- restecg_to_binary(test_data$restecg)
    nontest_data$restecg <- restecg_to_binary(nontest_data$restecg)
    cat("Converted restecg to binary class\n")

    ## Rescale binary/ternary classes to range from -1 to 1.
    test_data$sex <- rescale_binary_list(test_data$sex)
    nontest_data$sex <- rescale_binary_list(nontest_data$sex)
    test_data$exang <- rescale_binary_list(test_data$exang)
    nontest_data$exang <- rescale_binary_list(nontest_data$exang)
    test_data$fbs <- rescale_binary_list(test_data$fbs)
    nontest_data$fbs <- rescale_binary_list(nontest_data$fbs)
    cat("Rescaled binary and ternary classes to have range (-1, 1)\n")

    ## Convert target to the specified classification type.
    test_data <- target_to_class(test_data, cl_args$classification_type)
    nontest_data <- target_to_class(nontest_data, cl_args$classification_type)

    ## Remove duplicates rows created during preprocessing.
    rows_before <- nrow(nontest_data) + nrow(test_data)
    test_data <- remove_duplicates(test_data)
    nontest_data <- remove_duplicates(nontest_data)
    rows_after <- nrow(nontest_data) + nrow(test_data)
    cat(sprintf("Omitted %d duplicate rows created during preprocessing\n",
                rows_before - rows_after))

    ## Split nontest data into training and validation data.
    packed_data <- split_validation(nontest_data, validation_rows)
    training_data <- packed_data$training
    validation_data <- packed_data$validation

    ## Perform sanity checks on the datasets before writing them to disk.
    ## Check that each datasets don't include any duplicate samples.
    stopifnot(!has_duplicates(training_data, validation_data, test_data))

    ## Check that none of the datasets contain NAs.
    stopifnot(!has_nas(training_data))
    stopifnot(!has_nas(validation_data))
    stopifnot(!has_nas(test_data))

    ## Check that testing data was drawn exclusively from the test pool.
    stopifnot(nrow(union(test_data, uci_dataset$test)) == nrow(uci_dataset$test))

    ## Check that the datasets contain the number of samples expected.
    stopifnot(nrow(test_data) == test_rows)
    stopifnot(nrow(validation_data) == validation_rows)

    ## Write datasets to the filesystem.
    write.csv(test_data,
              file = cl_args$testing,
              quote = FALSE,
              row.names = FALSE)

    write.csv(validation_data,
              file = cl_args$validation,
              quote = FALSE,
              row.names = FALSE)

    write.csv(training_data,
              file = cl_args$training,
              quote = FALSE,
              row.names = FALSE)

    cat(sprintf("Wrote testing data with %d rows to %s\n",
                nrow(test_data),
                cl_args$testing))

    cat(sprintf("Wrote validation data with %d rows to %s\n",
                nrow(validation_data),
                cl_args$validation))

    cat(sprintf("Wrote training data with %d rows to %s\n",
                nrow(training_data),
                cl_args$training))

    cat(sprintf("Total samples written in all datasets: %d\n",
                test_rows + validation_rows + nrow(training_data)))
}


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

    help <- "Name of the dataset to draw the test data from."
    parser <- add_argument(parser, "test-pool",
                           help = help)

    help <- "State to initialize random number generators with."
    parser <- add_argument(parser, "--random-state",
                           default = 0,
                           help = help)

    help <-
        "Classification type. Possible values: binary, ternary, multiclass"
    parser <- add_argument(parser, "--classification-type",
                           default = "binary",
                           help = help)

    help <-
        "Fraction of data to use for testing as a real number between 0 and 1."
    parser <- add_argument(parser, "--test-fraction",
                           default = 0.2,
                           help = help)

    help <- "Fraction of data to use for validation as a real number between
0 and 1."
    parser <- add_argument(parser, "--validation-fraction",
                           default = 0.2,
                           help = help)

    parser <- add_argument(parser, "--features",
                           nargs = Inf,
                           help = "Features to select from the input datasets.")

    help <-
        "Impute rows with single NAs in the training and validation datasets."
    parser <- add_argument(parser, "--impute-missing",
                           flag = TRUE,
                           help = help)

    help <-
        "Impute rows with multiple NAs in the training and validation datasets.
 --impute-missing has no effect when --impute-multiple is present."
    parser <- add_argument(parser, "--impute-multiple",
                           flag = TRUE,
                           help = help)

    help <-
        "Methods to use for imputation. Methods must correspond to --features
 (if given) or columns of the input datasets."
    parser <- add_argument(parser, "--impute-methods",
                           nargs = Inf,
                           help = help)

    parse_args(parser, argv = argv)
}


check_impute_methods <- function(impute_methods,
                                 impute_missing,
                                 impute_multiple) {
    ## Check that --impute-methods is not present without --impute-missing
    ## or --impute-multiple.
    if (length(impute_methods) && !impute_missing && !impute_multiple) {
        warning <- "Warning message:\n--impute-methods has no effect if
 --impute-missing or --impute-multiple is not given.\n"
        cat(warning)
    }
}


replace_chol0_with_na <- function(uci_dataset) {
    ## Convert chol values == 0 to NA if chol is present in dataset.
    if ("chol" %in% colnames(uci_dataset$df)) {
        chol0_rows <-
            sum(uci_dataset$df$chol == 0, na.rm = TRUE)

        chol0_rows <-
            chol0_rows + sum(uci_dataset$test$chol == 0, na.rm = TRUE)

        uci_dataset$df <-
            replace_with_na(uci_dataset$df, replace = list(chol = 0))

        uci_dataset$test <-
            replace_with_na(uci_dataset$test, replace = list(chol = 0))

        cat(sprintf("Replaced %d rows where chol is 0 with NA\n", chol0_rows))
    }
    uci_dataset
}


estimate_total_rows <- function(uci_dataset, impute_missing, impute_multiple) {
    ## Estimate the total number of rows not containing NAs that will be present
    ## in the dataset after imputation is performed.
    if (impute_multiple) {
        total_rows <- nrow(uci_dataset$df) + nrow(uci_dataset$test)

    } else if (impute_missing) {
        total_rows <-
            (nrow(multi_na_omit(uci_dataset$df))
                + nrow(multi_na_omit(uci_dataset$test)))

    } else {
        total_rows <-
            nrow(na.omit(uci_dataset$df)) + nrow(na.omit(uci_dataset$test))
    }
    total_rows
}


check_test_rows <- function(test_pool, test_rows) {
    ## Check that the number of test rows doesn't exceed the number of rows in
    ## the test pool.
    test_pool_rows <- nrow(na.omit(test_pool))
    if (test_rows > test_pool_rows) {
        error <-
            "Too few samples in test pool to create test set. Need %d samples
 but only found %d."
        stop(sprintf(error,
                     test_rows,
                     test_pool_rows))
    }
}


split_test_data <- function(uci_dataset, test_rows) {
    ## Sample the specified number of test rows from the test pool, then add the
    ## remaining rows from the test pool to the main dataframe and return both
    ## dataframes as a named list (test, nontest).
    test_indices <- which(rowSums(is.na(uci_dataset$test)) == 0)[1:test_rows]
    test_data <- uci_dataset$test[test_indices, ]
    cat(sprintf("Constructed testing dataset from with %d samples\n",
                test_rows))

    ## Merge remaining samples from test pool into the main dataframe.
    all_indices <- as.numeric(rownames(uci_dataset$test))
    nontest_indices <- setdiff(all_indices, test_indices)
    nontest_data <-
        rbind(uci_dataset$df, uci_dataset$test[nontest_indices, ])

    stopifnot(nrow(intersect(test_data, nontest_data)) == 0)
    list(test = test_data, nontest = nontest_data)
}


read_dir <- function(path, features = c(), test_pool = "") {
    ## Read all CSV files from a directory into a common dataframe.
    ##
    ## Args:
    ##   path: Path to the directory to read from.
    ##   features: The features to select from the input datasets.
    ##             Defaults to all available features.
    ##   test_pool: The dataset to use as the testing set. If given,
    ##             `read_dir` will place this data at the beginning of
    ##             the dataframe. It should be the name of the file
    ##             without the '.csv' extension.
    ##
    ## Returns:p
    ##   A named list with components `df`, a dataframe with the combined
    ##   data from all CSVs in `path` except the test pool, and `test`, which is
    ##   the test pool dataframe.

    csv_files <- dir(path, pattern = ".csv")
    df <- data.frame()
    test <- data.frame()
    for (csv_file in csv_files) {
        full_path <- file.path(path, csv_file)
        data_subset <- utils::read.csv(full_path)
        if (length(features)) {
            data_subset <- data_subset[features]
        }
        test_set_match <- test_pool == unlist(strsplit(csv_file,
                                                       split = ".csv"))

        if (test_set_match) {
            test <- data_subset
        } else if (!nrow(df)) {
            df <- data_subset
        } else {
            df <- rbind(df, data_subset)
        }

    }
    if (!nrow(df)) {
        df <- test[FALSE, ]
    }
    list(df = df, test = test)
}


impute <- function(df, method = c(), random_state = 1) {
    ## Impute missing values in a dataframe using single imputation.
    ##
    ## Args:
    ##   df: Dataframe with missing values to impute.
    ##   method: Method or methods to use for imputation as a character vector.
    ##   random_state: An integer to seed the random number generator with.
    ##
    ## Returns:
    ##   A new dataframe with imputed values.
    cat("Imputing missing data...\n")
    cat(sprintf("NAs before imputation: %d\n",
                sum(!complete.cases(df))))

    imputed_data <- data.frame(df)
    imputed_data$restecg <- as.factor(imputed_data$restecg)
    imputed_data$fbs <- as.factor(imputed_data$fbs)
    if (length(method)) {
        uci_mids <- mice(imputed_data,
                         seed = random_state,
                         method = method,
                         visit = "monotone",
                         maxit = 60,
                         m = 1,
                         print = FALSE)
    } else {
        uci_mids <- mice(imputed_data,
                         seed = random_state,
                         visit = "monotone",
                         maxit = 60,
                         m = 1,
                         print = FALSE)
    }

    imputed_data <- complete(uci_mids, 1)

    ## Convert factor columns back to their original, numeric values.
    imputed_data$restecg <- as.numeric(imputed_data$restecg)
    imputed_data$restecg[imputed_data$restecg == 1] <- 0
    imputed_data$restecg[imputed_data$restecg == 2] <- 1
    imputed_data$restecg[imputed_data$restecg == 3] <- 2
    imputed_data$fbs <- as.numeric(imputed_data$fbs)
    imputed_data$fbs[imputed_data$fbs == 1] <- 0
    imputed_data$fbs[imputed_data$fbs == 2] <- 1

    cat("Imputation complete\n")
    cat(sprintf("NAs after imputation: %d\n",
                sum(!complete.cases(imputed_data))))

    imputed_data
}

cp_to_binary <- function(cp) {
    ## Convert all values in the given cp list to a binary class.
    ## Convert all categories in which chest pain (of any kind) is
    ## present to 1, and categories in which chest pain is absent to -1.
    cp[cp != 4] <- 1
    cp[cp == 4] <- -1
    cp
}

target_to_class <- function(df, class_type) {
    ## Convert all values in the target column of the given dataframe to the
    ## specified classification types ('binary', 'ternary', or 'multiclass').
    if (class_type == "binary") {
        ## Convert target (heart disease class) to a binary class.
        df$target[df$target != 0] <- 1
        df$target[df$target == 0] <- -1

    } else if (class_type == "ternary") {
        ## Convert target to a ternary class.
        df$target[df$target == 0] <- -1
        df$target[df$target == 1] <- 0
        df$target[df$target > 1] <- 1

    } else if (class_type != "multiclass") {
        ## Invalid classification type.
        stop(sprintf("Unknown classification type `%s`.",
                     class_type))
    }
    df
}

split_validation <- function(df, validation_rows) {
    ## Split a dataframe into training and validation sets based on the
    ## given number of validation rows. Return a named list of
    ## (training, validation) dataframes.
    if (validation_rows > 0) {
        validation_data <- df[1:validation_rows, ]
        training_data <- df[(validation_rows + 1):nrow(df), ]

    } else {
        validation_data <- df[FALSE, ]
        training_data <- df
    }
    list(training = training_data, validation = validation_data)
}


has_duplicates <- function(training, validation, testing) {
    combined_data <- data.frame(training)
    combined_data <- rbind(combined_data, validation)
    combined_data <- rbind(combined_data, testing)
    any(duplicated(combined_data))
}


complete_na_omit <- function(df) {
    ## Remove rows that are completely filled with NAs from a dataframe.
    ## Return a new dataframe without the complete NA rows.
    df[rowSums(is.na(df)) < ncol(df), ]
}


restecg_to_binary <- function(restecg) {
    ## Convert a restecg list to binary values by treating all values other than
    ## 1 (ST-T segment abnormalites) as false/-1.
    restecg[restecg != 1] <- -1
    restecg
}


rescale_binary_list <- function(l) {
    ## Rescale a list of values in the range (0, 1) to the range (-1, 1).
    l[l == 0] <- -1
    l
}


remove_duplicates <- function(x) x <- x[!duplicated(x), ]
remove_duplicates <- function(df) df[!duplicated(df), ]
has_nas <- function(df) sum(!complete.cases(df)) > 0
multi_na_omit <- function(df) df[rowSums(is.na(df)) < 2, ]
