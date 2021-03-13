## Read all CSV files from a directory into a common dataframe.
##
## Copyright 2020, 2021 Jerrad M. Genson
##
## This Source Code Form is subject to the terms of the Mozilla Public
## License, v. 2.0. If a copy of the MPL was not distributed with this
## file, You can obtain one at https://mozilla.org/MPL/2.0/.

read_dir <- function(path, features = NULL, test_pool = "") {
    ## Read all CSV files from a directory into a common dataframe.
    ## Args:
    ##   path: Path to the directory to read from.
    ##   features: The features to select from the input datasets.
    ##             Defaults to all available features.
    ##   test_pool: The dataset to use as the testing set. If given,
    ##             `read_dir` will place this data at the beginning of
    ##             the dataframe. It should be the name of the file
    ##             without the '.csv' extension.
    ##
    ## Returns:
    ##   A named list with components `df`, a dataframe with the combined
    ##   data from all CSVs in `path` except the test pool, and `test`, which is
    ##   the test pool dataframe.

    csv_files <- dir(path, pattern = ".csv")
    df <- data.frame()
    test <- data.frame()
    for (csv_file in csv_files) {
        full_path <- file.path(path, csv_file)
        data_subset <- utils::read.csv(full_path)
        if (!is.null(features)) {
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
