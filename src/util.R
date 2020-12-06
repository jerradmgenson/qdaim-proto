# A collection of various utility functions that are not related to the core
# algorithms or another, more specific library. At the present time, this
# includes a function for reading all csv files from a directory into a
# common dataframe.
#
# Copyright 2020 Jerrad M. Genson
#
# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.

options(error = traceback)

read_dir <- function(path, columns = NULL, test_set = "") {
    # Read all CSV files in a directory into a common dataframe.
    # Args:
    #   path: Path to the directory to read from.
    #   columns: The columns to select from the input datasets.
    #            Defaults to all available columns.
    #   test_set: The dataset to use as the testing set. If given,
    #             `read_dir` will place this data at the beginning of
    #             the dataframe. It should be the name of the file
    #             without the '.csv' extension.

    # Returns:
    #   A named list with components `df`, a dataframe with the combined
    #   data from all CSVs in `path`, and `test_rows`, which is the
    #   number of rows in the input test dataset.

    csv_files <- dir(path, pattern=".csv")
    df <- NULL
    test_rows <- 0
    for (csv_file in csv_files) {
        full_path <- file.path(path, csv_file)
        data_subset <- utils::read.csv(full_path)
        if (!is.null(columns)) {
            data_subset <- data_subset[columns]
        }
        if (is.null(df)) {
            df <- data_subset
        } else if (test_set != "" & (test_set == unlist(strsplit(csv_file, split = ".csv"))[1])) {
            df <- rbind(data_subset, df)
            test_rows <- length(data_subset)
        } else {
            df <- rbind(df, data_subset)
        }
    }
    list(df = df, test_rows = test_rows)
}
