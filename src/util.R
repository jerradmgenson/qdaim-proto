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


read_dir <- function(path) {
    # Read all CSV files in a directory into a common dataframe.
    # Args:
    #   path: Path to the directory to read from.

    # Returns:
    #   A dataframe with the combined data from all CSVs in `path`.

    csv_files <- dir(path, pattern=".csv")
    df <- NULL
    for (csv_file in csv_files) {
        full_path <- file.path(path, csv_file)
        data_subset <- utils::read.csv(full_path)
        if (is.null(df)) {
            df <- data_subset
        } else {
            df <- rbind(df, data_subset)
        }
    }
    df
}
