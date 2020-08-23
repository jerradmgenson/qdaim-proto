import argparse
import logging
import pickle

import pandas as pd
from sklearn import svm


DATASET_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'fbs', 'restecg', 'thalach', 'exang', 'oldpeak', 'slope', 'ca', 'thal', 'target']
DEFAULT_COLUMNS = ['age', 'sex', 'cp', 'trestbps', 'chol', 'restecg']


def main():
    command_line_arguments = parse_command_line()
    logger = configure_logging(command_line_arguments.log_level)
    logger.info('Loading cardiac dataset from %s.', command_line_arguments.input_path)
    cardiac_dataset = pd.read_csv(command_line_arguments.input_path)
    logger.info('Done loading cardiac dataset.')
    logger.info('Training heart disease model on cardiac dataset.')
    if command_line_arguments.columns:
        reduced_dataset = cardiac_dataset[command_line_arguments.columns + ['target']]

    else:
        reduced_dataset = cardiac_dataset[DEFAULT_COLUMNS + ['target']]

    shuffled_dataset = reduced_dataset.sample(frac=1)
    input_data = shuffled_dataset.values[:, 0:-1]
    target_data = shuffled_dataset.values[:, -1]
    training_rows = int(len(input_data) * (1 - command_line_arguments.testing_fraction))
    training_inputs = input_data[:training_rows]
    training_targets = target_data[:training_rows]
    testing_inputs = input_data[training_rows:]
    testing_targets = target_data[training_rows:]
    heart_disease_model = train_model(training_inputs, training_targets)
    logger.info('Done training heart disease model.')
    logger.info('Validating heart disease model.')
    validate_model(heart_disease_model, testing_inputs, testing_targets)
    logger.info('Done validating heart disease model.')
    logger.info('Accuracy: %f  Precision: %f Sensitivity: %f Specificity: %f',
                heart_disease_model.accuracy,
                heart_disease_model.precision,
                heart_disease_model.sensitivity,
                heart_disease_model.specificity)

    logger.info('Saving heart disease model to %s.', command_line_arguments.output_path)
    save_model(heart_disease_model, command_line_arguments.output_path)
    logger.info('Heart disease model saved successfully.')


def parse_command_line():
    parser = argparse.ArgumentParser(description='Build a machine learning model to predict heart disease.')
    parser.add_argument('input_path', help='Path to the input cardiac dataset.')
    parser.add_argument('output_path', help='Path to output the heart disease model to.')
    parser.add_argument('--testing_fraction',
                        type=testing_fraction,
                        default=0.2,
                        help='The fraction of the dataset to use for testing as a decimal between 0 and 1.')

    parser.add_argument('--columns',
                        type=heart_disease_data_column,
                        nargs='+',
                        help='Columns in the heart disease dataset to use as inputs to the model.')

    parser.add_argument('--log_level',
                        choices=('critical', 'error', 'warning', 'info', 'debug'),
                        default='info',
                        help='Log level to configure logging with.')

    return parser.parse_args()


def testing_fraction(n):
    x = float(n)
    if x < 0 or x > 1:
        raise ValueError('testing_fraction must not be less than 0 or greater than 1.')

    return x


def heart_disease_data_column(n):
    if n not in DATASET_COLUMNS:
        raise ValueError('column must be one of {}'.format(DATASET_COLUMNS))

    return n


def configure_logging(log_level):
    log_level_number = getattr(logging, log_level.upper())
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level_number)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level_number)
    formatter = logging.Formatter('%(levelname)s: %(message)s')
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    return logger


def train_model(input_data, target_data):
    classifier = svm.SVC()
    classifier.fit(input_data, target_data)

    return classifier


def validate_model(model, input_data, target_data):
    predictions = model.predict(input_data)
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0
    for prediction, target in zip(predictions, target_data):
        if prediction == 1 and target == 1:
            true_positives += 1

        elif prediction == 1 and target == 0:
            false_positives += 1

        elif prediction == 0 and target == 1:
            false_negatives += 1

        else:
            true_negatives += 1

    model.accuracy = (true_positives + true_negatives) / len(input_data)
    model.precision = true_positives / (true_positives + false_positives)
    model.sensitivity = true_positives / (true_positives + false_negatives)
    model.specificity = true_negatives / (true_negatives + false_positives)


def save_model(model, output_path):
    with open(output_path, 'wb') as output_file:
        pickle.dump(model, output_file)


if __name__ == '__main__':
    main()
