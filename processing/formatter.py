import argparse
import logging
import os
import dask
import gc

import dask.dataframe as dd
import numpy as np
import pandas as pd
from dask_ml.preprocessing import OneHotEncoder

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep
from utils import clear_path


class Formatter(ProcessingStep):
    """
    Class for performing the final processing step, whereby the data is saved as numpy arrays which can easily be read
    in for training and evaluation.
    """

    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name='windowed_with_currents_stride_3' +
            ('_debug' if args.debug else ''),
            to_name='formatted' +
            ('_debug' if args.debug else '')
        )

        self.from_dir = os.path.join(config.data_directory, "crate_valid")
        self.to_dir = os.path.join(config.data_directory, "crate_formatted")

        self._initialize_logging(args.save_log, 'format_with_weather_and_time')

        logging.info(
            f'categorical_columns used are {config.categorical_columns}')
        self.dataset_names = [file_name.split('.')[0] for file_name in os.listdir(
            self.from_dir) if file_name.endswith('.parquet')]
        self.timesteps_into_the_future = None

    def load(self):
        """
        Load the test, train, and validation sets

        This function will also repartition the datasets to make the partition sizes manageable

        :return:
        """
        for dataset_name in self.dataset_names:
            dataset_path = os.path.join(
                self.from_dir, f'{dataset_name}.parquet')
            self.datasets[dataset_name] = dd.read_parquet(dataset_path)
            tmp_dir = os.path.join(
                self.from_dir, f'.tmp_{dataset_name}.parquet')
            if not os.path.exists(tmp_dir):
                self._repartition_ds(dataset_name)
            self.datasets[dataset_name] = dd.read_parquet(tmp_dir)

        logging.info('File paths have been specified for dask')

    def save(self):
        """
        Save train/valid/test datasets safely using partition-wise processing.

        Reshape is performed here per-partition rather than lazily via map_partitions,
        which avoids materialising the full computation graph (one-hot + reshape) at once
        and keeps peak memory to a single partition at a time.
        """
        clear_path(self.to_dir)
        os.mkdir(self.to_dir)

        # Save feature metadata
        self.features = self.features.iloc[:9]
        self.features.to_csv(os.path.join(self.to_dir, 'features.csv'))

        for dataset_name in self.dataset_names:

            logging.info(f"Saving dataset: {dataset_name}")

            set_dir = os.path.join(self.to_dir, dataset_name)
            os.mkdir(set_dir)

            df = self.datasets[dataset_name].iloc[:, :9]

            if 'long_term' in dataset_name:
                timesteps_into_the_future = self.timesteps_into_the_future
            else:
                raise ValueError('Unknown Dataset')

            for new_time_gap in config.time_gaps:

                num_minutes = int(new_time_gap / 60)

                logging.info(
                    f"Processing {dataset_name} | {num_minutes} min gap (partition mode)"
                )

                # Index selection
                step = int(new_time_gap / config.interpolation_time_gap)

                x_idxs = np.arange(
                    0,
                    config.length_of_history,
                    step,
                    dtype=int
                )

                total_timesteps = (
                    config.length_of_history + config.length_into_the_future + 1
                )

                next_gap = config.length_of_history + step - 1

                y_idxs = np.arange(
                    next_gap,
                    total_timesteps,
                    step,
                    dtype=int
                )

                # Output dirs
                x_dir = os.path.join(set_dir, f'{num_minutes}_min_time_gap_x')
                y_dir = os.path.join(set_dir, f'{num_minutes}_min_time_gap_y')

                os.mkdir(x_dir)
                os.mkdir(y_dir)

                npart = df.npartitions
                x_len = 0
                y_len = 0

                for i in range(npart):

                    logging.info(
                        f"{dataset_name} | gap {num_minutes} | partition {i+1}/{npart}")

                    # Compute ONE partition as a pandas DataFrame
                    part_df = df.partitions[i].compute()

                    # Reshape from flat 2D DataFrame to 3D numpy array here,
                    # instead of via map_partitions, to avoid the meta mismatch
                    # error and to keep peak memory low (one partition at a time)
                    part = self._reshape_partition(
                        part_df, timesteps_into_the_future)
                    del part_df
                    gc.collect()

                    # part is now a 3D numpy array: (samples, timesteps, features)
                    X = part[:, x_idxs, :].astype(np.float32)
                    Y = part[:, y_idxs, :].astype(np.float32)
                    del part
                    gc.collect()

                    np.save(os.path.join(x_dir, f'{i}.npy'), X)
                    np.save(os.path.join(y_dir, f'{i}.npy'), Y)

                    x_len += X.shape[0]
                    y_len += Y.shape[0]

                    del X, Y
                    gc.collect()

                logging.info(
                    f"{dataset_name} | {num_minutes} min gap -> X samples: {x_len}, Y samples: {y_len}"
                )

            logging.info(f"{dataset_name} saved to {set_dir}")

            # Free this dataset's memory before moving to the next
            del self.datasets[dataset_name]
            gc.collect()

        self._clear_tmp_files()

    def _reshape_partition(self, partition: pd.DataFrame, into_the_future: int) -> np.ndarray:
        """
        Reshape partition so that it can be easily used by Keras.

        Creates a 3D array (samples, timesteps, features) from a flat 2D DataFrame.

        :param partition: pandas DataFrame for one partition
        :param into_the_future: number of future timesteps
        :return: numpy array of shape (samples, timesteps, features)
        """
        arr = partition.to_numpy()
        num_ts = config.length_of_history + into_the_future
        arr = arr[[range(idx, idx + num_ts)
                   for idx in range(0, len(arr), num_ts)]]
        return np.stack(arr)

    def _one_hot(self, dataset_name):
        """
        One hot encode categorical variables

        :param dataset_name: Whether this is the training, testing, or validation set
        :return: The one hot encoder object used for the transformation
        """
        # Convert dtypes to category with dask
        for col in config.categorical_columns:
            self.datasets[dataset_name][col] = self.datasets[dataset_name][col].astype(
                'category')

        # Identify the category values
        self.datasets[dataset_name] = self.datasets[dataset_name].categorize()

        # Based on the category values, fit a onehotencoder object
        encoder = OneHotEncoder()
        encoder = encoder.fit(
            self.datasets[dataset_name][config.categorical_columns])

        # Define a constant order for the categories
        if not hasattr(self, 'column_order'):
            self.column_order = []
            for col, levels in zip(config.categorical_columns, encoder.categories_):
                for level in levels:
                    self.column_order += [f'{col}_{level}']

        # Transform the dataset using the one hot encoder, and place columns in correct order
        transformed = encoder.transform(
            self.datasets[dataset_name][config.categorical_columns])
        transformed = transformed[self.column_order].astype('bool')

        # Log the categories
        for col in config.categorical_columns:
            categories = self.datasets[dataset_name][col].cat.categories.to_list(
            )
            logging.info(
                f'Using {len(categories)} values for column {col} in {dataset_name} set: {categories}')

        # Add one hot encoded versions to dataset, drop originals
        self.datasets[dataset_name] = self.datasets[dataset_name].drop(
            config.categorical_columns, axis=1)
        self.datasets[dataset_name] = dd.concat(
            [self.datasets[dataset_name], transformed], axis=1)
        return encoder

    def calculate(self):
        """
        Iterate through the datasets, performing one hot encoding.

        NOTE: _reshape_partition is intentionally NOT applied here via map_partitions.
        Doing so caused two problems:
          1. meta mismatch — map_partitions expected a DataFrame but got a numpy array,
             triggering 'DataFrame has no attribute name'
          2. memory spike — chaining one-hot + reshape in a single compute() call
             materialised everything at once and caused OOM kills
        The reshape is instead applied per-partition inside save().
        """
        for dataset_name in self.dataset_names:
            if config.categorical_columns:
                self._one_hot(dataset_name)

            self.features = self.datasets[dataset_name].dtypes.astype(str)
            self.features = self.features.replace(
                'Sparse[bool, False]', 'bool')
            self.features.name = 'dtype'
            self.features.index.name = 'column'

            if 'long_term' in dataset_name:
                self.timesteps_into_the_future = config.length_into_the_future + 1
            else:
                raise ValueError('Unknown Dataset')

            # map_partitions for _reshape_partition removed — see docstring above

        logging.info('Calculation methods have been defined for Dask')

    def _repartition_ds(self, dataset_name):
        """
        Change partition sizes to make sure each one can fit in memory at once.

        500MB per partition is a safe upper bound; actual peak memory per partition
        will be higher due to one-hot expansion and numpy stacking, but manageable.

        :param dataset_name: Dataset to use
        :return:
        """
        tmp_dir = os.path.join(self.from_dir, f'.tmp_{dataset_name}.parquet')

        df = self.datasets[dataset_name]
        df = df.repartition(partition_size="500MB")
        self.datasets[dataset_name] = df

        dd.to_parquet(df, tmp_dir, schema="infer")
        del self.datasets[dataset_name]

    def _clear_tmp_files(self):
        """
        Once dataset has been saved, clear any temporary files that are on disk.
        """
        for dataset_name in self.dataset_names:
            tmp_dir = os.path.join(
                self.from_dir, f'.tmp_{dataset_name}.parquet')
            clear_path(tmp_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()
    config.set_log_level(args.log_level)
    config.dataset_config = datasets[args.dataset_name]

    dask.config.set(scheduler='single-threaded')

    formatter = Formatter()
    formatter.load()
    formatter.calculate()
    formatter.save()
