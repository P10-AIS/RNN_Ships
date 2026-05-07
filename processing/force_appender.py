import os
import pickle

import pandas as pd

from config import config
import numpy as np
from force_map import Forcemap


class ForceAppender():
    def __init__(self):
        super().__init__()
        dataset_path = os.path.join(config.crate_dir, "Force")
        if not os.path.exists(dataset_path):
            os.makedirs(dataset_path)
        self.dataset_paths = [os.path.join(dataset_path, dataset)
                              for dataset in config.force_datasets_paths]

        self.ais_dataset_path = os.path.join(
            config.crate_dir, "DatasetTraj", config.ais_datasets_path)
        self.formatted_data: np.ndarray = np.array([])

    def save_formatted_data(self, name: str):
        columns = ['base_datetime', 'lat',
                   'lon', 'sog', 'cog', 'year', 'month']
        for i in range(len(config.force_datasets_paths)):
            columns += [f'water_u_{i}', f'water_v_{i}']

        df = pd.DataFrame(self.formatted_data, columns=columns)
        output_path = os.path.join(
            config.data_directory, "crate_valid", f"{name}_long_term.parquet")
        if not os.path.exists(os.path.dirname(output_path)):
            os.makedirs(os.path.dirname(output_path))
        df.to_parquet(output_path, index=False)

    def split_and_save(self):
        threshold_percent = config.train_val_split_percent
        cutoff = int(len(self.formatted_data) * threshold_percent)

        traj_length = config.min_track_length//config.interpolation_time_gap + 1

        cutoff = (cutoff // traj_length) * traj_length

        train_data = self.formatted_data[:cutoff]
        val_data = self.formatted_data[cutoff:]
        self.formatted_data = train_data
        self.save_formatted_data("train")

        self.formatted_data = val_data
        self.save_formatted_data("valid")

    def load_ais_dataset(self):
        with np.load(self.ais_dataset_path, allow_pickle=True) as data:
            # Shape: [Total_Points, 6] -> [time, lat, lon, cog, sog, type]
            all_points = data['trajectories']
            traj_indices = pickle.loads(data['trajectory_idxes'].item())

        num_trajectories = len(traj_indices)

        for i in range(num_trajectories):
            start_idx = traj_indices[i]
            end_idx = traj_indices[i+1] if i + \
                1 < num_trajectories else len(all_points)

            full_res_segment = all_points[start_idx:end_idx]

            segment = self._interpolate_trajectory(
                full_res_segment, interval_seconds=config.interpolation_time_gap)

            if (len(segment)) < config.min_track_length//config.interpolation_time_gap + 1:
                continue

            timestamps = pd.to_datetime(segment[:, 0], unit='s')

            traj = np.stack([
                segment[:, 0],  # TIMESTAMP
                segment[:, 1],  # LAT
                segment[:, 2],  # LON
                segment[:, 4],  # SOG
                segment[:, 3],  # COG
                timestamps.year,
                timestamps.month,
            ], axis=1)[:config.min_track_length // config.interpolation_time_gap + 1, :]

            self.formatted_data = np.vstack(
                [self.formatted_data, traj]) if self.formatted_data.size else traj

    def append_force_datasets(self):
        for dataset_path in self.dataset_paths:
            force_map = Forcemap.load(dataset_path)
            x, y = force_map.get_grid().coords_to_tiles(
                'epsg:4326', self.formatted_data[:, 2], self.formatted_data[:, 1])

            vec = force_map.get_array()[y, x]
            self.formatted_data = np.concatenate(
                [self.formatted_data, vec[:, 0:1], vec[:, 1:]], axis=1)

    def _interpolate_trajectory(self, segment, interval_seconds=600):
        """
        segment: numpy array [N, 5] -> [time, lat, lon, sog, cog]
        returns: interpolated segment with fixed time intervals
        """

        times = segment[:, 0]
        lat = segment[:, 1]
        lon = segment[:, 2]
        sog = segment[:, 3]
        cog = segment[:, 4]

        sort_idx = np.argsort(times)
        times = times[sort_idx]
        lat = lat[sort_idx]
        lon = lon[sort_idx]
        cog = cog[sort_idx]
        sog = sog[sort_idx]

        start_time = times[0]
        end_time = times[-1]

        new_times = np.arange(start_time, end_time, interval_seconds)

        lat_i = np.interp(new_times, times, lat)
        lon_i = np.interp(new_times, times, lon)
        sog_i = np.interp(new_times, times, sog)

        cog_rad = np.deg2rad(cog)
        sin_cog = np.sin(cog_rad)
        cos_cog = np.cos(cog_rad)

        sin_i = np.interp(new_times, times, sin_cog)
        cos_i = np.interp(new_times, times, cos_cog)

        cog_i = np.rad2deg(np.arctan2(sin_i, cos_i)) % 360

        interpolated = np.stack([
            new_times,
            lat_i,
            lon_i,
            sog_i,
            cog_i,
        ], axis=1)

        return interpolated


if __name__ == "__main__":
    fa = ForceAppender()
    fa.load_ais_dataset()
    fa.append_force_datasets()
    if config.should_split_data:
        fa.split_and_save()
    else:
        fa.save_formatted_data("test")
