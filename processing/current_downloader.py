import xarray as xr
import argparse
import logging
import os
import numpy as np

import pandas as pd
from pydap.client import open_url

from config import config
from config.dataset_config import datasets
from processing_step import ProcessingStep


class Downloader(ProcessingStep):
    """
    Class for downloading ocean current data from NOAA
    """

    def __init__(self):
        super().__init__()
        self._define_directories(
            from_name=None,
            to_name='ocean_current_downloads'
        )
        self._initialize_logging(args.save_log, 'ocean_current_download')

    def _get_hycom_region(self):
        """
        Find which weather region to download data from

        The coordinates for the weather regions are given through the link below
        https://www.ncei.noaa.gov/products/weather-climate-models/frnmoc-navy-global-hybrid-ocean

        :return:
        """
        ranges = {
            # RegionNum: [[Lat_min, lat_max], [lon_min, lon_max]]
            1: [(0.0, 70.0), (-99.99996948242188, -50.0)],
            6: [(10.0, 70.0), (-150.00001525878906, -210.0)],
            7: [(10.0, 60.0), (-149.99996948242188, -100.0)],
            17: [(60.0, 80.0), (-179.99996948242188, -120.0)]
        }

        for region_num, [(lat_min, lat_max), (lon_min, lon_max)] in ranges.items():
            if config.dataset_config.lat_1 >= lat_min and config.dataset_config.lat_2 <= lat_max:
                if config.dataset_config.lon_1 >= lon_min and config.dataset_config.lon_2 <= lon_max:
                    return region_num
        raise ValueError("Regional weather data not available for the lat/lon coordinates chosen. They may be "
                         "available in HYCOM's global surface currents dataset, which uses a slightly different"
                         "url format. See the link in the docstring to amend the code for that dataset. ")

    def _define_directories(self, from_name, to_name):
        """
        Save file paths to directory as member variable

        Override of the ProcessingStep's _define_directories, as the downloader does not have a from_dir.

        :param from_name: Should always be None, but included to keep signature in line with ProcessingStep's method
        :param to_name: Should always be 'ocean_current_downloads', but included to keep signature in line with
                        ProcessingStep's method
        :return:
        """
        self.box_and_year_dir = os.path.join(
            config.data_directory,
            f'{config.dataset_config.lat_1}_{config.dataset_config.lat_2}_'
            f'{config.dataset_config.lon_1}_{config.dataset_config.lon_2}_'
            f'{config.start_year}_{config.end_year}'
        )
        self.from_dir = from_name
        self.to_dir = os.path.join(self.box_and_year_dir, to_name)
        self.artifact_directory = os.path.join(
            self.box_and_year_dir, 'artifacts')

        self._create_directories()

    def _get_map_idxs(self, dataset, variable, map):
        if map == 'time':
            time_offset = pd.to_datetime(
                dataset['time'].units.replace('hours since ', ''))
            min = pd.to_datetime(
                f'{config.start_year}-01-01').tz_localize(time_offset.tzname())
            max = pd.to_datetime(
                f'{config.end_year + 1}-01-01').tz_localize(time_offset.tzname())
        else:
            max = getattr(config.dataset_config, f'{map}_2')
            min = getattr(config.dataset_config, f'{map}_1')

        if map in ['lat', 'lon']:
            lat_lon_margin = 1
            max += lat_lon_margin
            min -= lat_lon_margin
        if hasattr(dataset[map], 'modulo'):
            if dataset[map].modulo == '360 degrees':
                max %= 360
                min %= 360
            else:
                raise ValueError(
                    f'Unknown module: {dataset[variable][map].modulo} for dataset with id {dataset.id}')

        map_vals = np.array(dataset[map][:])
        if map == 'time':
            map_vals = pd.to_datetime(
                [time_offset + pd.Timedelta(hours=h) for h in map_vals])

        idxs = np.where((map_vals <= max) & (map_vals >= min))[0]

        if len(idxs) == 0:
            raise ValueError(
                'No surface current observations are in target range')

        if len(idxs) > 1:
            continuous = len(np.unique(idxs[1:] - idxs[:-1])) == 1
            if not continuous:
                raise ValueError(
                    f'Slice for map {map} with dataset {dataset.id} is not continuous')

        min_idx = idxs.min()
        max_idx = idxs.max()

        map_vals = map_vals[(map_vals <= max) & (map_vals >= min)]
        return min_idx, max_idx, map_vals

    import os

    def download(self):
        logging.info('Starting downloads')

        url = "https://tds.hycom.org/thredds/dodsC/GLBy0.08/expt_93.0/uv3z/2019"

        # Open lazily (important: no download yet)
        ds = xr.open_dataset(url, decode_times=False)

        # --- Get index ranges (adapt this to your bounding box logic) ---
        lat = ds['lat'].values
        lon = ds['lon'].values
        time = ds['time'].values

        time_offset = pd.to_datetime(
            ds['time'].units.replace('hours since ', ''))

        # Use your existing function
        time_min, time_max, time_vals = self._get_map_idxs(
            ds, 'water_u', 'time')
        depth_min, depth_max, depth_vals = self._get_map_idxs(
            ds, 'water_u', 'depth')
        lat_min, lat_max, lat_vals = self._get_map_idxs(ds, 'water_u', 'lat')
        lon_min, lon_max, lon_vals = self._get_map_idxs(ds, 'water_u', 'lon')

        # Only surface
        depth_idx = 0

        # Chunking (same idea as before)
        chunk_size = 100  # safer than 1000 for HYCOM
        time_slices = np.arange(time_min, time_max + 1, chunk_size)

        for var in ["water_v", "water_u"]:
            for j, t0 in enumerate(time_slices):
                if j == 3:
                    break
                t1 = min(time_max, t0 + chunk_size - 1)

                logging.info(f"Downloading {var} [{t0}:{t1}]")

                # --- Subset (this triggers OPeNDAP request only when loaded) ---
                subset = ds[var].isel(
                    time=slice(t0, t1 + 1),
                    depth=depth_idx,
                    lat=slice(lat_min, lat_max + 1),
                    lon=slice(lon_min, lon_max + 1)
                )

                # Load into memory (actual download)
                data = subset.load()

                # Convert to dataframe
                df = data.to_dataframe().reset_index()

                # Rename to match your previous format
                df = df.rename(columns={
                    "lat": "latitude",
                    "lon": "longitude",
                    var: "speed"
                })

                # Fix longitude if needed
                if (df['longitude'] > 180).all():
                    df['longitude'] -= 360

                df['time'] = pd.to_datetime(
                    [time_offset + pd.Timedelta(hours=h) for h in df["time"]])

                # Add time breakdown columns
                df['year'] = df['time'].dt.year
                df['month'] = df['time'].dt.month
                df['day'] = df['time'].dt.day
                df['hour'] = df['time'].dt.hour

                df = df.drop(columns=['time'])

                # Save
                out_path = os.path.join(self.to_dir, f"{var}_{j}.csv")
                df.to_csv(out_path, index=False)

                logging.info(f"Saved {out_path}")

        logging.info("All downloads complete")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dataset_name', choices=datasets.keys())
    # Tool for debugging
    parser.add_argument('-l', '--log_level', type=int,
                        default=2, choices=[0, 1, 2, 3, 4],
                        help='Level of logging to use')
    parser.add_argument('-s', '--save_log', action='store_true')

    args = parser.parse_args()

    config.dataset_config = datasets[args.dataset_name]
    config.set_log_level(args.log_level)

    downloader = Downloader()
    downloader.download()
