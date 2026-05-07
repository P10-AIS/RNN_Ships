import pickle
import numpy as np
import numpy as np
from pyproj import Transformer


class TileGrid:
    def __init__(self, tile_size: int, crs: str, E0: float, N0: float):
        self._tile_size = tile_size
        self._crs = crs
        self._E0 = E0
        self._N0 = N0

    def to_dict(self) -> dict:
        return {
            "tile_size": self._tile_size,
            "crs": self._crs,
            "E0": self._E0,
            "N0": self._N0
        }

    def coords_to_tiles(self, src_crs: str, xs: np.ndarray, ys: np.ndarray):
        transformer = Transformer.from_crs(src_crs, self._crs, always_xy=True)
        Es, Ns = transformer.transform(xs, ys)

        tx = np.floor((Es - self._E0) / self._tile_size).astype(int)
        ty = np.floor((Ns - self._N0) / self._tile_size).astype(int)
        return tx, ty


class Forcemap:
    def __init__(self, data: np.ndarray, grid: TileGrid):
        self._data = data.astype(np.float32)
        self._grid = grid

    def get_grid(self):
        return self._grid

    def get_array(self):
        return self._data

    def get_shape(self):
        return self._data.shape

    @staticmethod
    def load(path: str):
        with np.load(path, allow_pickle=True) as data:
            grid = TileGrid(**pickle.loads(data['grid'].item()))
            return Forcemap(data['data'], grid)
