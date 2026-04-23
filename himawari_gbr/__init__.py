"""himawari_gbr – efficient access to Himawari-9 AHI L2 cloud products for the GBR."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("himawari-gbr")
except PackageNotFoundError:
    __version__ = "0.0.0"
