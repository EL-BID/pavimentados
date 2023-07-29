from pavimentados.analyzers import calculators, gps_sources  # noqa: F401
from pavimentados.downloader import Downloader  # noqa: F401
from pavimentados.processing import processors, sources, workflows  # noqa: F401

dl = Downloader()
download_models = dl.download
