from pavimentados.processing import processors, workflows, sources
from pavimentados.analyzers import calculators, gps_sources
from pavimentados.downloader import Downloader
import os
import sys

dl = Downloader()
download_models = dl.download