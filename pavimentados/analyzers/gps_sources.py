import logging
import datetime as dt
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pynmea2
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from scipy.interpolate import interp1d
from tqdm import tqdm

from pavimentados.analyzers.utils import total_distance, decimal_coords

logger = logging.getLogger(__name__)

class GPS_Processer:
    def __init__(self):
        self._calculate_seconds_from_start()

    def _calculate_seconds_from_start(self):
        self.gps_df["seconds_from_start"] = self.gps_df.seconds.values - self.gps_df.seconds.values[0]

    def adjust_gps_data(self, number_images):
        # Interpolación lineal por cada segundo (corrección de outliers)
        seconds = sorted(set(self.gps_df["seconds_from_start"]))
        for second in seconds:
            lons = self.gps_df.loc[self.gps_df["seconds_from_start"] == second, "longitude"].values
            lats = self.gps_df.loc[self.gps_df["seconds_from_start"] == second, "latitude"].values
            if len(lons) > 2:
                self.gps_df.loc[self.gps_df["seconds_from_start"] == second, "longitude"] = sum(lons) / len(
                    lons
                )
                self.gps_df.loc[self.gps_df["seconds_from_start"] == second, "latitude"] = sum(lats) / len(
                    lats
                )

        list_values = np.linspace(0, self.gps_df.seconds_from_start.max(), number_images).astype("float64")
        f2 = interp1d(self.gps_df["seconds_from_start"].values, self.gps_df["longitude"].values, kind="linear")
        f3 = interp1d(self.gps_df["seconds_from_start"].values, self.gps_df["latitude"].values, kind="linear")
        initial_latitud = self.gps_df.loc[0].latitude
        initial_longitude = self.gps_df.loc[0].longitude
        final_longitude = f2(list_values)
        final_latitude = f3(list_values)
        final_longitude[0] = initial_longitude
        final_latitude[0] = initial_latitud
        self.gps_df = pd.DataFrame({"latitude": final_latitude, "longitude": final_longitude})

    def generate_gps_metrics(self, min_distance_group):
        latitudes = list(self.gps_df.latitude.values)
        longitudes = list(self.gps_df.longitude.values)
        distances = total_distance(latitudes[:-1], longitudes[:-1], latitudes[1:], longitudes[1:])
        distances = np.append(distances, distances[-1])
        id_distances = []
        id_dist = 0
        sum_dist = 0
        self.section_latitude = [latitudes[0]]
        self.section_longitude = [longitudes[0]]
        self.section_distances = []
        for i in range(len(distances)):
            if sum_dist > min_distance_group:
                id_dist += 1
                self.section_distances.append(sum_dist)
                sum_dist = 0
                self.section_latitude.append(latitudes[i])
                self.section_longitude.append(longitudes[i])
            sum_dist += distances[i]
            id_distances.append(id_dist)
        self.section_distances.append(sum_dist)
        self.section_latitude.append(latitudes[i])
        self.section_longitude.append(longitudes[i])
        self.gps_df["distances"] = distances
        self.gps_df["section"] = id_distances


class GPS_Standard_Loader(GPS_Processer):
    def __init__(self, config, route, **kwargs):
        self.config = config
        self.route = Path(route)
        self.gps_df = None
        self.load_gps_data()
        super().__init__()

    def load_gps_data(self):
        data = []
        with open(self.route) as file:
            for line in file.readlines():
                try:
                    data.append(pynmea2.parse(line))
                except pynmea2.ParseError:
                    continue
        gps = {}
        for value in data:
            salida = gps.get(value.sentence_type, None)
            if salida is None:
                gps[value.sentence_type] = []
            gps[value.sentence_type].append(value)
        select = list(gps.keys())[0]
        gps_list = gps[select]
        self.gps_df = pd.DataFrame(
            list(map(lambda x: (x.timestamp, x.longitude, x.latitude), gps_list)),
            columns=["timestamp", "longitude", "latitude"]
        )
        self.gps_df = self.gps_df.drop_duplicates(subset=["longitude", "latitude"]).reset_index(drop=True)
        if isinstance(self.gps_df.loc[0].timestamp, str):
            self.gps_df["seconds"] = list(map(lambda x: int(float(x)) / 1000, self.gps_df.timestamp.values))
        else:
            self.gps_df["seconds"] = list(
                map(lambda x: (x.hour * 3600) + (x.minute * 60) + (x.second), self.gps_df.timestamp.values))


class GPS_CSV_Loader(GPS_Processer):
    def __init__(self, config, route, **kwargs):
        self.config = config

        latitud_column = kwargs.get("latitud_column", None)
        longitud_column = kwargs.get("longitud_column", None)
        time_column = kwargs.get("time_column", None)
        date_column = kwargs.get("date_column", None)
        decimal_character = kwargs.get("decimal_character", ",")

        self.route = Path(route)
        self.default_column_names = ["timestamp", "longitude", "latitude"]
        self.decimal_character = decimal_character
        self.columns_names = []

        if time_column is not None:
            self.columns_names.append(time_column)
        if date_column is not None:
            self.columns_names.append(date_column)
        if longitud_column is not None:
            self.columns_names.append(longitud_column)
        if latitud_column is not None:
            self.columns_names.append(latitud_column)

        self.columns_names = self.default_column_names if len(self.columns_names) == 0 else self.columns_names
        self.load_gps_data()
        super().__init__()

    def load_gps_data(self):
        self.gps_df = pd.read_csv(self.route, sep=";", encoding="latin1", decimal=self.decimal_character)[
            self.columns_names]
        if len(self.columns_names) == 3:
            self.gps_df.columns = ["timestamp", "longitude", "latitude"]
            self.gps_df["timestamp"] = list(
                map(lambda x: dt.datetime.strptime(x, "%Y-%m-%d %H:%M:%S"), self.gps_df["timestamp"]))
        else:
            self.gps_df.columns = ["time", "date", "longitude", "latitude"]
            self.gps_df["timestamp"] = list(
                map(
                    lambda x, y: dt.datetime.strptime(x + " " + y, "%Y-%m-%d %H:%M:%S"),
                    self.gps_df["date"].values,
                    self.gps_df["time"].values,
                )
            )
            self.gps_df = self.gps_df[["timestamp", "longitude", "latitude"]]
        if isinstance(self.gps_df.loc[0].timestamp, str):
            self.gps_df["seconds"] = list(map(lambda x: int(float(x)) / 1000, self.gps_df.timestamp.values))
        else:
            self.gps_df["seconds"] = list(
                map(lambda x: (x.hour * 3600) + (x.minute * 60) + (x.second), self.gps_df.timestamp))


class GPS_Image_Route_Loader(GPS_Processer):
    def __init__(self, config, images_routes, **kwargs):
        self.config = config
        self.routes = images_routes
        self.load_gps_data()
        super().__init__()

    def load_gps_data(self):
        routes = [route for route in self.routes if route.suffix[1:].lower() in self.config["images_allowed"]]
        self.gps_df = pd.DataFrame(list(
            tqdm(map(lambda img_path: self.load_single_value(img_path), routes), desc="Loading GPS from images...",
                 total=len(routes))))
        self.gps_df["seconds"] = list(
            map(lambda x: (x.hour * 3600) + (x.minute * 60) + (x.second), self.gps_df.timestamp))

    def load_exif_data(self, img_path):
        image = Image.open(img_path)
        exifdata = image.getexif()

        d = {}
        for tag_id in exifdata:
            tag = TAGS.get(tag_id, tag_id)
            data = exifdata.get(tag_id)
            try:
                if isinstance(data, bytes):
                    data = data.decode()
                d[tag] = data
            except ValueError:  # noqa: E722
                pass
        return d

    def get_gpsdata_from_exif(self, filename):
        exif = Image.open(filename)._getexif()
        exif_data = {}
        gps_data = {}
        if exif is not None:
            for key, value in exif.items():
                name = TAGS.get(key, key)
                try:
                    data = exif[key].decode() if isinstance(exif[key], bytes) else exif[key]
                    exif_data[name] = data
                except:
                    logger.debug(f'Error decoding exif in key: {key}')

            if 'GPSInfo' in exif_data:
                for key in exif_data['GPSInfo'].keys():
                    name = GPSTAGS.get(key, key)
                    gps_data[name] = exif_data['GPSInfo'].get(key)

            gps_data['DateTimeOriginal'] = exif_data.get('DateTimeOriginal', None)

        return gps_data

    def load_single_value(self, img_path):
        try:
            # get from exif
            d = self.load_exif_data(str(img_path))
            lat = np.array(d["GPSInfo"][2])
            lon = np.array(d["GPSInfo"][4])
            lat = sum(np.array(lat[:, 0] / lat[:, 1]) * np.array([1.0, 1.0 / 60.0, 1.0 / 3600.0])) * (
                -1 if d["GPSInfo"][1] == "S" else 1)
            lon = sum(np.array(lon[:, 0] / lon[:, 1]) * np.array([1.0, 1.0 / 60.0, 1.0 / 3600.0])) * (
                -1 if d["GPSInfo"][3] == "W" else 1)
            time = dt.datetime.strptime(d["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
        except:  # noqa: E722
            try:
                # get from gpsinfo
                d = self.get_gpsdata_from_exif(str(img_path))
                lat = decimal_coords(d['GPSLatitude'], d['GPSLatitudeRef'])
                lon = decimal_coords(d['GPSLongitude'], d['GPSLongitudeRef'])
                time = dt.datetime.strptime(d["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
            except:
                logger.debug("Could not load GPS data from image: %s", img_path)
                return

        return {"timestamp": time, "longitude": lon, "latitude": lat}


class GPS_Image_Folder_Loader(GPS_Image_Route_Loader):
    def __init__(self, config, route, **kwargs):
        self.config = config
        self.routes = sorted([Path(route) / item for item in os.listdir(Path(route)) if not item.startswith(".")])
        self.load_gps_data()
        super(GPS_Image_Route_Loader, self).__init__()


gps_source_options_dict = {
    "image_routes": GPS_Image_Route_Loader,
    "image_folder": GPS_Image_Folder_Loader,
    "csv": GPS_CSV_Loader,
    "loc": GPS_Standard_Loader,
}


def GPS_Data_Loader(source_type, gps_in, config, **kwargs):
    if source_type not in gps_source_options_dict:
        raise NameError(f"{source_type} not implemented on the method")
    return gps_source_options_dict[source_type](config, gps_in, **kwargs)
