import numpy as np
import pandas as pd


def total_distance(lat1, lon1, lat2, lon2):
    """Calcula la distancia entre dos puntos (lat1, lon1) y (lat2, lon2)."""

    R = 6373.0  # approximate radius of earth in km.

    lat1 = np.radians(lat1)
    lon1 = np.radians(lon1)
    lat2 = np.radians(lat2)
    lon2 = np.radians(lon2)

    dlon = lon2 - lon1
    dlat = lat2 - lat1

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distance = R * c * 1000
    return distance


def area_calc(boxes, altura, base):
    """Calcula el area de la caja."""
    if len(boxes) > 0:
        return ((boxes[2] - boxes[0]) * altura) * ((boxes[3] - boxes[1]) * base)
    else:
        return 0


def box_center(boxes, altura, base):
    """Calcula el centro de la caja."""
    if len(boxes) > 0:
        return (((boxes[3] + boxes[1]) / 2) * base), (((boxes[2] + boxes[0]) / 2) * altura)
    else:
        return None, None


def box_height(boxes, altura):
    """Calcula la altura de la caja."""
    if len(boxes) > 0:
        return (boxes[2] - boxes[0]) * altura
    else:
        return None


def box_width(boxes, base):
    """Calcula la base de la caja."""
    if len(boxes) > 0:
        return (boxes[3] - boxes[1]) * base
    else:
        return None


def fail_id_generator(df, min_photogram_distance):
    """Genera un ID de fail.

    Esto se realiza ya que cada fail es detectada en fotogramas
    simultaneos y es necesario identificarla como una misma fail.
    """

    df_id_fails = []
    id_fail = 0
    for fail in list(df.classes.unique()):
        df_fail = df.loc[df.classes == fail].copy().reset_index(drop=True)
        fotogramas = df_fail.fotograma
        id_section_fail = [id_fail]
        for i in range(len(fotogramas) - 1):
            if (fotogramas[i + 1] - fotogramas[i]) > min_photogram_distance:
                id_fail += 1
            id_section_fail.append(id_fail)
        id_section_fail = np.array(id_section_fail)
        if len(id_section_fail) > 0:
            df_fail["fail_id_section"] = id_section_fail
            df_id_fails.append(df_fail)
    df = pd.concat(df_id_fails).sort_values(["fotograma", "classes", "fail_id_section"]).reset_index(drop=True)
    return df


def stack_columns_dataset(df, variables, static_variables):
    df["ind"] = df.index
    df_resulting = df[static_variables].copy()
    c = 0
    for v in variables:
        d = pd.DataFrame([[i, t] for i, T in df[["ind", v]].values for t in T], columns=["ind", v])
        d["ind2"] = d.index
        if c == 0:
            df_resulting = pd.merge(df_resulting, d, on="ind", how="left")
            c += 1
        else:
            df_resulting = pd.merge(df_resulting, d, on=["ind", "ind2"], how="left")
    return df_resulting


def assign_group_calculations(df):
    # df["area"] = df.width.values * df.distances.values
    df["start_coordinate"] = list(map(lambda x, y: (x, y), df.latitude.values, df.longitude.values))
    df["end_coordenate"] = list(map(lambda x, y: (x, y), df.latitude.values, df.longitude.values))
    df["start_latitude"] = df.latitude
    df["end_latitude"] = df.latitude
    df["start_longitude"] = df.longitude
    df["end_longitude"] = df.longitude
    return df


def decimal_coords(coords, ref):
    decimal_degrees = float(coords[0]) + float(coords[1]) / 60 + float(coords[2]) / 3600
    if ref == "S" or ref == 'W':
        decimal_degrees = -1 * decimal_degrees
    return decimal_degrees
