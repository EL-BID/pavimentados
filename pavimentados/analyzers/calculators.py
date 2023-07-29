import numpy as np
import pandas as pd

from pavimentados.analyzers.utils import (  # total_distance,
    area_calc,
    assign_group_calculations,
    box_center,
    box_height,
    box_width,
    fail_id_generator,
    stack_columns_dataset,
)

first_aggregation_dict = {"ind": "count", "perc_area": "sum"}

second_aggregation_dict = {
    "distances": "sum",
    "start_coordinate": "first",
    "start_latitude": "first",
    "start_longitude": "first",
    "end_coordenate": "last",
    "end_latitude": "last",
    "end_longitude": "last",
    "width": "mean",
    "area": "sum",
    "boxes": "sum",
}

third_aggregation_dict = {"width": "sum", "distances": np.mean, "boxes": "count"}


class Results_Calculator:
    @staticmethod
    def generate_paviment_results(
        results_obj,
        img_obj,
        gps_obj,
        min_fotogram_distance=5,
        columns_to_have=[
            "Grieta Lineal Longitudinal",
            "Intervalo Lineal Longitudinal",
            "Grieta Lineal Transversal",
            "Intervalo Lineal Transversal",
            "Piel de Cocodrilo",
            "Protuberancia, Bache",
            "Otras fallas",
        ],
    ):
        """
        Genera la table final de resultados de pavimentos.
        """
        # Genero el dataset.
        data = gps_obj.gps_df.copy()
        data["scores"] = results_obj["scores_pav"]
        data["boxes"] = results_obj["boxes_pav"]
        data["classes"] = results_obj["final_pav_clases"]
        data["fotograma"] = data.index
        altura, base = img_obj.get_altura_base()

        # Calculamos la cantidad de detecciones por fotograma.
        data["len"] = list(map(lambda x: len(x), data.scores.values))
        data = data.loc[data.len > 0].reset_index(drop=True)

        # Como el dataframe tiene listas de detecciones por fotograma
        # transformamos esas listas es varias lineas.
        data_resulting = stack_columns_dataset(
            data, ["classes", "scores", "boxes"], ["latitude", "longitude", "distances", "ind", "fotograma", "section"]
        )
        # Agregamos nuevas variables.
        data_resulting["class_id"] = data_resulting.classes.values
        data_resulting["area"] = list(map(lambda x: area_calc(x, altura, base), data_resulting.boxes.values))
        data_resulting["center"] = list(map(lambda x: box_center(x, altura, base), data_resulting.boxes.values))
        data_resulting["height"] = list(map(lambda x: box_height(x, altura), data_resulting.boxes.values))
        data_resulting["width"] = list(map(lambda x: box_width(x, base), data_resulting.boxes.values))
        data_resulting["total_area"] = base * altura
        data_resulting["perc_area"] = data_resulting.area.values / data_resulting.total_area.values

        # Agrupamos data.
        # data_resulting_g = (
        #     data_resulting.groupby(["latitude", "longitude", "fotograma", "distances", "classes", "section"])
        #     .aggregate(first_aggregation_dict)
        #     .reset_index()
        #     .rename({"ind": "cantidad"}, axis=1)
        #     .sort_values(["fotograma"])
        #     .reset_index(drop=True)
        # )
        # data_resulting_g_total = (
        #     data_resulting.groupby(["classes"])
        #     .aggregate(first_aggregation_dict)
        #     .reset_index()
        #     .rename({"ind": "quantity"}, axis=1)
        #     .reset_index(drop=True)
        # )

        # Generamos la ID de fail.
        # variables = list(data_resulting.classes.unique())
        data_resulting = fail_id_generator(data_resulting, min_fotogram_distance)

        # Agrupamos data por sections.
        data_resulting_g_sections = (
            data_resulting.groupby(["class_id", "classes", "fotograma", "latitude", "longitude", "fail_id_section"])
            .aggregate(third_aggregation_dict)
            .reset_index()
            .sort_values(["fotograma"])
        )
        data_resulting_g_sections2 = (
            data_resulting.groupby(["class_id", "classes", "fotograma", "latitude", "longitude", "section"])
            .aggregate(third_aggregation_dict)
            .reset_index()
            .sort_values(["fotograma"])
        )

        # Agregamos nuevas variables.
        data_resulting_g_sections = assign_group_calculations(data_resulting_g_sections)
        data_resulting_g_sections2 = assign_group_calculations(data_resulting_g_sections2)

        # Generamos los datasets resulting.
        data_resulting_fails = (
            data_resulting_g_sections.groupby(["class_id", "classes", "fail_id_section"]).aggregate(second_aggregation_dict).reset_index()
        )
        data_resulting_sections = (
            data_resulting_g_sections2.groupby(["class_id", "classes", "section"]).aggregate(second_aggregation_dict).reset_index()
        )

        table_summary_sections = (
            data_resulting_sections.pivot_table(index=["section"], columns=["classes"], values=["distances"]).fillna(0).astype("int")
        )
        table_summary_sections.columns = table_summary_sections.columns.droplevel(0)
        index_sections_location = list(table_summary_sections.reset_index().section.values)
        index_sections_location_end = list(np.array(index_sections_location) + 1)
        table_summary_sections["latitude"] = np.array(gps_obj.section_latitude)[index_sections_location]
        table_summary_sections["longitude"] = np.array(gps_obj.section_longitude)[index_sections_location]
        table_summary_sections["end_latitude"] = np.array(gps_obj.section_latitude)[index_sections_location_end]
        table_summary_sections["end_longitude"] = np.array(gps_obj.section_longitude)[index_sections_location_end]
        table_summary_sections["section_distance"] = np.array(gps_obj.section_distances)[index_sections_location]
        for col in columns_to_have:
            if col not in table_summary_sections.columns:
                table_summary_sections[col] = 0
        table_summary_sections = table_summary_sections.reset_index()
        table_summary_sections = pd.DataFrame(table_summary_sections.values, columns=list(table_summary_sections.columns))
        table_summary_sections = table_summary_sections[
            ["section", *columns_to_have, "latitude", "longitude", "end_longitude", "end_latitude", "section_distance"]
        ]
        return table_summary_sections, data_resulting, data_resulting_fails

    @staticmethod
    def generate_final_results_signal(
        results_obj,
        gps_obj,
        classes_names_yolo_signal=[
            "ADTCIA",
            "CIRCROJO",
            "CRUZAMLLA",
            "CUADBCO",
            "CUADCAFE",
            "CUADVERDE",
            "CUADZUL",
            "CUAMLLO",
            "CURVA",
            "INDAZUL",
            "OBRA",
            "OTRO",
            "PROBROJO",
            "ROMAMLLO",
            "RUTA",
            "SEMAFORO",
            "STOP",
            "TRIROJO",
        ],
    ):
        BOXES_SIGNAL = results_obj["boxes_signal"]
        CLASSES_SIGNAL = results_obj["classes_signal"]
        SIGNAL_CLASSES_siames = results_obj["final_signal_classes"]
        SIGNAL_CLASSES_BASE = results_obj["signal_base_predictions"]
        SIGNAL_STATE = results_obj["state_predictions"]
        SCORES_SIGNAL = results_obj["scores_signal"]
        final_latitude = gps_obj.gps_df.latitude.values
        final_longitude = gps_obj.gps_df.longitude.values
        fotograma = list(range(len(BOXES_SIGNAL)))

        # Nos dice el cuadrante.
        def position(center):
            if center < 0.33:
                return 0
            elif center < 0.66:
                return 1
            else:
                return 2

        position_boxes = [[position((box[1] + box[3]) / 2) for box in BOXES_SIGNAL[f]] for f in range(len(BOXES_SIGNAL))]

        rows = []
        for f in fotograma:
            for i in range(len(CLASSES_SIGNAL[f])):
                r = [
                    f,
                    position_boxes[f][i],
                    SCORES_SIGNAL[f][i],
                    SIGNAL_STATE[f][i],
                    SIGNAL_CLASSES_siames[f][i],
                    SIGNAL_CLASSES_BASE[f][i],
                    int(CLASSES_SIGNAL[f][i]),
                    final_latitude[f],
                    final_longitude[f],
                ]
                rows.append(r)

        df = pd.DataFrame(
            rows,
            columns=[
                "fotogram",
                "position_boxes",
                "score",
                "signal_state",
                "signal_class_siames",
                "signal_class_base",
                "signal_class",
                "latitude",
                "longitude",
            ],
        )

        df["signal_class_siames_names"] = df["signal_class_siames"].values
        df["signal_class_names"] = df["signal_class"].apply(lambda x: classes_names_yolo_signal[x])

        yolo_classes_to_keep = []

        x = tuple(zip(df["signal_class_siames_names"].values, df["signal_class_names"].values))

        final_classes = []
        for i in range(len(x)):
            if x[i][1] in yolo_classes_to_keep:
                final_classes.append(x[i][1])
            else:
                final_classes.append(x[i][0])

        df["final_classes"] = final_classes
        df["ID"] = range(len(df))

        df = df.sort_values(["final_classes", "position_boxes", "fotogram"]).reset_index(drop=True)

        # cantidad de fotogramas a sacar repetidas.
        N_fotogram = 5

        for i in range(len(df) - 1, 0, -1):
            if (
                (df.loc[i - 1, "final_classes"] == df.loc[i, "final_classes"])
                & (df.loc[i - 1, "position_boxes"] == df.loc[i, "position_boxes"])
                & (np.abs(df.loc[i - 1, "fotogram"] - df.loc[i, "fotogram"]) <= N_fotogram)
                & (df.loc[i, "final_classes"] != "OTRO")
            ):
                df.loc[i - 1, "ID"] = df.loc[i, "ID"]

        df = df.sort_values("fotogram", ascending=False).reset_index(drop=True).drop_duplicates(subset="ID")
        df = df.sort_values("fotogram").reset_index(drop=True)
        return df
