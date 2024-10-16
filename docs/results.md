# Result data
---

As a result of processing a video or images, the library returns the following datasets:


## data_resulting / detections_over_photogram
Information about the individual detections made on each image / photogram.

| Field           | Type         | Description                                                                         | Example                                                                        |
|-----------------|--------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| latitude        | float        | Latitude of the specific location.                                                  | -11.7836697676768                                                              |
| longitude       | float        | Longitude of the specific location.                                                 | -77.1556084848485                                                              |
| distances       | float        | Distance in meters from the frame where the detection is located.                   | 0.01                                                                           |
| ind             | int          | Index of the record or row.                                                         | 0                                                                              |
| fotograma       | int          | Identifier of the frame or image in which the record was made.                      | 27                                                                             |
| section         | int          | Section of the analysis.                                                            | 0                                                                              |
| classes         | string       | Class or label assigned to the object.                                              | OT0                                                                            |
| ind2            | int          | Secondary index or identifier.                                                      | 0                                                                              |
| scores          | float        | Confidence score of the model (probability).                                        | 0.22                                                                           |
| boxes           | list[float]  | Coordinates of the bounding box of the object, format [x_min, y_min, x_max, y_max]. | [0.4320632219314575, 0.8396928310394287, 0.6515443325042725, 0.99937504529953] |
| class_id        | string       | Identifier of the assigned class.                                                   | OT0                                                                            |
| area            | float        | Calculated area of the object in pixels.                                            | 72654.200                                                                      |
| center          | tuple[float] | Coordinates of the center of the box, format (x, y).                                | (1765.5051612854004, 585.1480793952942)                                        |
| height          | float        | Height of the box in pixels.                                                        | 237.03959941864                                                                |
| width           | float        | Width of the box in pixels.                                                         | 306.589851379395                                                               |
| total_area      | int          | Total area of the analyzed image.                                                   | 2073600                                                                        |
| perc_area       | float        | Percentage of area occupied by the object relative to total.                        | 0.0350472297245332                                                             |
| fail_id_section | int          | Identifier of the failure.                                                          | 0                                                                              |


## data_resulting_fails / failures_detected
Information about each of the failures. A failure is the aggregation of several detections that are considered part of 
the same problem (failures have a single type and can have many detections).

| Field            | Type   | Description                                                               | Example                                     |
|------------------|--------|---------------------------------------------------------------------------|---------------------------------------------|
| class_id         | int    | Identifier of the class assigned to the detection.                        | 0                                           |
| classes          | string | The class label for the detection.                                        | D00                                         |
| fail_id_section  | int    | Identifier of the failure section associated with the detection.          | 43                                          |
| distances        | float  | Distance measured in relation to a reference point.                       | 65.8743799240202                            |
| start_coordinate | string | Starting coordinate of the detection, formatted as (latitude, longitude). | "(-11.783669767676768, -77.15560848484849)" |
| start_latitude   | float  | Latitude of the starting point of the detection.                          | -11.783669767676768                         |
| start_longitude  | float  | Longitude of the starting point of the detection.                         | -77.15560848484849                          |
| end_coordenate   | string | Ending coordinate of the detection, formatted as (latitude, longitude).   | "(-11.7830935, -77.15626066666667)"         |
| end_latitude     | float  | Latitude of the ending point of the detection.                            | -11.7830935                                 |
| end_longitude    | float  | Longitude of the ending point of the detection.                           | -77.15626066666667                          |
| width            | float  | Average width of faults measured in pixels.                               | 789.3660466811236                           |
| area             | float  | Total area of fault detections measured in pixels.                        | 50000.46419064892                           |
| boxes            | int    | Number of bounding boxes representing the detection.                      | 19                                          |


## table_summary_sections / pavimenta2_sections
Information about what is detected every 100 meters.

| Field            | Type  | Description                                        | Example Value       |
|------------------|-------|----------------------------------------------------|---------------------|
| section          | float | Identifier of the section.                         | 0.0                 |
| D00              | float | Linear distance in meters of this type of failure. | 65.0                |
| D01              | float | Linear distance in meters of this type of failure. | 0.0                 |
| D10              | float | Linear distance in meters of this type of failure. | 0.0                 |
| D11              | float | Linear distance in meters of this type of failure. | 8.0                 |
| D20              | float | Linear distance in meters of this type of failure. | 33.0                |
| D40              | float | Linear distance in meters of this type of failure. | 0.0                 |
| D43              | float | Linear distance in meters of this type of failure. | 0.0                 |
| D44              | float | Linear distance in meters of this type of failure. | 0.0                 |
| latitude         | float | Starting latitude of the section.                  | -11.783696666666666 |
| longitude        | float | Starting longitude of the section.                 | -77.15555183333333  |
| end_longitude    | float | Ending longitude of the section.                   | -77.15626066666667  |
| end_latitude     | float | Ending latitude of the section.                    | -11.7830935         |
| section_distance | float | Distance of the section in meters.                 | 106.6830640804603   |


## signals_summary / signals_detected
Information about the detected signals.

| Field                     | Type          | Description                                                                                      | Example Value                      |
|---------------------------|---------------|--------------------------------------------------------------------------------------------------|------------------------------------|
| fotogram                  | int           | Frame number in which the detection occurred.                                                    | 22                                 |
| position_boxes            | int           | Identifier for the position of the bounding boxes.                                               | 1                                  |
| score                     | float         | Confidence score for the detection.                                                              | 0.2038                             |
| signal_state              | string        | State of the detected signal.                                                                    | "warning--curve-left--g1"          |
| signal_class_siames       | string        | Class of the signal in the Siamese network.                                                      | "warning--curve-left--g1"          |
| signal_class_base         | string        | Base class of the detected signal.                                                               | "warning--curve-left--g1"          |
| signal_class              | string        | Final classification of the detected signal.                                                     | "warning--curve-left--g1"          |
| latitude                  | float         | Latitude of the detection location.                                                              | -11.7837                           |
| longitude                 | float         | Longitude of the detection location.                                                             | -77.1556                           |
| boxes                     | list of float | Coordinates of the bounding box around the detected signal, format [x_min, y_min, x_max, y_max]. | "[0.3291, 0.4406, 0.3411, 0.4628]" |
| signal_class_siames_names | string        | Human-readable name for the Siamese class of the signal.                                         | "warning--curve-left--g1"          |
| signal_class_names        | string        | Human-readable name for the class of the signal.                                                 | "OTRO"                             |
| final_classes             | string        | Final classification output for the detection.                                                   | "warning--curve-left--g1"          |
| ID                        | int           | Unique identifier for the detection record.                                                      | 19                                 |

