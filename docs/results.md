# Result data
---

As a result of processing a video or images, the library returns the following datasets:

## data_resulting

Information about the individual detections made on each image / photogram.

| Field           | Type         | Description                                                                         | Example                                                                        |
|-----------------|--------------|-------------------------------------------------------------------------------------|--------------------------------------------------------------------------------|
| latitude        | float        | Latitude of the specific location.                                                  | -11.7836697676768                                                              |
| longitude       | float        | Longitude of the specific location.                                                 | -77.1556084848485                                                              |
| distances       | float        | Distance in meters from the frame where the detection is located.                   | 0.01                                                                           |
| ind             | int          | Index of the record or row.                                                         | 0                                                                              |
| fotograma       | int          | Identifier of the frame or image in which the record was made.                      | 27                                                                             |
| section         | int          | Section of the analysis.                                                            | 0                                                                              |
| classes         | string       | Class or label assigned to the object.                                              | D00                                                                            |
| ind2            | int          | Secondary index or identifier.                                                      | 0                                                                              |
| scores          | float        | Confidence score of the model (probability).                                        | 0.22                                                                           |
| boxes           | list[float]  | Coordinates of the bounding box of the object, format [x_min, y_min, x_max, y_max]. | [0.4320632219314575, 0.8396928310394287, 0.6515443325042725, 0.99937504529953] |
| class_id        | string       | Identifier of the assigned class.                                                   | D00                                                                            |
| area            | float        | Calculated area of the object in pixels.                                            | 72654.200                                                                      |
| center          | tuple[float] | Coordinates of the center of the box, format (x, y).                                | (1765.5051612854004, 585.1480793952942)                                        |
| height          | float        | Height of the box in pixels.                                                        | 237.03959941864                                                                |
| width           | float        | Width of the box in pixels.                                                         | 306.589851379395                                                               |
| total_area      | int          | Total area of the analyzed image.                                                   | 2073600                                                                        |
| fail_id_section | int          | Identifier of the failure.                                                          | 0                                                                              |

## data_resulting_fails

Information about each of the failures. A failure is the aggregation of several detections that are considered part of
the same problem (failures have a single type and can have many detections).

| Field            | Type   | Description                                                               | Example                                     |
|------------------|--------|---------------------------------------------------------------------------|---------------------------------------------|
| class_id         | int    | Identifier of the class assigned to the detection.                        | 0                                           |
| classes          | string | The class label for the detection.                                        | D00                                         |
| fail_id_section  | int    | Identifier of the failure section associated with the detection.          | 43                                          |
| distances        | float  | distance in meters of the failure.                                        | 65.8743799240202                            |
| start_coordinate | string | Starting coordinate of the detection, formatted as (latitude, longitude). | "(-11.783669767676768, -77.15560848484849)" |
| start_latitude   | float  | Latitude of the starting point of the failure.                            | -11.783669767676768                         |
| start_longitude  | float  | Longitude of the starting point of the failure.                           | -77.15560848484849                          |
| end_coordenate   | string | Ending coordinate of the detection, formatted as (latitude, longitude).   | "(-11.7830935, -77.15626066666667)"         |
| end_latitude     | float  | Latitude of the ending point of the failure.                              | -11.7830935                                 |
| end_longitude    | float  | Longitude of the ending point of the failure.                             | -77.15626066666667                          |
| width            | float  | Average width of faults measured in pixels.                               | 789.3660466811236                           |
| boxes            | int    | Number of bounding boxes representing the failure.                        | 19                                          |

## table_summary_sections

Information about what is detected every 100 meters.

| Field            | Type  | Description                                        | Example Value       |
|------------------|-------|----------------------------------------------------|---------------------|
| section          | float | Identifier of the section.                         | 0.0                 |
| D00              | float | Linear distance in meters of this type of failure. | 65.0                |
| D10              | float | Linear distance in meters of this type of failure. | 0.0                 |
| D20              | float | Linear distance in meters of this type of failure. | 33.0                |
| D40              | float | Linear distance in meters of this type of failure. | 0.0                 |
| D43              | float | Linear distance in meters of this type of failure. | 0.0                 |
| D44              | float | Linear distance in meters of this type of failure. | 0.0                 |
| latitude         | float | Starting latitude of the section.                  | -11.783696666666666 |
| longitude        | float | Starting longitude of the section.                 | -77.15555183333333  |
| end_longitude    | float | Ending longitude of the section.                   | -77.15626066666667  |
| end_latitude     | float | Ending latitude of the section.                    | -11.7830935         |
| section_distance | float | Distance of the section in meters.                 | 106.6830640804603   |

## signals_summary

Information about signals detected in each frame.

| Field              | Type   | Description                                              | Example                      |
|--------------------|--------|----------------------------------------------------------|------------------------------|
| fotogram           | int    | Frame number of the video.                               | 52                           |
| position_boxes     | int    | Position of the bounding box in the frame.               | 1                            |
| score              | float  | Confidence score of the detection.                       | 0.40509921312332153          |
| signal_class       | int    | Identifier of the signal class.                          | 8                            |
| latitude           | float  | Latitude of the location.                                | 10.380599907222223           |
| longitude          | float  | Longitude of the location.                               | -84.36906742202616           |
| boxes              | string | Bounding box coordinates in the format [x1, y1, x2, y2]. | [0.540, 0.205, 0.550, 0.219] |
| signal_class_names | string | Name of the signal class.                                | INDAZUL                      |
| ID                 | int    | Unique identifier of the detection.                      | 0                            |