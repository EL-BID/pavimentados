# GPS Data Formats
---

## CSV format

> Rows and columns are separated by a `;` field delimiter. 

| Field     | Description                           | Example              |
|-----------|---------------------------------------|----------------------|
| time      | Time in format HH:MM:SS               | 15:10:00             |
| date      | Date in format YYYY-MM-DD             | 2024-05-24           |
| latitude  | Latitude of the specific location.    | -24.1457549          |
| longitude | Longitude of the specific location.   | -49.827651299972224  |


## LOC format 

> NMEA like file with the GPS data.

Example:
```text
@Sonygps/ver5.0/wgs-84/20200625135753.000/
@Sonygpsoption/0/20200625135754.000/20200625135754.000/
$GPGGA,155802.0000,01148.51596,S,7707.94776,W,1,0,,,M,,M,,
$GPRMC,155802.0000,A,01148.51596,S,7707.94776,W,19.31,,250620,,,
$GPGGA,155802.0000,01148.51558,S,7707.94802,W,1,0,,,M,,M,,
$GPRMC,155802.0000,A,01148.51558,S,7707.94802,W,19.31,,250620,,,
$GPGGA,155802.0000,01148.51520,S,7707.94828,W,1,0,,,M,,M,,
$GPRMC,155802.0000,A,01148.51520,S,7707.94828,W,19.31,,250620,,,
$GPGGA,155802.0000,01148.51482,S,7707.94855,W,1,0,,,M,,M,,
$GPRMC,155802.0000,A,01148.51482,S,7707.94855,W,19.31,,250620,,,
$GPGGA,155802.0000,01148.51444,S,7707.94881,W,1,0,,,M,,M,,
$GPRMC,155802.0000,A,01148.51444,S,7707.94881,W,19.31,,250620,,,
$GPGGA,155802.0000,01148.51407,S,7707.94907,W,1,0,,,M,,M,,
$GPRMC,155802.0000,A,01148.51407,S,7707.94907,W,19.31,,250620,,,
```
