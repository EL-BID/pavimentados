# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.4] - 2024-12-07
### Added
- Add some custom exceptions on read inputs.

## [1.0.2] - 2024-11-20
### Added
- Add utility to convert gpx files to nmea.

## [1.0.1] - 2024-11-15
### Changed
- User manual updated.

## [1.0.0] - 2024-11-14
### Changed
- Unified config: all configurations in only one json file.
- Documentation updated with examples.
- Siamese model upgraded to 128x128 detections.
### Added
- Docstrings to some classes and methods.
- Mapping of paviment codes.
### Fixed
- Distance calculation for paviment results, in some cases wrong distance is calculated.
- Frame convertion to RGB.
### Removed
- Downloader and image/utils modules.
- YoloV3 models classes.
- Unnecessary JSON config files.
- Remove TensorFlow and Joblib from dependencies.
- Traffic sign results.

## [0.31.3] - 2024-03-23
### Changed
- Sets number of classes parameter in the model configuration file.

## [0.31.1] - 2024-03-19
### Changed
- Update pre-commit hooks.
### Fixed
- Fix first detection of road signals. Wrong class assigned in some situations.
- Fix position column in the signals dataframe.

## [0.31.0] - 2023-11-30
### Added
- `Signals detected`: add boxes column on results dataframe.
### Changed
- `Signals detected`: remove duplicated detections resulting from stopped vehicle.

## [0.30.1] - 2023-08-21
### Added
- `notebooks` folder for process put samples and notebooks
- Samples of images and videos processing 

## [0.30.0] - 2023-08-01
### Added
- `models` folder for extract downloaded model tar file 
### Fixed
- Removed unused imports
- Line breaks for too long lines
- F841 removed unused variables
- E711 Change comparisons to None
- F541 f-string is missing placeholders
- E712 comparison to True
- E741 ambiguous variable names
- E721 do not compare types, use 'isinstance()'
- Sonarcloud: Make sure that using this pseudorandom number generator is safe here

## [0.29.1] - 2023-04-12
### Fixed
- Fix tensorflow and open-cv version on requirements.txt 


The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).
