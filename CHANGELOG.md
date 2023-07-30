# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.30.0] - 2023-08-01
### Added
- `Models` folder for extract downloaded model tar file 
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
