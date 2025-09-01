# Changelog

All notable changes to the Advanced Motion Magnification System will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Performance profiling and benchmarking tools
- GPU acceleration support
- Video file input mode
- Parameter preset system
- Batch processing capabilities

### Changed
- Further optimization of pyramid processing
- Enhanced user interface design

## [2.0.0] - 2024-01-15

### Added
- **Video Stabilization System**: Automatic camera shake compensation using feature tracking
- **Intelligent Masking**: Region-specific motion magnification with background isolation
- **Mask Editor Window**: Real-time visual feedback for mask editing
- **Automatic Edge Detection**: One-click pipe/structural element detection using Canny edge detection
- **Interactive Mask Tools**: Brush size adjustment, mask clear/invert operations
- **Responsive Key Handling**: Optimized input processing for immediate response

### Changed
- **Pyramid Optimization**: Reduced from 3×6 to 2×4 (8 total bands) for better performance
- **Processing Architecture**: Streamlined pipeline for 60% faster processing
- **Key Response System**: Centralized key handling with immediate response (~50ms vs 2-3s)
- **Test Pattern Resolution**: Reduced to 240×180 for optimal responsiveness

### Fixed
- **Button Delay Issues**: Eliminated 2-3 second delays in key response
- **Memory Management**: Improved Mat type handling and conversion
- **Processing Flow**: Optimized early exit logic for better user experience

### Technical
- Stabilizer with feature detection and transform estimation
- Mask system with brush-based editing and auto-detection
- Performance optimizations reducing computational overhead
- Enhanced error handling and type safety

## [1.5.0] - 2024-01-10

### Added
- **Phase-Based Motion Magnification**: Advanced processing using steerable pyramid decomposition
- **Complex Steerable Pyramid**: 3 scales × 6 orientations for directional motion analysis
- **Gabor Filter Bank**: Orientation-selective filters with scale-adaptive parameters
- **Enhanced Frequency Sweep**: Scale-weighted energy analysis for better recommendations
- **Multi-Scale Processing**: Independent processing of different spatial frequencies

### Changed
- **Core Algorithm**: Replaced simple Laplacian pyramid with sophisticated steerable pyramid
- **Motion Processing**: Switched from amplitude-based to phase-based amplification
- **Energy Calculation**: Added scale weighting for more accurate frequency analysis
- **Visual Feedback**: Updated overlay to show "Phase-Based Steerable" processing

### Technical
- Implemented 18 frequency-orientation bands (3×6 pyramid)
- Added Gabor kernel generation with scale-dependent parameters
- Enhanced reconstruction algorithm for steerable pyramid
- Improved temporal filtering with per-orientation processing

## [1.2.0] - 2024-01-08

### Added
- **Frequency Sweep Mode**: Automatic analysis of motion frequencies from 0.2-10.0 Hz
- **Motion Energy Analysis**: Real-time energy calculation across frequency spectrum
- **Auto-Suggestion System**: Intelligent recommendations for optimal frequency bands
- **Visual Results Display**: ASCII bar chart showing energy distribution across frequencies
- **Progress Tracking**: Real-time progress indication during frequency sweep

### Changed
- **Parameter Display**: Enhanced overlay showing current values for all trackbars
- **Control Interface**: Added 's' key for frequency sweep activation
- **Energy Weighting**: Scale-weighted energy calculation for better analysis

### Fixed
- **Frequency Range Validation**: Automatic enforcement of fh > fl + 0.05 Hz minimum gap
- **Sweep Progress**: Proper frame counting and progress indication

## [1.1.0] - 2024-01-05

### Added
- **Dynamic Parameter Display**: Real-time overlay showing trackbar values and computed parameters
- **Enhanced Trackbar Names**: More descriptive labels with value ranges
- **Two-Line Overlay**: Separated parameter display for better readability
- **Value Mapping**: Clear indication of trackbar-to-parameter conversion

### Changed
- **Visual Interface**: Improved trackbar labeling and status display
- **Parameter Feedback**: Real-time display of Alpha, frequency bands, and chroma attenuation
- **Overlay Layout**: Optimized text positioning and sizing for better visibility

### Fixed
- **Trackbar Deprecation Warnings**: Replaced deprecated value pointers with safer API calls
- **Parameter Synchronization**: Ensured trackbar values match actual processing parameters

## [1.0.0] - 2024-01-01

### Added
- **QuickTime-Compatible Video Export**: H.264 MP4 output with proper encoding
- **FFmpeg Integration**: Automatic video conversion with multiple codec fallbacks
- **Two-Stage Recording**: Reliable AVI intermediate with MP4 final output
- **Comprehensive Error Handling**: Graceful fallbacks for missing codecs/tools

### Changed
- **Video Pipeline**: Moved from direct MP4 to AVI→MP4 conversion workflow
- **Codec Strategy**: Multiple fallback options for maximum compatibility
- **File Management**: Automatic cleanup of intermediate files

### Fixed
- **QuickTime Playback Issues**: Resolved codec and container compatibility problems
- **Video Corruption**: Eliminated frame drops and encoding artifacts
- **Cross-Platform Compatibility**: Consistent video output across systems

## [0.2.0] - 2023-12-28

### Added
- **Interactive Trackbar Controls**: Real-time parameter adjustment during processing
- **Parameter Trackbars**: Alpha (0-100), Low/High frequency cutoffs, Chroma attenuation
- **Live Parameter Updates**: Instant visual feedback for all parameter changes
- **Enhanced User Interface**: Dedicated controls window with organized parameter layout

### Changed
- **Processing Pipeline**: Dynamic parameter updates without restart
- **User Experience**: Interactive real-time control vs. static parameters
- **Interface Layout**: Multi-window design with controls and preview

## [0.1.0] - 2023-12-25

### Added
- **Core Motion Magnification**: Laplacian pyramid-based Eulerian video magnification
- **Real-Time Processing**: Live camera feed processing and display
- **Temporal Filtering**: IIR bandpass filtering for frequency-selective amplification
- **Basic Video Recording**: MJPEG AVI output for processed video
- **Multi-Level Pyramid**: 4-level Laplacian decomposition for multi-scale analysis
- **Color Space Processing**: YCrCb conversion with chroma attenuation

### Technical
- OpenCV-based computer vision pipeline
- Rust implementation with memory-safe processing
- IIR temporal filtering with configurable frequency bands
- Pyramid construction and reconstruction algorithms
- Basic parameter configuration system

---

## Version History Summary

- **v2.0.0**: Stabilization + Masking + Performance optimization
- **v1.5.0**: Phase-based processing with steerable pyramids  
- **v1.2.0**: Automatic frequency sweep and analysis
- **v1.1.0**: Enhanced trackbar interface and real-time feedback
- **v1.0.0**: QuickTime-compatible video export
- **v0.2.0**: Interactive parameter control with trackbars
- **v0.1.0**: Initial motion magnification implementation