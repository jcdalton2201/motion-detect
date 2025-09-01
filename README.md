# 🔍 Advanced Motion Magnification System

A sophisticated **real-time motion magnification** application built in Rust using OpenCV, featuring phase-based amplification with complex steerable pyramids, stabilization, and intelligent masking for professional motion analysis.

![Rust](https://img.shields.io/badge/rust-%23000000.svg?style=for-the-badge&logo=rust&logoColor=white)
![OpenCV](https://img.shields.io/badge/opencv-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-blue.svg?style=for-the-badge)

## ✨ Features

### 🎯 **Core Capabilities**
- **Phase-Based Motion Magnification**: Advanced steerable pyramid decomposition (2 scales × 4 orientations)
- **Real-Time Processing**: Live camera feed with interactive parameter adjustment
- **Frequency Sweep Analysis**: Automatic optimal frequency band detection
- **QuickTime-Compatible Output**: Professional-grade H.264 video export

### 🛡️ **Advanced Processing**
- **Video Stabilization**: Automatic camera shake compensation using feature tracking
- **Intelligent Masking**: Region-specific magnification with background isolation
- **Automatic Edge Detection**: One-click pipe/structural element detection
- **Artifact Reduction**: Phase-based processing for cleaner results

### 🎮 **Interactive Controls**
- **Real-Time Trackbars**: Live adjustment of magnification, frequency bands, and attenuation
- **Hotkey System**: Instant response for all major functions
- **Visual Feedback**: Multi-window interface with mask editor and parameter display
- **Test Pattern Mode**: Built-in synthetic patterns when camera unavailable

## 🚀 Installation

### Prerequisites

- **Rust** (1.70+): [Install Rust](https://rustup.rs/)
- **OpenCV** (4.0+): [Installation Guide](https://opencv.org/get-started/)
- **FFmpeg** (optional): For optimal video encoding
  ```bash
  # macOS
  brew install opencv ffmpeg
  
  # Ubuntu/Debian
  sudo apt install libopencv-dev ffmpeg
  
  # Windows
  # Follow OpenCV Windows installation guide
  ```

### Build & Run

```bash
# Clone repository
git clone https://github.com/yourusername/motion-detect.git
cd motion-detect

# Build and run
cargo run --release
```

## 🎛️ Usage

### Quick Start
1. **Launch**: Run `cargo run --release`
2. **Camera Permission**: Grant camera access when prompted
3. **Live Preview**: Adjust parameters using trackbars
4. **Record**: Video automatically saves as `magnified.mp4`
5. **Quit**: Press `Q` or `ESC` to stop

### Basic Workflow
```
1. Start application → Camera feed appears
2. Adjust Alpha slider → Control magnification strength  
3. Set frequency band → Define motion frequency range
4. Press 'S' → Run frequency sweep for optimal settings
5. Press 'M' → Enable masking for focused magnification
6. Press 'A' → Auto-detect edges for pipe analysis
7. Press 'Q' → Save and quit
```

## ⌨️ Controls & Hotkeys

| Key | Function | Description |
|-----|----------|-------------|
| **Q** / **ESC** | Quit | Exit application or stop frequency sweep |
| **S** | Frequency Sweep | Auto-analyze optimal frequency bands |
| **M** | Toggle Mask | Enable/disable region-specific magnification |
| **A** | Auto-Detect | Automatic edge detection for pipes/structures |
| **T** | Stabilization | Toggle camera shake compensation |
| **C** | Clear Mask | Reset mask to magnify all regions |
| **I** | Invert Mask | Flip magnification regions |
| **+** / **-** | Brush Size | Adjust mask editing brush size |

### Trackbar Controls

- **Magnification (0-100)**: Motion amplification strength
- **Low Cutoff × 10 (0-100)**: Low frequency bound (0.0-10.0 Hz)
- **High Cutoff × 10 (0-100)**: High frequency bound (0.0-10.0 Hz)
- **Chrome Suppress (0-100)**: Color amplification reduction (0.0-1.0)

## 🔬 Technical Details

### Architecture
```
Input Frame → Stabilization → YCrCb Conversion → Steerable Pyramid
     ↓
Luma Channel → 2 Scales × 4 Orientations → Temporal Filtering
     ↓
Bandpass Filter → Phase Amplification → Mask Application → Reconstruction
     ↓
Color Restoration → Display & Recording → H.264 Export
```

### Steerable Pyramid
- **Scales**: 2 levels (full + half resolution)
- **Orientations**: 4 directions (0°, 45°, 90°, 135°)
- **Filters**: Gabor kernel bank with scale-adaptive parameters
- **Processing**: 8 total frequency-orientation bands

### Temporal Filtering
- **Method**: Cascaded IIR lowpass filters
- **Bandpass**: Difference between low and high cutoff frequencies
- **Amplification**: Phase-based motion enhancement
- **Energy Weighting**: Scale-weighted analysis for frequency sweep

## 📋 Use Cases

### 🏭 **Industrial Applications**
- **Pipe Vibration Analysis**: Detect structural fatigue and resonance
- **Bearing Diagnostics**: Identify mechanical wear patterns  
- **Machinery Monitoring**: Analyze equipment health through micro-motions
- **Quality Control**: Detect assembly defects via motion patterns

### 🏥 **Medical & Biological**
- **Pulse Detection**: Non-contact heart rate monitoring
- **Breathing Analysis**: Respiratory pattern assessment
- **Micro-Circulation**: Blood flow visualization
- **Tissue Motion**: Subtle biological movement analysis

### 🏗️ **Civil Engineering**
- **Bridge Monitoring**: Structural health assessment
- **Building Sway**: Wind-induced motion analysis
- **Seismic Response**: Earthquake damage evaluation
- **Infrastructure**: Long-term stability monitoring

### 🔬 **Research & Development**
- **Material Testing**: Fatigue and stress analysis
- **Fluid Dynamics**: Flow visualization
- **Acoustic Analysis**: Sound-induced vibrations
- **Scientific Measurements**: Precise motion quantification

## 📊 Output Specifications

### Video Format
- **Container**: MP4 (QuickTime compatible)
- **Codec**: H.264 (libx264)
- **Profile**: Baseline (maximum compatibility)
- **Pixel Format**: yuv420p (standard)
- **Quality**: CRF 20 (high quality, ~10Mbps max bitrate)

### Frequency Sweep Results
```
=== Frequency Sweep Complete ===
Peak motion at 2.4 Hz with energy 156.32
Recommended frequency range: 2.2 - 2.8 Hz
Set Low Cutoff to: 2.1 Hz  
Set High Cutoff to: 2.9 Hz

--- All Results ---
 0.2 Hz:  12.45 |████
 2.4 Hz: 156.32 |██████████████████████████████████████████████████
 4.8 Hz:  89.21 |████████████████████████████████
```

## 🛠️ Development

### Building from Source
```bash
# Debug build (faster compilation)
cargo build

# Release build (optimized performance)  
cargo build --release

# Run with logging
RUST_LOG=debug cargo run
```

### Dependencies
- **opencv**: Computer vision operations
- **clap**: Command-line argument parsing  
- **env_logger**: Logging framework

### Testing
```bash
# Run unit tests
cargo test

# Test with synthetic patterns (no camera needed)
cargo run -- --test-mode
```

## 🤝 Contributing

We welcome contributions! Here's how to get started:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Contribution Areas
- **Performance Optimization**: SIMD, GPU acceleration, threading
- **New Algorithms**: Different pyramid types, filtering methods
- **UI Improvements**: Better visualization, parameter presets
- **Platform Support**: Windows, mobile platforms
- **Documentation**: Tutorials, examples, API docs

## 📈 Performance

### Benchmarks (1080p input)
- **2×4 Steerable Pyramid**: ~15-25 FPS (real-time)
- **Memory Usage**: ~200-400 MB (depending on resolution)
- **CPU Usage**: 30-60% (single core, optimized build)

### Optimization Tips
- Use **Release build** for best performance (`cargo run --release`)
- Lower **input resolution** for faster processing
- Reduce **pyramid scales/orientations** if needed
- Enable **hardware acceleration** in OpenCV build

## ❓ FAQ

**Q: Why does the camera fail to initialize?**  
A: Grant camera permissions in System Preferences > Security & Privacy > Camera

**Q: Can I use video files instead of live camera?**  
A: Yes! Modify the `VideoCapture::new(0, CAP_ANY)` line to use `VideoCapture::from_file("video.mp4", CAP_ANY)`

**Q: The output video doesn't play in QuickTime**  
A: Ensure FFmpeg is installed. The app automatically converts to QuickTime-compatible H.264

**Q: How do I adjust processing speed?**  
A: Reduce pyramid complexity in `build_steerable_pyramid()` or lower input resolution

**Q: What's the optimal frequency range?**  
A: Use the frequency sweep ('S' key) to automatically determine the best range for your specific motion

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **MIT CSAIL**: Original Eulerian Video Magnification research
- **OpenCV Community**: Computer vision framework
- **Rust Community**: Systems programming language
- **FFmpeg Team**: Video processing capabilities

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/yourusername/motion-detect/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/motion-detect/discussions)
- **Documentation**: [Project Wiki](https://github.com/yourusername/motion-detect/wiki)

---

**⭐ Star this project** if you find it useful!

Built with ❤️ in Rust 🦀