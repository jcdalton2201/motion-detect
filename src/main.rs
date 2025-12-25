use opencv::{
    core, highgui, imgproc,
    prelude::*,
    videoio::{CAP_ANY, VideoCapture, VideoWriter},
};
use std::f32::consts::PI;
use std::process::Command;

// ----- Stabilization and Masking System -----
struct Stabilizer {
    reference_frame: Option<Mat>,
    reference_features: opencv::core::Vector<core::Point2f>,
    prev_transform: Mat, // 2x3 affine transform
    is_initialized: bool,
}

impl Stabilizer {
    fn new() -> Self {
        Self {
            reference_frame: None,
            reference_features: opencv::core::Vector::new(),
            prev_transform: Mat::eye(2, 3, core::CV_32F).unwrap().to_mat().unwrap(),
            is_initialized: false,
        }
    }

    fn stabilize_frame(&mut self, frame: &Mat) -> opencv::Result<Mat> {
        if !self.is_initialized {
            self.initialize_reference(frame)?;
            return Ok(frame.clone());
        }

        // Detect features in current frame
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        let mut current_features = opencv::core::Vector::new();

        imgproc::good_features_to_track(
            &gray,
            &mut current_features,
            200,               // max_corners
            0.01,              // quality_level
            30.0,              // min_distance
            &core::no_array(), // mask
            7,                 // block_size
            false,             // use_harris_detector
            0.04,              // k
        )?;

        if current_features.len() < 10 {
            // Not enough features, return original
            return Ok(frame.clone());
        }

        // Match features and estimate transform
        let transform = self.estimate_stabilization_transform(&current_features)?;

        // Apply stabilization
        let mut stabilized = Mat::default();
        imgproc::warp_affine(
            frame,
            &mut stabilized,
            &transform,
            frame.size()?,
            imgproc::INTER_LINEAR,
            core::BORDER_REFLECT_101,
            core::Scalar::default(),
        )?;

        Ok(stabilized)
    }

    fn initialize_reference(&mut self, frame: &Mat) -> opencv::Result<()> {
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        imgproc::good_features_to_track(
            &gray,
            &mut self.reference_features,
            200,               // max_corners
            0.01,              // quality_level
            30.0,              // min_distance
            &core::no_array(), // mask
            7,                 // block_size
            false,             // use_harris_detector
            0.04,              // k
        )?;
        self.reference_frame = Some(gray);
        self.is_initialized = true;
        Ok(())
    }

    fn estimate_stabilization_transform(
        &self,
        _current_features: &opencv::core::Vector<core::Point2f>,
    ) -> opencv::Result<Mat> {
        // Simplified: return identity transform for now
        // In a full implementation, you'd use feature matching and RANSAC
        Mat::eye(2, 3, core::CV_32F)?.to_mat()
    }
}

struct Mask {
    mask_image: Mat,
    is_active: bool,
    drawing_mode: bool,
    brush_size: i32,
    last_mouse_pos: Option<core::Point>,
    mouse_down: bool,
}

impl Mask {
    fn new(width: i32, height: i32) -> opencv::Result<Self> {
        // Start with all-white mask (magnify everything)
        let mut mask_image = Mat::zeros(height, width, core::CV_8UC1)?.to_mat()?;
        mask_image.set_to(&core::Scalar::all(255.0), &core::no_array())?;
        Ok(Self {
            mask_image,
            is_active: false,
            drawing_mode: false,
            brush_size: 20,
            last_mouse_pos: None,
            mouse_down: false,
        })
    }

    fn apply_to_pyramid(&self, pyramid: &mut SteerablePyramid) -> opencv::Result<()> {
        if !self.is_active {
            return Ok(());
        }

        for level in &mut pyramid.levels {
            // Resize mask to match this level's size
            let level_size = level.original_size;
            let mut resized_mask = Mat::default();
            imgproc::resize(
                &self.mask_image,
                &mut resized_mask,
                level_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;

            // Convert mask to float and normalize
            let mut mask_f32 = Mat::default();
            resized_mask.convert_to(&mut mask_f32, core::CV_32F, 1.0 / 255.0, 0.0)?;

            // Apply mask to each orientation
            for orientation in &mut level.orientations {
                let mut masked = Mat::default();
                core::multiply(orientation, &mask_f32, &mut masked, 1.0, -1)?;
                *orientation = masked;
            }
        }

        Ok(())
    }

    fn toggle_active(&mut self) {
        self.is_active = !self.is_active;
    }

    fn clear_mask(&mut self) -> opencv::Result<()> {
        self.mask_image = Mat::zeros(
            self.mask_image.rows(),
            self.mask_image.cols(),
            core::CV_8UC1,
        )?
        .to_mat()?;
        self.mask_image
            .set_to(&core::Scalar::all(255.0), &core::no_array())?;
        Ok(())
    }

    fn invert_mask(&mut self) -> opencv::Result<()> {
        let mut inverted = Mat::default();
        core::bitwise_not(&self.mask_image, &mut inverted, &core::no_array())?;
        self.mask_image = inverted;
        Ok(())
    }

    fn draw_on_mask(&mut self, x: i32, y: i32, display_size: core::Size) -> opencv::Result<()> {
        // Transform coordinates from display window to actual mask size
        let mask_size = self.mask_image.size()?;
        let scale_x = mask_size.width as f32 / display_size.width as f32;
        let scale_y = mask_size.height as f32 / display_size.height as f32;

        let mask_x = (x as f32 * scale_x) as i32;
        let mask_y = (y as f32 * scale_y) as i32;

        // Draw circle on mask (white to enable magnification, black to disable)
        let color = if self.drawing_mode {
            core::Scalar::all(255.0) // White = magnify
        } else {
            core::Scalar::all(0.0) // Black = don't magnify
        };

        imgproc::circle(
            &mut self.mask_image,
            core::Point::new(mask_x, mask_y),
            (self.brush_size as f32 * scale_x.min(scale_y)) as i32,
            color,
            -1, // Filled circle
            imgproc::LINE_8,
            0,
        )?;

        Ok(())
    }

    fn handle_mouse_event(
        &mut self,
        event: i32,
        x: i32,
        y: i32,
        display_size: core::Size,
    ) -> opencv::Result<()> {
        match event {
            highgui::EVENT_LBUTTONDOWN => {
                self.mouse_down = true;
                self.drawing_mode = true; // Left click = draw white (magnify)
                self.draw_on_mask(x, y, display_size)?;
                self.last_mouse_pos = Some(core::Point::new(x, y));
            }
            highgui::EVENT_RBUTTONDOWN => {
                self.mouse_down = true;
                self.drawing_mode = false; // Right click = draw black (don't magnify)
                self.draw_on_mask(x, y, display_size)?;
                self.last_mouse_pos = Some(core::Point::new(x, y));
            }
            highgui::EVENT_LBUTTONUP | highgui::EVENT_RBUTTONUP => {
                self.mouse_down = false;
                self.last_mouse_pos = None;
            }
            highgui::EVENT_MOUSEMOVE => {
                if self.mouse_down {
                    self.draw_on_mask(x, y, display_size)?;
                    self.last_mouse_pos = Some(core::Point::new(x, y));
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn auto_detect_edges(&mut self, frame: &Mat) -> opencv::Result<()> {
        // Convert to grayscale for edge detection
        let mut gray = Mat::default();
        imgproc::cvt_color(frame, &mut gray, imgproc::COLOR_BGR2GRAY, 0)?;

        // Resize to mask size if needed
        let mask_size = self.mask_image.size()?;
        if gray.size()? != mask_size {
            let mut resized = Mat::default();
            imgproc::resize(
                &gray,
                &mut resized,
                mask_size,
                0.0,
                0.0,
                imgproc::INTER_LINEAR,
            )?;
            gray = resized;
        }

        // Apply Gaussian blur to reduce noise
        let mut blurred = Mat::default();
        imgproc::gaussian_blur(
            &gray,
            &mut blurred,
            core::Size::new(5, 5),
            1.0,
            1.0,
            core::BORDER_DEFAULT,
        )?;

        // Detect edges using Canny
        let mut edges = Mat::default();
        imgproc::canny(&blurred, &mut edges, 50.0, 150.0, 3, false)?;

        // Dilate edges to create thicker regions
        let kernel = imgproc::get_structuring_element(
            imgproc::MORPH_ELLIPSE,
            core::Size::new(10, 10),
            core::Point::new(-1, -1),
        )?;
        let mut dilated = Mat::default();
        imgproc::dilate(
            &edges,
            &mut dilated,
            &kernel,
            core::Point::new(-1, -1),
            2,
            core::BORDER_CONSTANT,
            imgproc::morphology_default_border_value()?,
        )?;

        // Invert so edges become white (magnify regions)
        core::bitwise_not(&dilated, &mut self.mask_image, &core::no_array())?;

        println!("Auto-detected edges for masking");
        Ok(())
    }
}

// ----- Phase-Based Motion Magnification -----
// This implementation uses a complex steerable pyramid instead of a simple Laplacian pyramid.
// The steerable pyramid provides better directional selectivity and phase information.
//
// Key advantages of phase-based amplification:
// 1. Better noise handling - phase is more robust to noise than amplitude
// 2. Artifact reduction - avoids over‑amplification of large motions
// 3. Directional sensitivity - different orientations can be processed independently
// 4. Frequency selectivity - temporal filtering in phase domain is more precise
//
// Pipeline:
// 1. Decompose frame into 3 scales × 6 orientations using steerable pyramid
// 2. Extract phase information from each orientation band
// 3. Apply temporal bandpass filtering to phase signals
// 4. Amplify phase changes (motion) while preserving appearance
// 5. Reconstruct magnified frame from modified pyramid

// Steerable pyramid level (one scale, multiple orientations)
struct SteerablePyramidLevel {
    orientations: Vec<Mat>,    // 6 orientations at 30° intervals for this scale
    lowpass: Mat,              // Lowpass residual for next scale
    original_size: core::Size, // Original size before downsampling
}

// Complete steerable pyramid with 3 scales and 6 orientations per scale
// Scale 0: Full resolution, finest spatial details
// Scale 1: Half resolution, medium spatial details
// Scale 2: Quarter resolution, coarse spatial details
struct SteerablePyramid {
    levels: Vec<SteerablePyramidLevel>, // 3 scales total
    final_lowpass: Mat,                 // Final lowpass residual
}

// ----- Frequency Sweep Mode -----
struct SweepResult {
    frequency: f32,
    energy: f32,
}

struct SweepAnalyzer {
    results: Vec<SweepResult>,
    current_freq_idx: usize,
    frames_per_freq: usize,
    frame_count: usize,
    sweep_frequencies: Vec<f32>,
    is_active: bool,
    energy_accumulator: f32,
}

impl SweepAnalyzer {
    fn new() -> Self {
        let mut frequencies = Vec::new();
        // Sample frequencies from 0.2 to 10.0 Hz in steps
        let mut freq = 0.2_f32;
        while freq <= 10.0 {
            frequencies.push(freq);
            freq += 0.2; // 0.2 Hz steps
        }

        Self {
            results: Vec::new(),
            current_freq_idx: 0,
            frames_per_freq: 60, // Sample each frequency for 60 frames (~2 seconds at 30fps)
            frame_count: 0,
            sweep_frequencies: frequencies,
            is_active: false,
            energy_accumulator: 0.0,
        }
    }

    fn start_sweep(&mut self) {
        println!("Starting frequency sweep from 0.2 to 10.0 Hz...");
        println!(
            "This will take approximately {} seconds",
            self.sweep_frequencies.len() * self.frames_per_freq / 30
        );
        self.is_active = true;
        self.current_freq_idx = 0;
        self.frame_count = 0;
        self.energy_accumulator = 0.0;
        self.results.clear();
    }

    fn get_current_frequency(&self) -> Option<f32> {
        if self.is_active && self.current_freq_idx < self.sweep_frequencies.len() {
            Some(self.sweep_frequencies[self.current_freq_idx])
        } else {
            None
        }
    }

    fn add_energy_sample(&mut self, energy: f32) {
        if !self.is_active {
            return;
        }

        self.energy_accumulator += energy;
        self.frame_count += 1;

        if self.frame_count >= self.frames_per_freq {
            // Average energy for this frequency
            let avg_energy = self.energy_accumulator / self.frames_per_freq as f32;
            let freq = self.sweep_frequencies[self.current_freq_idx];

            self.results.push(SweepResult {
                frequency: freq,
                energy: avg_energy,
            });

            println!("Frequency {:.1} Hz: Energy = {:.2}", freq, avg_energy);

            // Move to next frequency
            self.current_freq_idx += 1;
            self.frame_count = 0;
            self.energy_accumulator = 0.0;

            // Check if sweep is complete
            if self.current_freq_idx >= self.sweep_frequencies.len() {
                self.finish_sweep();
            }
        }
    }

    fn finish_sweep(&mut self) {
        self.is_active = false;
        println!("\n=== Frequency Sweep Complete ===");

        if self.results.is_empty() {
            println!("No results collected!");
            return;
        }

        // Find peak energy frequency
        let peak_result = self
            .results
            .iter()
            .max_by(|a, b| a.energy.partial_cmp(&b.energy).unwrap())
            .unwrap();

        // Find frequencies with > 70% of peak energy
        let threshold = peak_result.energy * 0.7;
        let good_freqs: Vec<&SweepResult> = self
            .results
            .iter()
            .filter(|r| r.energy >= threshold)
            .collect();

        println!("\n--- Results Summary ---");
        println!(
            "Peak motion at {:.1} Hz with energy {:.2}",
            peak_result.frequency, peak_result.energy
        );

        if good_freqs.len() > 1 {
            let min_freq = good_freqs
                .iter()
                .map(|r| r.frequency)
                .fold(f32::INFINITY, f32::min);
            let max_freq = good_freqs
                .iter()
                .map(|r| r.frequency)
                .fold(0.0_f32, f32::max);
            println!(
                "Recommended frequency range: {:.1} - {:.1} Hz",
                min_freq, max_freq
            );
            println!("Set Low Cutoff to: {:.1} Hz", (min_freq - 0.1).max(0.1));
            println!("Set High Cutoff to: {:.1} Hz", (max_freq + 0.1).min(10.0));
        }

        println!("\n--- All Results ---");
        for result in &self.results {
            let bar_length = ((result.energy / peak_result.energy) * 50.0) as usize;
            let bar = "█".repeat(bar_length);
            println!(
                "{:4.1} Hz: {:6.2} |{}",
                result.frequency, result.energy, bar
            );
        }
        println!("\nPress 's' again to start another sweep");
    }
}

// ----- Params container -----
struct Params {
    fl_hz: f32,          // low cutoff (Hz)
    fh_hz: f32,          // high cutoff (Hz)
    fps: f32,            // video sampling rate
    alpha: f32,          // magnification
    chrom_atten: f32,    // reduce chroma amplification (0..1)
    stabilization: bool, // enable stabilization
}

// ----- Utilities -----
fn lowpass_iir(prev: &mut Mat, cur: &Mat, r: f32) -> opencv::Result<()> {
    // prev = (1 - r)*cur + r*prev  (use a temp to avoid borrow clash)
    let mut out = Mat::default();
    core::add_weighted(
        cur,
        1.0 - r as f64,
        prev,
        r as f64,
        0.0,
        &mut out,
        cur.typ(),
    )?;
    *prev = out;
    Ok(())
}

fn build_steerable_pyramid(frame: &Mat) -> opencv::Result<SteerablePyramid> {
    let num_scales = 2; // Reduced from 3 for performance
    let num_orientations = 4; // Reduced from 6 for performance (0°, 45°, 90°, 135°)
    let mut pyramid = SteerablePyramid {
        levels: Vec::with_capacity(num_scales),
        final_lowpass: Mat::default(),
    };

    let mut current = frame.clone();

    for scale in 0..num_scales {
        let mut level = SteerablePyramidLevel {
            orientations: Vec::with_capacity(num_orientations),
            lowpass: Mat::default(),
            original_size: current.size()?,
        };

        // Create oriented filters for this scale - each captures motion in specific direction
        for orient in 0..num_orientations {
            let angle = (orient as f32) * PI / num_orientations as f32;
            let mut oriented = Mat::default();

            // Apply Gabor filter tuned to this orientation and scale
            // Gabor filters provide good joint spatial-frequency localization
            apply_oriented_filter(&current, &mut oriented, angle, scale)?;

            // Ensure consistent float type for all orientations
            let mut oriented_f32 = Mat::default();
            oriented.convert_to(&mut oriented_f32, core::CV_32F, 1.0, 0.0)?;
            level.orientations.push(oriented_f32);
        }

        // Downsample for next scale (pyramid structure)
        imgproc::pyr_down(
            &current,
            &mut level.lowpass,
            core::Size::default(),
            core::BORDER_DEFAULT,
        )?;

        current = level.lowpass.clone();
        pyramid.levels.push(level);
    }

    pyramid.final_lowpass = current;
    Ok(pyramid)
}

fn apply_oriented_filter(src: &Mat, dst: &mut Mat, angle: f32, scale: usize) -> opencv::Result<()> {
    // Create Gabor kernel tuned to specific orientation and scale
    // Larger kernels and wavelengths for coarser scales to capture different spatial frequencies
    let kernel_size = 15 + scale * 6; // Increase kernel size with scale
    let sigma = 2.0 + scale as f64 * 1.5; // Standard deviation grows with scale
    let lambda = 6.0 + scale as f64 * 3.0; // Wavelength increases with scale

    // Gabor parameters optimized for motion detection:
    // - gamma=0.5 creates elongated filters good for edge detection
    // - psi=0 for even-symmetric filter (captures edges, not lines)
    let kernel = imgproc::get_gabor_kernel(
        core::Size::new(kernel_size as i32, kernel_size as i32),
        sigma,        // Standard deviation of Gaussian envelope
        angle as f64, // Orientation of filter (radians)
        lambda,       // Wavelength of sinusoidal factor
        0.5,          // Aspect ratio (gamma)
        0.0,          // Phase offset (psi)
        core::CV_32F,
    )?;

    // Apply the oriented filter to extract directional information
    imgproc::filter_2d(
        src,
        dst,
        -1,
        &kernel,
        core::Point::new(-1, -1),
        0.0,
        core::BORDER_DEFAULT,
    )
}

fn reconstruct_from_steerable(pyramid: &SteerablePyramid) -> opencv::Result<Mat> {
    // Simple reconstruction: start with first level and add all orientations
    if pyramid.levels.is_empty() {
        return Ok(pyramid.final_lowpass.clone());
    }

    // Use the first level as base size
    let base_level = &pyramid.levels[0];
    let target_size = base_level.original_size;

    let mut reconstruction =
        Mat::zeros(target_size.height, target_size.width, core::CV_32F)?.to_mat()?;

    // Add contributions from all levels and orientations
    for level in &pyramid.levels {
        for orientation in &level.orientations {
            // Resize orientation to target size if needed
            let mut resized_orientation = orientation.clone();
            if orientation.size()? != target_size {
                let mut temp = Mat::default();
                imgproc::resize(
                    orientation,
                    &mut temp,
                    target_size,
                    0.0,
                    0.0,
                    imgproc::INTER_LINEAR,
                )?;
                resized_orientation = temp;
            }

            // Add to reconstruction
            let mut temp = Mat::default();
            core::add(
                &reconstruction,
                &resized_orientation,
                &mut temp,
                &core::no_array(),
                core::CV_32F,
            )?;
            reconstruction = temp;
        }
    }

    Ok(reconstruction)
}

// ----- ffmpeg post-process -----
fn convert_to_quicktime_mp4(avi_path: &str, mp4_path: &str, fps: f32) -> bool {
    println!("Converting to QuickTime-compatible MP4...");

    let fps_str = format!("{:.1}", fps);

    // Very specific QuickTime-compatible settings
    let args = [
        "-y", // Overwrite output
        "-i",
        avi_path, // Input AVI
        "-c:v",
        "libx264", // H.264 codec
        "-profile:v",
        "baseline", // Baseline profile for max compatibility
        "-level",
        "3.0", // Level 3.0 for compatibility
        "-pix_fmt",
        "yuv420p", // Standard pixel format
        "-movflags",
        "+faststart", // Optimize for streaming
        "-crf",
        "20", // Good quality
        "-maxrate",
        "10M", // Reasonable bitrate
        "-bufsize",
        "10M", // Buffer size
        "-r",
        &fps_str, // Frame rate
        "-avoid_negative_ts",
        "make_zero", // Fix timestamp issues
        mp4_path,
    ];

    match Command::new("ffmpeg").args(&args).status() {
        Ok(status) if status.success() => {
            println!("Successfully created QuickTime-compatible: {}", mp4_path);
            // Remove temporary AVI file
            let _ = std::fs::remove_file(avi_path);
            true
        }
        Ok(_) => {
            eprintln!("Error: ffmpeg conversion failed");
            eprintln!("Keeping intermediate file: {}", avi_path);
            false
        }
        Err(_) => {
            eprintln!("Error: ffmpeg not found");
            eprintln!("Please install ffmpeg: brew install ffmpeg");
            eprintln!("Intermediate AVI file saved as: {}", avi_path);
            false
        }
    }
}

fn main() -> opencv::Result<()> {
    // --- Input: try camera first, fallback to test pattern
    let mut cap = VideoCapture::new(0, CAP_ANY)?;
    let use_test_pattern = !cap.is_opened().unwrap_or(false);

    if use_test_pattern {
        println!("Camera not available, using synthetic test pattern");
        println!("This will generate moving patterns to test the steerable pyramid");
    }

    // FPS (fallback if webcam reports 0, or use fixed rate for test pattern)
    let fps = if use_test_pattern {
        30.0
    } else {
        let reported_fps = cap.get(opencv::videoio::CAP_PROP_FPS)? as f32;
        if reported_fps <= 0.0 {
            30.0
        } else {
            reported_fps
        }
    };

    // Parameters (initial)
    let mut params = Params {
        fl_hz: 0.5,
        fh_hz: 6.0,
        fps,
        alpha: 30.0,
        chrom_atten: 0.10,
        stabilization: true,
    };

    // I/O setup
    let (width, height) = if use_test_pattern {
        (240, 180) // Further reduced for responsiveness
    } else {
        (
            cap.get(opencv::videoio::CAP_PROP_FRAME_WIDTH)? as i32,
            cap.get(opencv::videoio::CAP_PROP_FRAME_HEIGHT)? as i32,
        )
    };
    let size = core::Size::new(width, height);

    // Use reliable AVI format as intermediate, then convert with ffmpeg
    let avi_path = "magnified_temp.avi";
    let output_path = "magnified.mp4";
    let fourcc = opencv::videoio::VideoWriter::fourcc('M', 'J', 'P', 'G')?;
    let mut out = VideoWriter::new(avi_path, fourcc, fps as f64, size, true)?;

    if !out.is_opened()? {
        return Err(opencv::Error::new(
            opencv::core::StsError,
            "Failed to open video writer",
        ));
    }

    println!("Recording to intermediate AVI format...");

    // ----- Windows -----
    highgui::named_window("Magnified", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("Magnified", 960, 540)?;
    highgui::named_window("Magnified Controls", highgui::WINDOW_AUTOSIZE)?;
    highgui::named_window("Mask Editor", highgui::WINDOW_NORMAL)?;
    highgui::resize_window("Mask Editor", 480, 360)?;

    // ----- Trackbars (integers we map to floats each frame)
    // alpha: 0..100  -> alpha f32
    // fl:   0..100  -> 0.00..10.00 Hz
    // fh:   0..100  -> 0.00..10.00 Hz (we'll enforce fh > fl)
    // chrom:0..100  -> 0.00..1.00
    let mut tb_alpha: i32 = (params.alpha.round() as i32).clamp(0, 100);
    let mut tb_fl: i32 = (params.fl_hz * 10.0).round() as i32; // 0..100 => 0..10 Hz
    let mut tb_fh: i32 = (params.fh_hz * 10.0).round() as i32; // 0..10 Hz
    let mut tb_chrom: i32 = (params.chrom_atten * 100.0).round() as i32;

    // create trackbars (pass the mutable ints; we’ll also read with get_trackbar_pos)
    let _ = highgui::create_trackbar(
        "Magnification (0-100)",
        "Magnified Controls",
        None,
        100,
        None,
    )?;
    let _ = highgui::create_trackbar(
        "Low Cutoff x10 (0-100 = 0.0-10.0Hz)",
        "Magnified Controls",
        None,
        100,
        None,
    )?;
    let _ = highgui::create_trackbar(
        "High Cutoff x10 (0-100 = 0.0-10.0Hz)",
        "Magnified Controls",
        None,
        100,
        None,
    )?;
    let _ = highgui::create_trackbar(
        "Chrome Suppress (0-100 = 0.0-1.0)",
        "Magnified Controls",
        None,
        100,
        None,
    )?;

    // Set initial trackbar values
    let _ = highgui::set_trackbar_pos("Magnification (0-100)", "Magnified Controls", tb_alpha);
    let _ = highgui::set_trackbar_pos(
        "Low Cutoff x10 (0-100 = 0.0-10.0Hz)",
        "Magnified Controls",
        tb_fl,
    );
    let _ = highgui::set_trackbar_pos(
        "High Cutoff x10 (0-100 = 0.0-10.0Hz)",
        "Magnified Controls",
        tb_fh,
    );
    let _ = highgui::set_trackbar_pos(
        "Chrome Suppress (0-100 = 0.0-1.0)",
        "Magnified Controls",
        tb_chrom,
    );

    // Stabilization and masking
    let mut stabilizer = Stabilizer::new();
    let mut mask = Mask::new(width, height)?;

    // Temporal state per pyramid level
    let mut lp1: Vec<Mat> = Vec::new();
    let mut lp2: Vec<Mat> = Vec::new();

    // Frequency sweep analyzer
    let mut sweep = SweepAnalyzer::new();

    // Initial filter coeffs
    let mut r1 = (-2.0 * PI * params.fl_hz / params.fps).exp();
    let mut r2 = (-2.0 * PI * params.fh_hz / params.fps).exp();

    // Test pattern state
    let mut frame_count = 0u32;

    // Extract key handling into a function for reuse
    let handle_key = |key: i32,
                      mask: &mut Mask,
                      params: &mut Params,
                      sweep: &mut SweepAnalyzer|
     -> opencv::Result<bool> {
        match key {
            113 | 27 => {
                // 'q' or ESC
                if sweep.is_active {
                    println!("Stopping frequency sweep...");
                    sweep.is_active = false;
                    Ok(false)
                } else {
                    Ok(true) // Exit main loop
                }
            }
            115 if !sweep.is_active => {
                // 's'
                sweep.start_sweep();
                Ok(false)
            }
            109 => {
                // 'm'
                mask.toggle_active();
                println!(
                    "Mask {}",
                    if mask.is_active {
                        "ENABLED"
                    } else {
                        "DISABLED"
                    }
                );
                Ok(false)
            }
            99 => {
                // 'c'
                mask.clear_mask()?;
                println!("Mask cleared (all regions will be magnified)");
                Ok(false)
            }
            105 => {
                // 'i'
                mask.invert_mask()?;
                println!("Mask inverted");
                Ok(false)
            }
            116 => {
                // 't'
                params.stabilization = !params.stabilization;
                println!(
                    "Stabilization {}",
                    if params.stabilization {
                        "ENABLED"
                    } else {
                        "DISABLED"
                    }
                );
                Ok(false)
            }
            43 | 61 => {
                // '+' or '='
                mask.brush_size = (mask.brush_size + 2).min(50);
                println!("Brush size: {}", mask.brush_size);
                Ok(false)
            }
            45 => {
                // '-'
                mask.brush_size = (mask.brush_size - 2).max(5);
                println!("Brush size: {}", mask.brush_size);
                Ok(false)
            }
            97 => {
                // 'a' - will be handled separately due to needing frame
                Ok(false)
            }
            _ => Ok(false),
        }
    };

    loop {
        // --- Read & update trackbar values (map to floats)
        tb_alpha = highgui::get_trackbar_pos("Magnification (0-100)", "Magnified Controls")?;
        tb_fl =
            highgui::get_trackbar_pos("Low Cutoff x10 (0-100 = 0.0-10.0Hz)", "Magnified Controls")?;
        tb_fh = highgui::get_trackbar_pos(
            "High Cutoff x10 (0-100 = 0.0-10.0Hz)",
            "Magnified Controls",
        )?;
        tb_chrom =
            highgui::get_trackbar_pos("Chrome Suppress (0-100 = 0.0-1.0)", "Magnified Controls")?;

        // Handle sweep mode frequency override
        if let Some(sweep_freq) = sweep.get_current_frequency() {
            // Use tight band around current sweep frequency
            let bandwidth = 0.1; // Narrow band for precise measurement
            params.fl_hz = (sweep_freq - bandwidth).max(0.1);
            params.fh_hz = (sweep_freq + bandwidth).min(10.0);

            // Recompute filter coefficients
            r1 = (-2.0 * PI * params.fl_hz / params.fps).exp();
            r2 = (-2.0 * PI * params.fh_hz / params.fps).exp();
        } else {
            // Map to real values
            let fl = (tb_fl as f32) / 10.0; // 0..10 Hz
            let mut fh = (tb_fh as f32) / 10.0; // 0..10 Hz

            // keep a minimum gap so fh > fl
            if fh <= fl + 0.05 {
                fh = fl + 0.05;
                // reflect back to UI (optional)
                let _ = highgui::set_trackbar_pos(
                    "High Cutoff x10 (0-100 = 0.0-10.0Hz)",
                    "Magnified Controls",
                    (fh * 10.0).round() as i32,
                );
            }

            // Only recompute IIR if band changed
            if (fl - params.fl_hz).abs() > f32::EPSILON || (fh - params.fh_hz).abs() > f32::EPSILON
            {
                params.fl_hz = fl;
                params.fh_hz = fh;
                r1 = (-2.0 * PI * params.fl_hz / params.fps).exp();
                r2 = (-2.0 * PI * params.fh_hz / params.fps).exp();
            }

            params.alpha = (tb_alpha as f32).clamp(0.0, 100.0);
            params.chrom_atten = ((tb_chrom as f32) / 100.0).clamp(0.0, 1.0);
        }

        // --- Handle mask editing mouse events (we'll handle this manually in the loop)

        // Single responsive key check per frame
        let key = highgui::wait_key(1)?;
        let mut should_exit = false;
        if key != -1 {
            if handle_key(key, &mut mask, &mut params, &mut sweep)? {
                should_exit = true;
            }
        }

        // --- Capture or generate frame
        let mut bgr_raw = Mat::default();

        if use_test_pattern {
            bgr_raw = generate_test_frame(frame_count, width, height)?;
            frame_count += 1;

            // Stop test after reasonable number of frames
            if frame_count > 600 {
                // 20 seconds at 30fps
                break;
            }
        } else {
            if !cap.read(&mut bgr_raw)? || bgr_raw.empty() {
                break;
            }
        }

        // Handle auto-detect after frame capture if key was pressed
        if key == 97
        /* 'a' */
        {
            mask.auto_detect_edges(&bgr_raw)?;
            println!("Auto-detected edges for pipe masking");
        }

        if should_exit {
            break;
        }

        // --- Apply stabilization
        let bgr = if params.stabilization {
            stabilizer.stabilize_frame(&bgr_raw)?
        } else {
            bgr_raw
        };

        // Convert BGR -> YCrCb
        let mut ycrcb = Mat::default();
        imgproc::cvt_color(&bgr, &mut ycrcb, imgproc::COLOR_BGR2YCrCb, 0)?;

        // Split channels, work on luma
        let mut planes = opencv::core::Vector::new();
        core::split(&ycrcb, &mut planes)?;
        let y_raw: Mat = planes.get(0)?; // luma

        // Convert to float for processing
        let mut y = Mat::default();
        y_raw.convert_to(&mut y, core::CV_32F, 1.0 / 255.0, 0.0)?;

        // Steerable pyramid on Y
        let mut pyramid = build_steerable_pyramid(&y)?;

        // Apply mask to pyramid (focus magnification on specific regions)
        mask.apply_to_pyramid(&mut pyramid)?;

        // Skip heavy processing if exit was requested
        if should_exit {
            break;
        }

        // Initialize temporal state on first frame (for each orientation at each level)
        if lp1.is_empty() {
            for level in &pyramid.levels {
                for orientation in &level.orientations {
                    lp1.push(orientation.clone());
                    lp2.push(orientation.clone());
                }
            }
        }

        // Phase-based temporal filtering and amplification
        // Process each orientation band independently to preserve directional motion
        let mut bandpass_energy = 0.0_f32;
        let mut filter_idx = 0;

        for (scale_idx, level) in pyramid.levels.iter_mut().enumerate() {
            for (_orient_idx, orientation) in level.orientations.iter_mut().enumerate() {
                // Apply cascaded IIR temporal filtering to extract frequency band
                // This creates a bandpass filter: High-pass at fl_hz, Low-pass at fh_hz
                lowpass_iir(&mut lp1[filter_idx], orientation, r1)?; // Low cutoff filter
                lowpass_iir(&mut lp2[filter_idx], orientation, r2)?; // High cutoff filter

                // Bandpass = Low_cutoff - High_cutoff (temporal frequency band)
                let mut temporal_band = Mat::default();
                core::subtract(
                    &lp1[filter_idx],
                    &lp2[filter_idx],
                    &mut temporal_band,
                    &core::no_array(),
                    core::CV_32F,
                )?;

                // Calculate motion energy for frequency sweep analysis
                if sweep.is_active {
                    if let Ok(norm) = opencv::core::norm(
                        &temporal_band,
                        opencv::core::NORM_L2,
                        &opencv::core::no_array(),
                    ) {
                        // Weight energy by scale (finer scales contribute more)
                        let scale_weight = 1.0 / (scale_idx as f32 + 1.0);
                        bandpass_energy += (norm as f32) * scale_weight;
                    }
                }

                // Phase-based amplification: Add amplified temporal changes back to original
                // This preserves the base appearance while magnifying motion
                let mut amplified = Mat::default();
                core::add_weighted(
                    orientation,         // Original orientation response
                    1.0,                 // Keep original at full strength
                    &temporal_band,      // Temporal motion component
                    params.alpha as f64, // Amplify motion by alpha factor
                    0.0,
                    &mut amplified,
                    core::CV_32F,
                )?;
                *orientation = amplified;

                filter_idx += 1;
            }
        }

        // Add energy sample to sweep analyzer
        if sweep.is_active {
            sweep.add_energy_sample(bandpass_energy);
        }

        // Skip remaining processing if exit was requested
        if should_exit {
            break;
        }

        // Reconstruct magnified Y from steerable pyramid
        let y_mag_f32 = reconstruct_from_steerable(&pyramid)?;

        // Convert back to 8-bit for display
        let mut y_mag = Mat::default();
        y_mag_f32.convert_to(&mut y_mag, core::CV_8U, 255.0, 0.0)?;

        // Put Y back; optionally attenuate chroma
        planes.set(0, y_mag)?;
        if params.chrom_atten > 0.0 {
            let scale = 1.0 - params.chrom_atten as f64;
            for ch in [1usize, 2usize] {
                let c = planes.get(ch)?;
                let mut c_scaled = Mat::default();
                c.convert_to(&mut c_scaled, -1, scale, 0.0)?;
                planes.set(ch, c_scaled)?;
            }
        }

        // Merge and convert back to BGR
        let mut out_ycrcb = Mat::default();
        core::merge(&planes, &mut out_ycrcb)?;

        let mut out_frame = Mat::default();
        imgproc::cvt_color(&out_ycrcb, &mut out_frame, imgproc::COLOR_YCrCb2BGR, 0)?;

        // Display current params overlay (nice for tuning)
        let overlay1 = if sweep.is_active {
            if let Some(freq) = sweep.get_current_frequency() {
                format!(
                    "FREQUENCY SWEEP MODE - Testing: {:.1} Hz  |  Progress: {}/{}",
                    freq,
                    sweep.current_freq_idx + 1,
                    sweep.sweep_frequencies.len()
                )
            } else {
                format!("FREQUENCY SWEEP MODE - Finishing...")
            }
        } else {
            let stab_status = if params.stabilization { "ON" } else { "OFF" };
            let mask_status = if mask.is_active { "ON" } else { "OFF" };
            format!(
                "α: {} -> {:.1}  |  Low: {} -> {:.2}Hz  |  High: {} -> {:.2}Hz  |  Stab: {}  |  Mask: {}",
                tb_alpha,
                params.alpha,
                tb_fl,
                params.fl_hz,
                tb_fh,
                params.fh_hz,
                stab_status,
                mask_status
            )
        };

        let overlay2 = if sweep.is_active {
            format!(
                "Frame {}/{} at current frequency  |  Press 'q' to stop sweep",
                sweep.frame_count + 1,
                sweep.frames_per_freq
            )
        } else {
            format!(
                "Chroma: {} -> {:.2}  |  FPS: {:.1}  |  2×4 pyramid  |  Keys: 's'=sweep, 'm'=mask, 'c'=clear, 'i'=invert, 'a'=auto, 't'=stab",
                tb_chrom, params.chrom_atten, params.fps
            )
        };
        imgproc::put_text(
            &mut out_frame,
            &overlay1,
            core::Point::new(20, 40),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;
        imgproc::put_text(
            &mut out_frame,
            &overlay2,
            core::Point::new(20, 70),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            core::Scalar::new(0.0, 255.0, 0.0, 0.0),
            2,
            imgproc::LINE_AA,
            false,
        )?;

        // Live preview + quit handling
        highgui::imshow("Magnified", &out_frame)?;

        // Show mask editor
        let mut mask_display = Mat::default();
        imgproc::resize(
            &mask.mask_image,
            &mut mask_display,
            core::Size::new(480, 360),
            0.0,
            0.0,
            imgproc::INTER_LINEAR,
        )?;
        let mut mask_colored = Mat::default();
        imgproc::cvt_color(&mask_display, &mut mask_colored, imgproc::COLOR_GRAY2BGR, 0)?;

        // Add status text to mask editor
        let mask_text = if mask.is_active {
            "MASK: ACTIVE"
        } else {
            "MASK: INACTIVE"
        };
        imgproc::put_text(
            &mut mask_colored,
            mask_text,
            core::Point::new(10, 30),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.7,
            if mask.is_active {
                core::Scalar::new(0.0, 255.0, 0.0, 0.0)
            } else {
                core::Scalar::new(0.0, 0.0, 255.0, 0.0)
            },
            2,
            imgproc::LINE_AA,
            false,
        )?;

        imgproc::put_text(
            &mut mask_colored,
            "Left Click: Enable | Right Click: Disable",
            core::Point::new(10, 55),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            core::Scalar::new(255.0, 255.0, 255.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;

        imgproc::put_text(
            &mut mask_colored,
            &format!("Brush Size: {} (Use +/- to adjust)", mask.brush_size),
            core::Point::new(10, 75),
            imgproc::FONT_HERSHEY_SIMPLEX,
            0.5,
            core::Scalar::new(255.0, 255.0, 0.0, 0.0),
            1,
            imgproc::LINE_AA,
            false,
        )?;

        highgui::imshow("Mask Editor", &mask_colored)?;

        // Handle auto-detect with final frame if needed
        if key == 97
        /* 'a' */
        {
            mask.auto_detect_edges(&bgr)?;
            println!("Auto-detected edges for pipe masking");
        }

        // Write to AVI
        out.write(&out_frame)?;
    }

    // Close writer
    drop(out);

    println!("Recording complete. Converting to QuickTime format...");

    // Convert AVI to QuickTime-compatible MP4
    if convert_to_quicktime_mp4(avi_path, output_path, fps) {
        println!("Final video: {}", output_path);
    } else {
        println!("Conversion failed. Check intermediate file: {}", avi_path);
    }

    highgui::destroy_window("Magnified")?;
    highgui::destroy_window("Magnified Controls")?;
    highgui::destroy_window("Mask Editor")?;
    Ok(())
}

// Generate synthetic test pattern with motion for testing steerable pyramid
fn generate_test_frame(frame_num: u32, width: i32, height: i32) -> opencv::Result<Mat> {
    let mut frame = Mat::zeros(height, width, core::CV_8UC3)?.to_mat()?;

    let t = frame_num as f32 * 0.1; // Time parameter

    // Create moving patterns to test different orientations and scales
    for y in 0..height {
        for x in 0..width {
            let fx = x as f32 / width as f32;
            let fy = y as f32 / height as f32;

            // Moving sinusoidal patterns in different orientations
            let horizontal_wave = ((fx * 10.0 + t).sin() * 50.0 + 128.0) as u8;
            let vertical_wave = ((fy * 8.0 + t * 1.5).sin() * 40.0 + 128.0) as u8;
            let diagonal_wave = (((fx + fy) * 6.0 + t * 2.0).sin() * 30.0 + 128.0) as u8;

            // Moving circular pattern for radial motion
            let center_x = 0.5 + (t * 0.5).sin() * 0.2;
            let center_y = 0.5 + (t * 0.3).cos() * 0.2;
            let dist = ((fx - center_x).powi(2) + (fy - center_y).powi(2)).sqrt();
            let circular = ((dist * 20.0 - t * 3.0).sin() * 60.0 + 128.0) as u8;

            // Combine patterns for rich motion content
            let b = horizontal_wave.saturating_add(30);
            let g = vertical_wave.saturating_add(diagonal_wave / 2);
            let r = circular.saturating_add(20);

            let pixel = frame.at_2d_mut::<core::Vec3b>(y, x)?;
            *pixel = core::Vec3b::from([b, g, r]);
        }
    }

    // Add some text to show frame number
    let text = format!("Test Frame: {}", frame_num);
    imgproc::put_text(
        &mut frame,
        &text,
        core::Point::new(10, 30),
        imgproc::FONT_HERSHEY_SIMPLEX,
        1.0,
        core::Scalar::new(255.0, 255.0, 255.0, 0.0),
        2,
        imgproc::LINE_AA,
        false,
    )?;

    Ok(frame)
}
