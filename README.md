# Rotation-Based Fasciculation Analysis Using Optical Flow

A research-oriented computer vision framework for analyzing rotational motion patterns in synthetic fasciculation simulations using dense optical flow and multi-scale region-based motion estimation.

The project focuses on detecting and quantifying localized rotational dynamics from biomedical-inspired synthetic motion videos. Dense optical flow is computed using the Farneback algorithm, followed by rotational field estimation across multiple rectangular regions of interest (ROIs). The framework supports motion normalization, temporal smoothing, visualization, and CSV-based quantitative analysis.

---

# Projects

## Multi-Scale Rotation Analysis
Implements region-based rotational motion estimation using dense optical flow across multiple ROI sizes.

### Features
- Dense optical flow estimation using Farneback Optical Flow
- Frame-wise rotational field computation
- Multi-scale ROI-based motion quantification
- Rotation normalization using perimeter scaling
- Temporal smoothing with moving average filtering
- CSV export for downstream analysis
- Visualization of rotational dynamics over time

---

## Synthetic Fasciculation Motion Simulation
Analyzes synthetic biomedical motion videos representing muscle fasciculation-like rotational patterns.

### Applications
- Fasciculation motion analysis
- Biomedical motion quantification
- Muscle activity simulation
- Motion dynamics evaluation
- Synthetic medical video analysis
- Optical flow research

---

# Methodology

The framework follows the following processing pipeline:

1. Video frame acquisition
2. Dense optical flow estimation
3. Velocity field extraction
4. Local rotational motion computation
5. Multi-scale ROI averaging
6. Temporal smoothing
7. Rotation normalization
8. Visualization and CSV export

---

# Technologies

- Python
- OpenCV
- NumPy
- Matplotlib
- CSV Processing

---

# Algorithms

## Dense Optical Flow
Uses the Farneback dense optical flow algorithm implemented in OpenCV for estimating pixel-wise motion between consecutive frames.

## Rotation Estimation
Local rotational dynamics are computed from spatial derivatives of horizontal and vertical flow components.

## Motion Normalization
Rotation values are normalized using ROI perimeter scaling to enable comparison across multiple region sizes.

---

# Datasets

## Synthetic Fasciculation Dataset
Custom-generated synthetic videos simulating rotational and localized muscle-like motion patterns.

### Characteristics
- Frame-wise rotational motion
- Controlled motion dynamics
- Multi-scale region analysis
- Biomedical-inspired synthetic deformation

---

# Outputs

The framework generates:

- Rotation dynamics plots
- Frame-wise motion metrics
- Multi-scale rotational analysis
- CSV files containing normalized rotation measurements

---

# References

## Optical Flow
- Farneback, G. Two-Frame Motion Estimation Based on Polynomial Expansion.
- Lucas, B. D., & Kanade, T. An Iterative Image Registration Technique with an Application to Stereo Vision.

## Deep Learning for Optical Flow
- Dosovitskiy, A. et al. FlowNet: Learning Optical Flow with Convolutional Networks.
- Teed, Z. and Deng, J. RAFT: Recurrent All-Pairs Field Transforms for Optical Flow.

## Biomedical Motion Analysis
- Research on optical flow-based biomedical motion quantification
- Motion analysis in synthetic medical imaging datasets

---

# Topics

computer-vision  
optical-flow  
dense-optical-flow  
opencv  
python  
motion-analysis  
biomedical-imaging  
fasciculation  
synthetic-data  
rotation-analysis  
video-processing  
biomedical-signal-processing  
research  
motion-quantification  
computer-vision-research  
