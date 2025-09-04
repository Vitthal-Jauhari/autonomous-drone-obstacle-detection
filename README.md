
# Obstacle Detection Using Depth-Anything-V2

This repository implements **obstacle detection** using the Depth-Anything V2 depth estimation models. It is built on top of the original [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) project.  

For complete model details, usage instructions, and pre-trained checkpoints, please refer to the [original Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2).

## Features

- Depth estimation on images and videos
- Simple scripts to visualize depth maps
- Compatible with Small, Base, Large, and (future) Giant Depth-Anything-V2 models
- GPU/CPU support
- Obstacle detection for autonomous drone applications

## Requirements

- **Python 3.11** (tested and recommended)
- PyTorch 2.0+
- CUDA 11.7+ (for GPU support) or CPU

## Usage

1. Clone this repository:  
```bash
git clone https://github.com/Vitthal-Jauhari/autonomous-drone-obstacle-detection.git
cd autonomous-drone-obstacle-detection
```

2. Install requirements (tested with Python 3.11):  
```bash
pip install -r requirements.txt
```

3. Put the Depth-Anything-V2 model checkpoints in the `checkpoints` folder.  

4. Run on images:  
```bash
python run.py --encoder <vits|vitb|vitl> --img-path <path_to_images> --outdir <output_folder>
```

5. Run on videos:  
```bash
python run_video.py --encoder <vits|vitb|vitl> --video-path <path_to_video> --outdir <output_folder>
```

6. Run obstacle detection using webcam or any other video input
```bash
python realtime_depth.py
```

## Project Structure

```
autonomous-drone-obstacle-detection/
├── checkpoints/          # Model weights (.pth files)
├── models/              # ONNX models
├── assets/              # Example images and videos
│   └── examples_video/
├── vis_video_depth/     # Output depth visualization
├── run.py              # Image processing script
├── run_video.py        # Video processing script
└── requirements.txt    # Python dependencies
```

## Notes

- For advanced usage, model configurations, or fine-tuning, see the [parent repo](https://github.com/DepthAnything/Depth-Anything-V2).  
- Designed to focus on **obstacle detection**, leveraging depth maps for drone or robotics applications.
- **Tested with Python 3.11** - recommended for compatibility
- Large model files are managed using Git LFS

## License

This project inherits licenses from the original Depth-Anything-V2 repository.

## Acknowledgments

Based on the [Depth-Anything-V2](https://github.com/DepthAnything/Depth-Anything-V2) project.
