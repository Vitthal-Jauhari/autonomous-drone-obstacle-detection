import argparse
import cv2
import glob
import matplotlib
import numpy as np
import os
import torch
import torch.nn.functional as F
import time

from depth_anything_v2.dpt import DepthAnythingV2

class VideoDepthProcessorCUDA:
    def __init__(self, encoder='vits', device='cuda'):
        """
        CUDA-optimized Depth Anything V2 model for video processing
        
        Args:
            encoder: Model size ('vits', 'vitb', 'vitl', 'vitg')
            device: Device to run on ('cuda' or 'cpu')
        """
        # Set device
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name()}")
            
        # Model configurations
        self.model_configs = {
            'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
            'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
            'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
            'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
        }
        
        self.encoder = encoder
        self.model = self._load_model()
        self.cmap = matplotlib.colormaps.get_cmap('Spectral_r')
        
        # Precompute mean and std on the correct device
        self.mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(self.device)
        self.std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(self.device)
        
    def _load_model(self):
        """Load and initialize the model with CUDA optimizations"""
        print(f"Loading {self.encoder} model...")
        
        # Initialize model
        model = DepthAnythingV2(**self.model_configs[self.encoder])
        
        # Load weights
        checkpoint_path = f'checkpoints/depth_anything_v2_{self.encoder}.pth'
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
            
        model.load_state_dict(torch.load(checkpoint_path, map_location='cpu', weights_only=True))
        
        # Move to device and set to eval mode
        model = model.to(self.device).eval()
        
        # Enable CUDA optimizations
        if self.device.type == 'cuda':
            model = model.half()  # Use FP16 for faster inference
            torch.backends.cudnn.benchmark = True  # Optimize CUDNN
            
        print(f"Model loaded successfully on {self.device}")
        return model
        
    def warmup(self, input_size=518, num_warmup=3):
        """Warm up the model for consistent timing"""
        print("Warming up model...")
        dummy_input = torch.randn(1, 3, input_size, input_size, device=self.device)
        if self.device.type == 'cuda':
            dummy_input = dummy_input.half()
            
        with torch.no_grad():
            for _ in range(num_warmup):
                _ = self.model(dummy_input)
                
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        print("Warmup complete")
        
    def process_frame(self, frame, input_size=518):
        if frame is None:
            raise ValueError("Input frame is None")
            
        original_height, original_width = frame.shape[:2]
        
        # Preprocessing - convert BGR to RGB and normalize
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        
        # Convert to tensor and resize
        image_tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0)
        image_tensor = F.interpolate(
            image_tensor, 
            (input_size, input_size), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Move to device first, then normalize
        image_tensor = image_tensor.to(self.device)
        
        # Normalize with proper device placement
        mean = self.mean.to(self.device)
        std = self.std.to(self.device)
        image_tensor = (image_tensor - mean) / std
        
        if self.device.type == 'cuda':
            image_tensor = image_tensor.half()
            
        # Inference with timing
        start_time = time.time()
        
        with torch.no_grad():
            depth = self.model(image_tensor)
            
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
            
        inference_time = time.time() - start_time
        
        # Post-processing
        depth = F.interpolate(
            depth, 
            (original_height, original_width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Convert back to numpy
        depth = depth.squeeze().cpu().numpy()
        
        return depth, inference_time
    
    def process_depth_for_display(self, depth, grayscale=False):
        """
        Process depth map for display
        
        Args:
            depth: Raw depth map
            grayscale: Whether to use grayscale output
            
        Returns:
            depth_vis: Visualizable depth map
        """
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        
        # Apply colormap or grayscale
        if grayscale:
            depth_vis = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (self.cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
        return depth_vis
    
    def process_video(self, video_path, output_dir='./vis_video_depth', 
                     input_size=518, grayscale=False, pred_only=True, 
                     show_preview=False):
        """
        Process a video file with depth estimation using CUDA
        
        Args:
            video_path: Path to input video
            output_dir: Directory to save output
            input_size: Model input size
            grayscale: Whether to use grayscale output
            pred_only: Whether to output only depth or side-by-side
            show_preview: Whether to show preview during processing
        """
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.model = self.model.float().to(self.device)  # Convert back to float32 for CPU

        # Check if video exists
        if not os.path.isfile(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
            
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
            
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video properties: {frame_width}x{frame_height}, {fps:.2f} FPS, {total_frames} frames")
        
        # Set up output video
        output_filename = os.path.splitext(os.path.basename(video_path))[0]
        output_path = os.path.join(output_dir, f"{output_filename}_depth.mp4")
        
        # Determine output dimensions
        if pred_only:
            output_width, output_height = frame_width, frame_height
        else:
            margin_width = 50
            output_width = frame_width * 2 + margin_width
            output_height = frame_height
            
        # Initialize video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
        
        # Warm up model
        self.warmup(input_size)
        
        # Process frames
        frame_count = 0
        total_time = 0
        frame_times = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process frame
            depth, inference_time = self.process_frame(frame, input_size)
            total_time += inference_time
            frame_times.append(inference_time)
            frame_count += 1
            
            # Process depth for display
            depth_vis = self.process_depth_for_display(depth, grayscale)
            
            # Create output frame
            if pred_only:
                output_frame = depth_vis
            else:
                separator = np.ones((frame_height, 50, 3), dtype=np.uint8) * 255
                output_frame = cv2.hconcat([frame, separator, depth_vis])
            
            # Write to output video
            out.write(output_frame)
            
            # Show preview if requested
            if show_preview:
                preview = cv2.resize(output_frame, (1280, 720))
                cv2.imshow('Depth Estimation Preview', preview)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            # Print progress
            if frame_count % 10 == 0 or frame_count == total_frames:
                avg_time = total_time / frame_count
                remaining_frames = total_frames - frame_count
                eta = remaining_frames * avg_time
                print(f"Processed {frame_count}/{total_frames} frames "
                      f"({frame_count/total_frames*100:.1f}%), "
                      f"Avg: {avg_time*1000:.1f}ms, "
                      f"ETA: {eta:.1f}s")
        
        # Release resources
        cap.release()
        out.release()
        if show_preview:
            cv2.destroyAllWindows()
            
        # Print summary
        if frame_times:
            avg_fps = frame_count / total_time
            frame_times_ms = [t * 1000 for t in frame_times]
            print(f"\nProcessing complete!")
            print(f"Total frames: {frame_count}")
            print(f"Total time: {total_time:.2f}s")
            print(f"Average FPS: {avg_fps:.2f}")
            print(f"Average time per frame: {1000/avg_fps:.2f}ms")
            print(f"Min time: {min(frame_times_ms):.2f}ms")
            print(f"Max time: {max(frame_times_ms):.2f}ms")
            print(f"Output saved to: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Depth Anything V2 for Videos with CUDA Optimization')
    
    parser.add_argument('--video-path', type=str, required=True, help='Path to input video')
    parser.add_argument('--input-size', type=int, default=378, help='Model input size')
    parser.add_argument('--outdir', type=str, default='./vis_video_depth', help='Output directory')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Model encoder type')
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', 
                       help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', 
                       help='do not apply colorful palette')
    parser.add_argument('--preview', dest='preview', action='store_true',
                       help='show preview during processing')
    
    args = parser.parse_args()
    
    # Initialize processor with CUDA
    processor = VideoDepthProcessorCUDA(encoder=args.encoder, device='cuda')
    
    # Process video
    processor.process_video(
        video_path=args.video_path,
        output_dir=args.outdir,
        input_size=args.input_size,
        grayscale=args.grayscale,
        pred_only=args.pred_only,
        show_preview=args.preview
    )

if __name__ == '__main__':
    main()