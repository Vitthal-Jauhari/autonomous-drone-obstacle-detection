# Pehle Baaki poori script run karne se pehle ise separately run karana 
# To check if tumhara sab sahi se install hua ki nahin
# After that un-comment the other part

# import torch
# print(f"PyTorch version: {torch.__version__}")
# print(f"CUDA available: {torch.cuda.is_available()}")
# if torch.cuda.is_available():
#     print(f"CUDA version: {torch.version.cuda}")
#     print(f"GPU device: {torch.cuda.get_device_name(0)}")
# else:
#     print("CUDA is not available. Reasons could be:")
#     print("1. No NVIDIA GPU installed")
#     print("2. NVIDIA drivers not installed")
#     print("3. PyTorch was installed without CUDA support")

import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import matplotlib

from depth_anything_v2.dpt import DepthAnythingV2

class RealTimeDepthProcessorCUDA:
    def __init__(self, encoder='vits', device='cuda'):
        """
        CUDA-optimized Depth Anything V2 model for real-time webcam processing
        
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
        
    def warmup(self, input_size=378, num_warmup=3):
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
    
    def process_frame(self, frame, input_size=378):
        """Process a single frame for depth estimation"""
        if frame is None:
            return None, 0
            
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
    
    def run_realtime(self, camera_id=0, input_size=378, grayscale=False, pred_only=False):
        """
        Run real-time depth estimation from webcam
        
        Args:
            camera_id: Webcam device ID (usually 0 for built-in camera)
            input_size: Model input size
            grayscale: Whether to use grayscale output
            pred_only: Whether to show only depth or side-by-side
        """
        if not torch.cuda.is_available():
            print("WARNING: CUDA not available, falling back to CPU")
            self.device = torch.device('cpu')
            self.model = self.model.float().to(self.device)

        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam with ID {camera_id}")
            
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Get actual camera properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Webcam: {frame_width}x{frame_height}, {fps:.2f} FPS")
        print("Press 'q' to quit, 'g' to toggle grayscale, 'p' to toggle preview mode")
        
        # Warm up model
        self.warmup(input_size)
        
        # Performance tracking
        frame_count = 0
        total_time = 0
        fps_history = []
        
        # Main loop
        running = True
        while running:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            # Process frame
            depth, inference_time = self.process_frame(frame, input_size)
            
            if depth is not None:
                total_time += inference_time
                frame_count += 1
                
                # Process depth for display
                depth_vis = self.process_depth_for_display(depth, grayscale)
                
                # Create output frame
                if pred_only:
                    output_frame = depth_vis
                else:
                    # Resize frames to same size if needed
                    if frame.shape != depth_vis.shape:
                        depth_vis = cv2.resize(depth_vis, (frame.shape[1], frame.shape[0]))
                    
                    # Create side-by-side view
                    separator = np.ones((frame_height, 10, 3), dtype=np.uint8) * 255
                    output_frame = cv2.hconcat([frame, separator, depth_vis])
                
                # Calculate current FPS
                current_fps = 1.0 / inference_time if inference_time > 0 else 0
                fps_history.append(current_fps)
                if len(fps_history) > 10:
                    fps_history.pop(0)
                
                avg_fps = sum(fps_history) / len(fps_history) if fps_history else 0
                
                # Display FPS and info
                cv2.putText(output_frame, f"FPS: {avg_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                cv2.putText(output_frame, f"Mode: {'Grayscale' if grayscale else 'Color'}", (10, 70), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(output_frame, f"View: {'Depth Only' if pred_only else 'Side-by-Side'}", (10, 110), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Show the frame
                cv2.imshow('Real-Time Depth Estimation', output_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                running = False
            elif key == ord('g'):
                grayscale = not grayscale
                print(f"Grayscale mode: {'ON' if grayscale else 'OFF'}")
            elif key == ord('p'):
                pred_only = not pred_only
                print(f"Preview mode: {'Depth Only' if pred_only else 'Side-by-Side'}")
        
        # Release resources
        cap.release()
        cv2.destroyAllWindows()
        
        # Print performance summary
        if frame_count > 0:
            print(f"\nProcessing complete!")
            print(f"Total frames processed: {frame_count}")
            print(f"Average FPS: {frame_count/total_time:.2f}")
            print(f"Average time per frame: {total_time/frame_count*1000:.2f}ms")

def main():
    parser = argparse.ArgumentParser(description='Real-Time Depth Estimation with Webcam')
    
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--input-size', type=int, default=378, help='Model input size')
    
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Model encoder type')
    
    parser.add_argument('--pred-only', dest='pred_only', action='store_true', 
                       help='only display the prediction')
    parser.add_argument('--grayscale', dest='grayscale', action='store_true', 
                       help='do not apply colorful palette')
    
    args = parser.parse_args()
    
    # Initialize processor with CUDA
    processor = RealTimeDepthProcessorCUDA(encoder=args.encoder, device='cuda')
    
    # Run real-time processing
    processor.run_realtime(
        camera_id=args.camera,
        input_size=args.input_size,
        grayscale=args.grayscale,
        pred_only=args.pred_only
    )

if __name__ == '__main__':
    main()