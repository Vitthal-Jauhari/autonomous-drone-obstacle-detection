import argparse
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import os
import time
import matplotlib
from depth_anything_v2.dpt import DepthAnythingV2

class ColorBasedObstacleAvoidance:
    def __init__(self):
        # Color ranges for different distance zones (in HSV for better detection)
        self.critical_color_range = ((0, 100, 100), (10, 255, 255))  # Red - Very close
        self.warning_color_range = ((10, 100, 100), (25, 255, 255))   # Orange - Close
        self.safe_color_range = ((40, 100, 100), (80, 255, 255))      # Green - Safe
        
        # Thresholds for obstacle detection (percentage of pixels in danger zone)
        self.critical_threshold = 0.05  # 5% of center area is red → CRITICAL
        self.warning_threshold = 0.15   # 15% of center area is orange → WARNING
        
        # Navigation parameters
        self.center_region = (0.35, 0.65, 0.35, 0.65)  # x_start, x_end, y_start, y_end
        
    def detect_obstacle_from_depth_colors(self, depth_visualization, params=None):
        """
        Analyze the colored depth map to detect obstacles based on colors
        """
        # Use custom parameters if provided, otherwise use defaults
        critical_thresh = params['critical_thresh'] if params else self.critical_threshold
        warning_thresh = params['warning_thresh'] if params else self.warning_threshold
        
        height, width = depth_visualization.shape[:2]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(depth_visualization, cv2.COLOR_BGR2HSV)
        
        # Define center region for analysis
        x_start = int(self.center_region[0] * width)
        x_end = int(self.center_region[1] * width)
        y_start = int(self.center_region[2] * height)
        y_end = int(self.center_region[3] * height)
        
        center_region = hsv[y_start:y_end, x_start:x_end]
        total_pixels = center_region.shape[0] * center_region.shape[1]
        
        if total_pixels == 0:
            return {'obstacle': False, 'critical': False, 'avoid_direction': 'none'}
        
        # Create masks for different danger colors
        critical_mask = cv2.inRange(center_region, 
                                  np.array(self.critical_color_range[0]), 
                                  np.array(self.critical_color_range[1]))
        
        warning_mask = cv2.inRange(center_region, 
                                 np.array(self.warning_color_range[0]), 
                                 np.array(self.warning_color_range[1]))
        
        # Calculate percentages
        critical_percent = np.sum(critical_mask > 0) / total_pixels
        warning_percent = np.sum(warning_mask > 0) / total_pixels
        
        # Determine obstacle status
        if critical_percent > critical_thresh:
            return {
                'obstacle': True,
                'critical': True,
                'avoid_direction': self.get_avoidance_direction(hsv),
                'critical_percent': critical_percent,
                'warning_percent': warning_percent
            }
        elif warning_percent > warning_thresh:
            return {
                'obstacle': True,
                'critical': False,
                'avoid_direction': self.get_avoidance_direction(hsv),
                'critical_percent': critical_percent,
                'warning_percent': warning_percent
            }
        else:
            return {
                'obstacle': False,
                'critical': False,
                'avoid_direction': 'none',
                'critical_percent': critical_percent,
                'warning_percent': warning_percent
            }
    
    def get_avoidance_direction(self, hsv_frame):
        """
        Determine which direction to avoid based on side clearances
        """
        height, width = hsv_frame.shape[:2]
        
        # Analyze left and right sides
        left_region = hsv_frame[:, :width//3]    # Left third
        right_region = hsv_frame[:, 2*width//3:] # Right third
        
        # Check for safe (green) areas on sides
        safe_mask_left = cv2.inRange(left_region, 
                                   np.array(self.safe_color_range[0]), 
                                   np.array(self.safe_color_range[1]))
        
        safe_mask_right = cv2.inRange(right_region, 
                                    np.array(self.safe_color_range[0]), 
                                    np.array(self.safe_color_range[1]))
        
        safe_pixels_left = np.sum(safe_mask_left > 0)
        safe_pixels_right = np.sum(safe_mask_right > 0)
        
        # More safe area on left → turn right, and vice versa
        if safe_pixels_left > safe_pixels_right * 1.5:  # 50% more clearance on left
            return 'right'
        elif safe_pixels_right > safe_pixels_left * 1.5:
            return 'left'
        else:
            # Equal clearance or unclear, default to right
            return 'right'
    
    def generate_navigation_commands(self, obstacle_info, speed_factor=1.0):
        """
        Convert obstacle detection to navigation commands
        """
        commands = {
            'vx': 0.5 * speed_factor,  # Default forward speed
            'vy': 0.0,
            'vz': 0.0,
            'yaw': 0.0,
            'obstacle_detected': obstacle_info['obstacle'],
            'critical': obstacle_info['critical']
        }
        
        if obstacle_info['critical']:
            # EMERGENCY: Stop and back up
            commands['vx'] = -0.3 * speed_factor  # Reverse
            commands['vy'] = 0.0
            commands['yaw'] = 0.0
            
        elif obstacle_info['obstacle']:
            # Obstacle detected, avoid it
            if obstacle_info['avoid_direction'] == 'left':
                commands['vy'] = -0.4 * speed_factor  # Move left
                commands['yaw'] = 0.3 * speed_factor   # Turn left
                commands['vx'] = 0.2 * speed_factor    # Slow down
            elif obstacle_info['avoid_direction'] == 'right':
                commands['vy'] = 0.4 * speed_factor    # Move right
                commands['yaw'] = -0.3 * speed_factor  # Turn right
                commands['vx'] = 0.2 * speed_factor    # Slow down
        
        return commands

class AutonomousDroneNavigator:
    def __init__(self, encoder='vits', device='cuda'):
        """
        Autonomous navigation system with real-time depth estimation and obstacle avoidance
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
        
        # Navigation parameters
        self.safe_distance = 3.0  # meters
        self.critical_distance = 0.5  # meters
        self.danger_zones = {
            'center': (0.3, 0.7, 0.3, 0.7),  # x_start, x_end, y_start, y_end (relative coordinates)
            'left': (0.0, 0.3, 0.3, 0.7),
            'right': (0.7, 1.0, 0.3, 0.7),
            'top': (0.3, 0.7, 0.0, 0.3),
            'bottom': (0.3, 0.7, 0.7, 1.0)
        }
        
        # Color-based obstacle avoidance
        self.color_analyzer = ColorBasedObstacleAvoidance()
        
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
        """Convert depth map to visualization using colormap or grayscale"""
        # Normalize depth for visualization
        depth_normalized = (depth - depth.min()) / (depth.max() - depth.min()) * 255.0
        depth_normalized = depth_normalized.astype(np.uint8)
        
        if grayscale:
            depth_vis = np.repeat(depth_normalized[..., np.newaxis], 3, axis=-1)
        else:
            depth_vis = (self.cmap(depth_normalized)[:, :, :3] * 255)[:, :, ::-1].astype(np.uint8)
            
        return depth_vis
    
    def run_color_based_avoidance(self, camera_id=0, input_size=378):
        """
        Simplified obstacle avoidance using color detection
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam with ID {camera_id}")
            
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Starting color-based obstacle avoidance")
        print("Press 'q' to quit")
        
        # Warm up model
        self.warmup(input_size)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Get depth visualization (with colors)
            depth, _ = self.process_frame(frame, input_size)
            depth_vis = self.process_depth_for_display(depth, grayscale=False)
            
            # Analyze colors for obstacles
            obstacle_info = self.color_analyzer.detect_obstacle_from_depth_colors(depth_vis)
            
            # Generate navigation commands
            commands = self.color_analyzer.generate_navigation_commands(obstacle_info)
            
            # Visualize
            self.visualize_avoidance(frame, depth_vis, obstacle_info, commands)
            
            # Send commands to drone (in real application)
            if obstacle_info['obstacle']:
                print(f"OBSTACLE! Avoiding {obstacle_info['avoid_direction']}")
                print(f"Commands: Vx={commands['vx']}, Vy={commands['vy']}, Yaw={commands['yaw']}")
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def visualize_avoidance(self, frame, depth_vis, obstacle_info, commands):
        """
        Simple visualization for obstacle avoidance
        """
        # Combine original and depth view
        combined = cv2.hconcat([frame, depth_vis])
        
        # Add status text
        status = "CLEAR PATH"
        color = (0, 255, 0)  # Green
        
        if obstacle_info['critical']:
            status = "CRITICAL OBSTACLE! BACKING UP"
            color = (0, 0, 255)  # Red
        elif obstacle_info['obstacle']:
            status = f"OBSTACLE - AVOIDING {obstacle_info['avoid_direction'].upper()}"
            color = (0, 165, 255)  # Orange
        
        cv2.putText(combined, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show command info
        cmd_text = f"Vx: {commands['vx']:.1f}, Vy: {commands['vy']:.1f}, Yaw: {commands['yaw']:.1f}"
        cv2.putText(combined, cmd_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        cv2.imshow('Color-Based Obstacle Avoidance', combined)

    def run_color_based_avoidance_with_tuning(self, camera_id=0, input_size=378):
        """
        Enhanced version with real-time parameter tuning
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Could not open webcam with ID {camera_id}")
            
        # Set camera resolution for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        # Default parameters (can be adjusted in real-time)
        params = {
            'critical_thresh': 0.05,
            'warning_thresh': 0.15,
        }
        
        # Create trackbars for real-time adjustment
        cv2.namedWindow('Parameters')
        cv2.createTrackbar('Critical %', 'Parameters', 5, 20, lambda x: None)  # 0-20%
        cv2.createTrackbar('Warning %', 'Parameters', 15, 30, lambda x: None)   # 0-30%
        cv2.createTrackbar('Speed', 'Parameters', 50, 100, lambda x: None)      # 0-100%
        
        print("Starting color-based obstacle avoidance with tuning")
        print("Press 'q' to quit")
        
        # Warm up model
        self.warmup(input_size)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Update parameters from trackbars
            params['critical_thresh'] = cv2.getTrackbarPos('Critical %', 'Parameters') / 100.0
            params['warning_thresh'] = cv2.getTrackbarPos('Warning %', 'Parameters') / 100.0
            speed_factor = cv2.getTrackbarPos('Speed', 'Parameters') / 100.0
            
            # Get depth visualization
            depth, _ = self.process_frame(frame, input_size)
            depth_vis = self.process_depth_for_display(depth, grayscale=False)
            
            # Analyze obstacles with current parameters
            obstacle_info = self.color_analyzer.detect_obstacle_from_depth_colors(depth_vis, params)
            
            # Generate commands with current speed
            commands = self.color_analyzer.generate_navigation_commands(obstacle_info, speed_factor)
            
            # Visualize with parameter info
            self.visualize_avoidance_with_params(frame, depth_vis, obstacle_info, commands, params, speed_factor)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def visualize_avoidance_with_params(self, frame, depth_vis, obstacle_info, commands, params, speed_factor):
        """
        Enhanced visualization showing current parameters
        """
        # Combine original and depth view
        combined = cv2.hconcat([frame, depth_vis])
        
        # Add status text
        status = "CLEAR PATH"
        color = (0, 255, 0)  # Green
        
        if obstacle_info['critical']:
            status = "CRITICAL OBSTACLE! BACKING UP"
            color = (0, 0, 255)  # Red
        elif obstacle_info['obstacle']:
            status = f"OBSTACLE - AVOIDING {obstacle_info['avoid_direction'].upper()}"
            color = (0, 165, 255)  # Orange
        
        cv2.putText(combined, status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        
        # Show command info
        cmd_text = f"Vx: {commands['vx']:.1f}, Vy: {commands['vy']:.1f}, Yaw: {commands['yaw']:.1f}"
        cv2.putText(combined, cmd_text, (10, 70), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Show parameter info
        param_text = f"Critical: {params['critical_thresh']*100:.1f}%, Warning: {params['warning_thresh']*100:.1f}%, Speed: {speed_factor*100:.1f}%"
        cv2.putText(combined, param_text, (10, 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Color-Based Obstacle Avoidance (Tuning)', combined)

def main():
    parser = argparse.ArgumentParser(description='Autonomous Drone Navigation with Obstacle Avoidance')
    
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--input-size', type=int, default=378, help='Model input size')
    parser.add_argument('--encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'],
                       help='Model encoder type')
    parser.add_argument('--mode', type=str, default='color', choices=['color', 'color_tuning'],
                       help='Detection mode: color-based or color-based with tuning')
    
    args = parser.parse_args()
    
    # Initialize autonomous navigator
    navigator = AutonomousDroneNavigator(encoder=args.encoder, device='cuda')
    
    # Run appropriate mode
    if args.mode == 'color':
        navigator.run_color_based_avoidance(
            camera_id=args.camera,
            input_size=args.input_size
        )
    elif args.mode == 'color_tuning':
        navigator.run_color_based_avoidance_with_tuning(
            camera_id=args.camera,
            input_size=args.input_size
        )
    else:
        print("Unsupported mode")

if __name__ == '__main__':
    main()