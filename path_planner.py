#!/usr/bin/env python3
"""
SLAM Map Path Planning System
Generates exploration paths using RRT algorithm on SLAM-generated occupancy grid maps.
"""

import numpy as np
import cv2
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from scipy.spatial import KDTree
from typing import List, Tuple, Optional, Dict
import random
import math


# ============================================================================
# CONSTANTS
# ============================================================================

# SLAM export parameters (from LiDARSensor.py)
GRID_RESOLUTION_MM = 30  # mm per cell (line 217 in LiDARSensor.py)
EXPORT_SCALE_PX_PER_CELL = 4  # pixels per cell (line 719 in LiDARSensor.py)
MM_PER_PIXEL = GRID_RESOLUTION_MM / EXPORT_SCALE_PX_PER_CELL  # 7.5 mm/pixel

# Path planning parameters
OBSTACLE_PADDING_PX = 5  # Safety margin around obstacles
RRT_STEP_SIZE_PX = 30  # Maximum extension distance per iteration
RRT_MAX_ITERATIONS = 5000  # Maximum iterations for RRT
GOAL_BIAS_PROBABILITY = 0.1  # Probability of sampling toward goal


# ============================================================================
# MAP PROCESSING MODULE
# ============================================================================

class MapProcessor:
    """Handles map loading and preprocessing."""
    
    def __init__(self, image_path: str):
        """
        Load and process a SLAM map image.
        
        Args:
            image_path: Path to the PNG map image
        """
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Map image not found: {image_path}")
        
        # Load image
        self.original_image = cv2.imread(str(self.image_path))
        if self.original_image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        # Convert to grayscale
        self.gray_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2GRAY)
        self.height, self.width = self.gray_image.shape
        
        # Extract free space mask
        self.free_space_mask = self._extract_free_space()
        
        # Create padded obstacle map for safety
        self.safe_mask = self._create_safe_mask()
        
        print(f"Loaded map: {self.width}x{self.height} pixels")
        print(f"Free space pixels: {np.sum(self.free_space_mask)}")
        print(f"Safe space pixels: {np.sum(self.safe_mask)}")
    
    def _extract_free_space(self) -> np.ndarray:
        """
        Extract free space from the map.
        White pixels (255) = free space
        Black pixels (0) = obstacles or unknown
        
        Returns:
            Binary mask where True = navigable space
        """
        # Threshold: consider pixels > 200 as free space
        _, binary_mask = cv2.threshold(self.gray_image, 200, 255, cv2.THRESH_BINARY)
        return binary_mask == 255
    
    def _create_safe_mask(self) -> np.ndarray:
        """
        Create a safety-padded version of free space.
        Erodes free space to add safety margin around obstacles.
        
        Returns:
            Binary mask with safety padding applied
        """
        # Convert free space to uint8 for morphological operations
        free_uint8 = (self.free_space_mask * 255).astype(np.uint8)
        
        # Erode to create safety margin (padding around obstacles)
        kernel = np.ones((OBSTACLE_PADDING_PX, OBSTACLE_PADDING_PX), np.uint8)
        eroded = cv2.erode(free_uint8, kernel, iterations=1)
        
        return eroded == 255
    
    def is_point_free(self, x: int, y: int, use_safe_mask: bool = True) -> bool:
        """
        Check if a point is in free space.
        
        Args:
            x, y: Pixel coordinates
            use_safe_mask: If True, use safety-padded mask
            
        Returns:
            True if point is navigable
        """
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return False
        
        mask = self.safe_mask if use_safe_mask else self.free_space_mask
        return mask[y, x]
    
    def is_line_free(self, x1: int, y1: int, x2: int, y2: int, 
                     use_safe_mask: bool = True) -> bool:
        """
        Check if a line segment is entirely in free space.
        
        Args:
            x1, y1: Start point
            x2, y2: End point
            use_safe_mask: If True, use safety-padded mask
            
        Returns:
            True if entire line is navigable
        """
        # Use Bresenham's line algorithm to sample points along line
        num_samples = int(np.hypot(x2 - x1, y2 - y1)) + 1
        xs = np.linspace(x1, x2, num_samples, dtype=int)
        ys = np.linspace(y1, y2, num_samples, dtype=int)
        
        for x, y in zip(xs, ys):
            if not self.is_point_free(x, y, use_safe_mask):
                return False
        
        return True
    
    def get_random_free_point(self, use_safe_mask: bool = True) -> Optional[Tuple[int, int]]:
        """
        Get a random point in free space.
        
        Args:
            use_safe_mask: If True, use safety-padded mask
            
        Returns:
            (x, y) tuple or None if no free space found
        """
        mask = self.safe_mask if use_safe_mask else self.free_space_mask
        free_points = np.argwhere(mask)
        
        if len(free_points) == 0:
            return None
        
        # Random selection
        idx = random.randint(0, len(free_points) - 1)
        y, x = free_points[idx]
        return (x, y)
    
    def get_center_free_point(self, use_safe_mask: bool = True) -> Optional[Tuple[int, int]]:
        """
        Get a free point near the center of the largest connected component.
        
        Args:
            use_safe_mask: If True, use safety-padded mask
            
        Returns:
            (x, y) tuple or None if no free space found
        """
        mask = self.safe_mask if use_safe_mask else self.free_space_mask
        mask_uint8 = (mask * 255).astype(np.uint8)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            mask_uint8, connectivity=8
        )
        
        if num_labels <= 1:  # Only background
            return None
        
        # Find largest component (skip 0 which is background)
        areas = stats[1:, cv2.CC_STAT_AREA]
        largest_idx = np.argmax(areas) + 1
        
        # Get centroid of largest component
        cx, cy = centroids[largest_idx]
        x, y = int(cx), int(cy)
        
        # Verify it's actually free
        if self.is_point_free(x, y, use_safe_mask):
            return (x, y)
        
        # If centroid is not free, find nearest free point
        component_mask = labels == largest_idx
        free_points = np.argwhere(component_mask)
        
        if len(free_points) == 0:
            return None
        
        # Find closest to centroid
        distances = np.sqrt(np.sum((free_points - np.array([cy, cx]))**2, axis=1))
        closest_idx = np.argmin(distances)
        y, x = free_points[closest_idx]
        
        return (x, y)


# ============================================================================
# COORDINATE SYSTEM
# ============================================================================

class CoordinateConverter:
    """Converts between pixel and metric coordinates."""
    
    @staticmethod
    def pixel_to_metric(x_px: float, y_px: float) -> Tuple[float, float]:
        """
        Convert pixel coordinates to metric (mm).
        
        Args:
            x_px, y_px: Pixel coordinates
            
        Returns:
            (x_mm, y_mm) tuple
        """
        x_mm = x_px * MM_PER_PIXEL
        y_mm = y_px * MM_PER_PIXEL
        return (x_mm, y_mm)
    
    @staticmethod
    def metric_to_pixel(x_mm: float, y_mm: float) -> Tuple[int, int]:
        """
        Convert metric (mm) coordinates to pixels.
        
        Args:
            x_mm, y_mm: Metric coordinates in millimeters
            
        Returns:
            (x_px, y_px) tuple
        """
        x_px = int(x_mm / MM_PER_PIXEL)
        y_px = int(y_mm / MM_PER_PIXEL)
        return (x_px, y_px)
    
    @staticmethod
    def path_to_metric(path_px: List[Tuple[int, int]]) -> List[Tuple[float, float]]:
        """
        Convert a path from pixel to metric coordinates.
        
        Args:
            path_px: List of (x, y) tuples in pixels
            
        Returns:
            List of (x, y) tuples in millimeters
        """
        return [CoordinateConverter.pixel_to_metric(x, y) for x, y in path_px]
    
    @staticmethod
    def calculate_path_length(path: List[Tuple[float, float]]) -> float:
        """
        Calculate total path length in mm.
        
        Args:
            path: List of (x, y) tuples
            
        Returns:
            Total path length in millimeters
        """
        if len(path) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            segment_length = math.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            total_length += segment_length
        
        return total_length


# ============================================================================
# RRT PATH PLANNING ALGORITHM
# ============================================================================

class RRTNode:
    """Node in the RRT tree."""
    
    def __init__(self, x: int, y: int, parent: Optional['RRTNode'] = None):
        self.x = x
        self.y = y
        self.parent = parent
    
    def position(self) -> Tuple[int, int]:
        """Get (x, y) position tuple."""
        return (self.x, self.y)
    
    def distance_to(self, x: int, y: int) -> float:
        """Calculate Euclidean distance to a point."""
        return math.sqrt((self.x - x)**2 + (self.y - y)**2)


class RRTPathPlanner:
    """RRT-based path planner for exploration."""
    
    def __init__(self, map_processor: MapProcessor):
        """
        Initialize RRT planner.
        
        Args:
            map_processor: MapProcessor instance with loaded map
        """
        self.map = map_processor
        self.nodes: List[RRTNode] = []
        self.kdtree: Optional[KDTree] = None
    
    def plan_exploration_path(self, num_waypoints: int = 5, 
                            max_iterations: int = RRT_MAX_ITERATIONS,
                            start_point: Optional[Tuple[int, int]] = None) -> List[Tuple[int, int]]:
        """
        Generate an exploration path with multiple waypoints.
        
        Args:
            num_waypoints: Number of waypoints to visit
            max_iterations: Maximum RRT iterations
            start_point: Starting point (x, y) or None for auto
            
        Returns:
            List of (x, y) waypoints in pixels
        """
        # Get starting point
        if start_point is None:
            start_point = self.map.get_center_free_point()
            if start_point is None:
                raise ValueError("No free space found in map")
        
        print(f"Starting from: {start_point}")
        
        # Initialize tree with start point
        self.nodes = [RRTNode(start_point[0], start_point[1])]
        self._rebuild_kdtree()
        
        # Generate random goal points for exploration
        goal_points = self._generate_goal_points(num_waypoints)
        
        # Grow RRT tree
        print(f"Growing RRT tree (max {max_iterations} iterations)...")
        for i in range(max_iterations):
            # Sample random point (with occasional bias toward unexplored goals)
            if random.random() < GOAL_BIAS_PROBABILITY and goal_points:
                # Sample toward an unvisited goal
                sample_point = random.choice(goal_points)
            else:
                # Random sampling in map bounds
                sample_point = self.map.get_random_free_point()
                if sample_point is None:
                    continue
            
            # Extend tree toward sample
            new_node = self._extend_tree(sample_point)
            
            if new_node is not None:
                self.nodes.append(new_node)
                
                # Periodically rebuild KD-tree for efficiency
                if len(self.nodes) % 100 == 0:
                    self._rebuild_kdtree()
                
                # Check if we reached any goal
                for goal in goal_points[:]:
                    if new_node.distance_to(goal[0], goal[1]) < RRT_STEP_SIZE_PX:
                        goal_points.remove(goal)
                        print(f"Reached waypoint {num_waypoints - len(goal_points)}/{num_waypoints}")
            
            # Progress update
            if (i + 1) % 1000 == 0:
                print(f"  Iteration {i + 1}/{max_iterations}, Tree nodes: {len(self.nodes)}")
        
        print(f"RRT complete: {len(self.nodes)} nodes in tree")
        
        # Extract path through waypoints
        path = self._extract_exploration_path(start_point, goal_points, num_waypoints)
        
        # Smooth path
        smoothed_path = self._smooth_path(path)
        
        return smoothed_path
    
    def _generate_goal_points(self, num_points: int) -> List[Tuple[int, int]]:
        """
        Generate random goal points distributed across the map.
        
        Args:
            num_points: Number of goal points to generate
            
        Returns:
            List of (x, y) goal points
        """
        goals = []
        attempts = 0
        max_attempts = num_points * 100
        
        while len(goals) < num_points and attempts < max_attempts:
            point = self.map.get_random_free_point()
            if point is not None:
                # Check if sufficiently far from existing goals
                if not goals or all(math.hypot(point[0] - g[0], point[1] - g[1]) > 50 
                                   for g in goals):
                    goals.append(point)
            attempts += 1
        
        print(f"Generated {len(goals)} exploration goals")
        return goals
    
    def _rebuild_kdtree(self):
        """Rebuild KD-tree for fast nearest neighbor queries."""
        if len(self.nodes) > 0:
            positions = np.array([[node.x, node.y] for node in self.nodes])
            self.kdtree = KDTree(positions)
    
    def _find_nearest_node(self, x: int, y: int) -> RRTNode:
        """
        Find nearest node in tree to a point.
        
        Args:
            x, y: Target point
            
        Returns:
            Nearest RRTNode
        """
        if self.kdtree is None:
            self._rebuild_kdtree()
        
        _, idx = self.kdtree.query([x, y])
        return self.nodes[idx]
    
    def _extend_tree(self, target: Tuple[int, int]) -> Optional[RRTNode]:
        """
        Extend tree toward a target point.
        
        Args:
            target: (x, y) target point
            
        Returns:
            New RRTNode if extension successful, None otherwise
        """
        # Find nearest node
        nearest = self._find_nearest_node(target[0], target[1])
        
        # Calculate direction to target
        dx = target[0] - nearest.x
        dy = target[1] - nearest.y
        distance = math.sqrt(dx**2 + dy**2)
        
        if distance == 0:
            return None
        
        # Limit extension to step size
        if distance > RRT_STEP_SIZE_PX:
            dx = (dx / distance) * RRT_STEP_SIZE_PX
            dy = (dy / distance) * RRT_STEP_SIZE_PX
        
        # Calculate new point
        new_x = int(nearest.x + dx)
        new_y = int(nearest.y + dy)
        
        # Check if path is collision-free
        if self.map.is_line_free(nearest.x, nearest.y, new_x, new_y):
            return RRTNode(new_x, new_y, parent=nearest)
        
        return None
    
    def _extract_exploration_path(self, start: Tuple[int, int], 
                                 remaining_goals: List[Tuple[int, int]],
                                 total_waypoints: int) -> List[Tuple[int, int]]:
        """
        Extract a path that visits as many waypoints as possible.
        
        Args:
            start: Starting point
            remaining_goals: Goals not yet reached
            total_waypoints: Total number of waypoints requested
            
        Returns:
            List of waypoints forming exploration path
        """
        # Find nodes closest to goals (both reached and unreached)
        waypoint_nodes = []
        
        # Add reached waypoints by finding leaf nodes far from start
        leaf_nodes = [node for node in self.nodes if not any(
            other.parent == node for other in self.nodes
        )]
        
        # Sort by distance from start
        leaf_nodes.sort(key=lambda n: n.distance_to(start[0], start[1]), reverse=True)
        
        # Select diverse waypoints
        selected_waypoints = [start]
        for node in leaf_nodes:
            # Check if sufficiently far from existing waypoints
            if all(math.hypot(node.x - w[0], node.y - w[1]) > RRT_STEP_SIZE_PX * 2 
                   for w in selected_waypoints):
                selected_waypoints.append(node.position())
                
                if len(selected_waypoints) >= total_waypoints:
                    break
        
        # If we don't have enough, add random explored nodes
        if len(selected_waypoints) < total_waypoints:
            for node in self.nodes[::len(self.nodes) // (total_waypoints - len(selected_waypoints) + 1)]:
                if node.position() not in selected_waypoints:
                    selected_waypoints.append(node.position())
                    if len(selected_waypoints) >= total_waypoints:
                        break
        
        return selected_waypoints
    
    def _smooth_path(self, path: List[Tuple[int, int]], 
                    iterations: int = 3) -> List[Tuple[int, int]]:
        """
        Smooth path by removing unnecessary waypoints.
        
        Args:
            path: Original path
            iterations: Number of smoothing passes
            
        Returns:
            Smoothed path
        """
        if len(path) <= 2:
            return path
        
        smoothed = list(path)
        
        for _ in range(iterations):
            i = 0
            while i < len(smoothed) - 2:
                # Try to connect point i directly to point i+2
                x1, y1 = smoothed[i]
                x2, y2 = smoothed[i + 2]
                
                if self.map.is_line_free(x1, y1, x2, y2):
                    # Remove intermediate point
                    smoothed.pop(i + 1)
                else:
                    i += 1
        
        return smoothed


# ============================================================================
# VISUALIZATION & OUTPUT
# ============================================================================

class PathVisualizer:
    """Handles path visualization and export."""
    
    def __init__(self, map_processor: MapProcessor):
        """
        Initialize visualizer.
        
        Args:
            map_processor: MapProcessor instance
        """
        self.map = map_processor
    
    def draw_path(self, path_px: List[Tuple[int, int]], 
                  color: Tuple[int, int, int] = (0, 0, 255),
                  thickness: int = 3) -> np.ndarray:
        """
        Draw path on map image.
        
        Args:
            path_px: Path in pixel coordinates
            color: BGR color tuple
            thickness: Line thickness
            
        Returns:
            Image with path drawn
        """
        # Create copy of original image
        output_image = self.map.original_image.copy()
        
        # Draw path segments
        for i in range(len(path_px) - 1):
            pt1 = path_px[i]
            pt2 = path_px[i + 1]
            cv2.line(output_image, pt1, pt2, color, thickness)
        
        # Draw waypoint markers
        for i, point in enumerate(path_px):
            # Draw circle
            cv2.circle(output_image, point, 6, color, -1)
            cv2.circle(output_image, point, 8, (255, 255, 255), 2)
            
            # Draw waypoint number
            if i == 0:
                label = "START"
                label_color = (0, 255, 0)
            elif i == len(path_px) - 1:
                label = "END"
                label_color = (255, 0, 0)
            else:
                label = str(i)
                label_color = (255, 255, 255)
            
            cv2.putText(output_image, label, 
                       (point[0] + 12, point[1] + 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, label_color, 2)
        
        return output_image
    
    def save_visualization(self, path_px: List[Tuple[int, int]], 
                          output_path: str):
        """
        Save path visualization to file.
        
        Args:
            path_px: Path in pixel coordinates
            output_path: Output file path
        """
        output_image = self.draw_path(path_px)
        cv2.imwrite(output_path, output_image)
        print(f"Saved visualization: {output_path}")
    
    def export_coordinates(self, path_px: List[Tuple[int, int]], 
                          path_mm: List[Tuple[float, float]],
                          output_path: str,
                          metadata: Optional[Dict] = None):
        """
        Export path coordinates to JSON file.
        
        Args:
            path_px: Path in pixel coordinates
            path_mm: Path in metric coordinates
            output_path: Output JSON file path
            metadata: Additional metadata to include
        """
        # Calculate path statistics
        path_length_mm = CoordinateConverter.calculate_path_length(path_mm)
        
        # Build output data
        data = {
            "source_map": self.map.image_path.name,
            "timestamp": datetime.now().isoformat(),
            "resolution_mm_per_pixel": MM_PER_PIXEL,
            "grid_resolution_mm": GRID_RESOLUTION_MM,
            "waypoints_pixel": path_px,
            "waypoints_mm": [[round(x, 2), round(y, 2)] for x, y in path_mm],
            "path_length_mm": round(path_length_mm, 2),
            "path_length_m": round(path_length_mm / 1000, 3),
            "num_waypoints": len(path_px)
        }
        
        # Add metadata if provided
        if metadata:
            data.update(metadata)
        
        # Write to file
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Saved coordinates: {output_path}")


# ============================================================================
# MAIN PROGRAM
# ============================================================================

def main():
    """Main program entry point."""
    
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="SLAM Map Path Planning System - Generates exploration paths using RRT"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input SLAM map image (PNG)'
    )
    parser.add_argument(
        '--num-waypoints',
        type=int,
        default=5,
        help='Number of exploration waypoints (default: 5)'
    )
    parser.add_argument(
        '--max-distance',
        type=int,
        default=50,
        help='Maximum path segment length in pixels (default: 50)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='path_outputs',
        help='Directory for output files (default: path_outputs/)'
    )
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Show interactive visualization (requires display)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=None,
        help='Random seed for reproducibility'
    )
    
    args = parser.parse_args()
    
    # Set random seed if provided
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        print(f"Using random seed: {args.seed}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("SLAM MAP PATH PLANNING SYSTEM")
    print("=" * 70)
    print(f"Input map: {args.input}")
    print(f"Output directory: {output_dir}")
    print(f"Waypoints: {args.num_waypoints}")
    print("=" * 70)
    
    # Load and process map
    print("\n[1/4] Loading map...")
    map_processor = MapProcessor(args.input)
    
    # Plan path
    print("\n[2/4] Planning exploration path...")
    start_time = time.time()
    
    planner = RRTPathPlanner(map_processor)
    path_px = planner.plan_exploration_path(
        num_waypoints=args.num_waypoints,
        max_iterations=RRT_MAX_ITERATIONS
    )
    
    planning_time = time.time() - start_time
    print(f"Planning completed in {planning_time:.2f} seconds")
    
    # Convert to metric coordinates
    print("\n[3/4] Converting coordinates...")
    path_mm = CoordinateConverter.path_to_metric(path_px)
    path_length_mm = CoordinateConverter.calculate_path_length(path_mm)
    
    print(f"Path length: {path_length_mm:.1f} mm ({path_length_mm/1000:.2f} m)")
    print(f"Waypoints: {len(path_px)}")
    
    # Generate output filenames
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    map_name = Path(args.input).stem
    viz_filename = output_dir / f"path_map_{map_name}_{timestamp}.png"
    json_filename = output_dir / f"path_coords_{map_name}_{timestamp}.json"
    
    # Visualize and export
    print("\n[4/4] Exporting results...")
    visualizer = PathVisualizer(map_processor)
    
    visualizer.save_visualization(path_px, str(viz_filename))
    visualizer.export_coordinates(
        path_px, path_mm, str(json_filename),
        metadata={
            "planning_time_seconds": round(planning_time, 2),
            "rrt_nodes_explored": len(planner.nodes)
        }
    )
    
    # Display summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Waypoints: {len(path_px)}")
    print(f"Path length: {path_length_mm/1000:.2f} m")
    print(f"Planning time: {planning_time:.2f} s")
    print(f"RRT nodes: {len(planner.nodes)}")
    print(f"\nOutputs:")
    print(f"  Visualization: {viz_filename}")
    print(f"  Coordinates:   {json_filename}")
    print("=" * 70)
    
    # Show visualization if requested
    if args.visualize:
        print("\nDisplaying visualization (press any key to close)...")
        output_image = visualizer.draw_path(path_px)
        cv2.imshow("Path Planning Result", output_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    print("\nDone!")


if __name__ == "__main__":
    main()



