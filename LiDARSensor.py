import threading
import json
from rplidar import RPLidar, RPLidarException
import time
import serial.tools.list_ports
import os
import pygame
import math
from collections import deque
import numpy as np
from scipy.spatial import KDTree
from scipy.ndimage import binary_closing
from scipy.spatial import KDTree
from scipy.ndimage import binary_closing
from numba import jit
import argparse
import socket
import struct

# --- SLAM Implementation ---
def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape

    # get number of dimensions
    m = A.shape[1]

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)
    AA = A - centroid_A
    BB = B - centroid_B

    # rotation matrix
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    R = np.dot(Vt.T, U.T)

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R, centroid_A.T)

    # homogeneous transformation
    T = np.identity(m+1)
    T[:m, :m] = R
    T[:m, m] = t

    return T, R, t

@jit(nopython=True)
def bresenham_line(x0, y0, x1, y1):
    """
    Bresenham's Line Algorithm
    Returns a list of (x, y) tuples from (x0, y0) to (x1, y1)
    """
    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))
    return points

def compute_normals(points, k=5):
    """
    Estimate normals for a 2D point cloud using k-nearest neighbors and PCA.
    Returns: N x 2 array of normal vectors.
    """
    if len(points) < k:
        return np.zeros((len(points), 2)) # Fallback
        
    tree = KDTree(points)
    normals = []
    
    # Query k neighbors for all points at once
    # k+1 because the point itself is included
    dists, indices = tree.query(points, k=k+1)
    
    for i, idxs in enumerate(indices):
        neighbors = points[idxs]
        # PCA
        mean = np.mean(neighbors, axis=0)
        centered = neighbors - mean
        cov = np.dot(centered.T, centered)
        values, vectors = np.linalg.eig(cov)
        
        # Normal is eigenvector corresponding to smallest eigenvalue
        min_idx = np.argmin(values)
        normal = vectors[:, min_idx]
        normals.append(normal)
        
    return np.array(normals)

def icp(A, B, init_pose=(0,0,0), max_iterations=20, tolerance=0.001):
    """
    The Iterative Closest Point method: aligns source A to target B
    Uses Point-to-Line metric via projected targets.
    """
    # Initial transform
    src = np.copy(A)
    
    # Pre-compute Target Info
    # 1. Build KD-Tree for B (Efficiency)
    target_tree = KDTree(B)
    
    # 2. Compute Normals for B (Point-to-Line)
    # Only if B has enough points
    use_plicp = len(B) > 10
    if use_plicp:
        B_normals = compute_normals(B, k=6)
    
    # Apply initial estimate
    theta = init_pose[2]
    c, s = math.cos(theta), math.sin(theta)
    R_init = np.array([[c, -s], [s, c]])
    src = np.dot(src, R_init.T)
    src += init_pose[:2]
    
    prev_error = 0
    total_T = np.identity(3)
    
    # Handle initial offset in matrix form
    T_init = np.identity(3)
    T_init[:2, :2] = R_init
    T_init[:2, 2] = init_pose[:2]
    total_T = np.dot(T_init, total_T)

    for i in range(max_iterations):
        # 1. Find correspondence
        # Query tree for all src points
        distances, indices = target_tree.query(src)
        
        target_points = B[indices]
        
        # 2. Optimize
        if use_plicp:
            # Point-to-Line Improvement:
            # Project src points onto the plane defined by (target_point, normal)
            # q_proj = p - dot(p - q, n) * n
            # But wait, we want to align 'src' to 'target line'.
            # We treat 'q_proj' as the point 'src' should try to reach to satisfy the planar constraint.
            
            normals = B_normals[indices]
            
            # Vector from point in B to point in A
            diff = src - target_points
            
            # Project this vector onto the normal
            # dot product row-wise
            dist_along_normal = np.sum(diff * normals, axis=1)
            
            # The target point on the "line" is src shifted back along the normal
            # dest = src - dist * normal
            # This works effectively as allowing sliding along the wall
            target_points_proj = src - (normals * dist_along_normal[:, np.newaxis])
            
            # Use these projected points as targets for standard SVD solver
            T, _, _ = best_fit_transform(src, target_points_proj)
        else:
             # Standard Point-to-Point
             T, _, _ = best_fit_transform(src, target_points)

        # 3. Update
        src_h = np.ones((src.shape[0], 3))
        src_h[:,:2] = src
        src_h = np.dot(T, src_h.T).T
        src = src_h[:,:2]

        total_T = np.dot(T, total_T)

        # Check error (Mean Distance)
        mean_error = np.mean(distances)
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    return total_T, distances

class SimpleSLAM:
    def __init__(self):
        # PROBABILISTIC GRID MAP
        # Key: (int_x, int_y) tuple representing 30mm grid cells
        # Value: [sum_x, sum_y, count]
        self.global_grid = {} 
        self.GRID_RESOLUTION = 30 # mm
        
        self.pose = np.array([0.0, 0.0, 0.0]) # x, y, theta
        self.path = [] # History of poses
        
        self.scan_buffer = [] # Buffer to accumulate scans before SLAM update
        self.SCAN_BUFFER_SIZE = 4 # How many scans to merge (robustness)
        self.prev_dense_scan = None
        
    def process_scan(self, scan_points):
        """
        Accumulates scans to create a dense 'Keyframe'.
        Runs ICP only when buffer is full.
        """
        # 1. Convert to Cartesian (Robot Frame)
        current_scan = []
        for p in scan_points:
            if p.distance > 0:
                rad = math.radians(p.angle)
                x = p.distance * math.cos(rad)
                y = p.distance * math.sin(rad)
                current_scan.append([x, y])
        
        # Add to buffer
        if len(current_scan) > 10:
             self.scan_buffer.extend(current_scan)
             
        # Only process if buffer is full (Signal Integration)
        # We assume robot moves slowly enough that 4 scans (~0.5s) is "static" enough
        # or that the blur is acceptable for specific density.
        current_rev = scan_points[0].revolution if scan_points else 0
        
        # Check if we have enough data (using a simple counter or checking length)
        # Since process_scan is called once per revolution roughly
        # We need a counter in the class, or just check list length logic
        # But process_scan is called per frame? No, per revolution based on main loop.
        # So we just count calls?
        # Let's use the length of the buffer as a proxy? No, scans vary.
        # Let's add a counter.
        
        if not hasattr(self, 'buffer_count'):
             self.buffer_count = 0
        self.buffer_count += 1
        
        if self.buffer_count < self.SCAN_BUFFER_SIZE:
             return
             
        # --- PROCESS KEYFRAME ---
        self.buffer_count = 0
        if not self.scan_buffer:
             return
             
        dense_scan = np.array(self.scan_buffer)
        self.scan_buffer = [] # Clear for next batch
        
        # DOWNSAMPLE if too huge (ICP speed)
        # 4 scans * 200 pts = 800 pts. limit to 400 for speed?
        if len(dense_scan) > 400:
             indices = np.linspace(0, len(dense_scan)-1, 400, dtype=int)
             dense_scan_small = dense_scan[indices]
        else:
             dense_scan_small = dense_scan

        # 2. Estimate Motion (ICP)
        d_pose = np.identity(3)
        moved = False
        
        if self.prev_dense_scan is not None:
             try:
                 # Align current dense scan to previous dense scan
                 T, distances = icp(dense_scan_small, self.prev_dense_scan)
                 d_pose = T
                 
                 # Check Motion Thresholds (Keyframe Logic)
                 # prevents drift when static
                 dx = d_pose[0, 2]
                 dy = d_pose[1, 2]
                 dtheta = math.atan2(d_pose[1, 0], d_pose[0, 0])
                 
                 dist_moved = math.sqrt(dx*dx + dy*dy)
                 rot_moved = abs(math.degrees(dtheta))
                 
                 if dist_moved > 20 or rot_moved > 2.0: # 20mm or 2 degrees
                      moved = True
                 else:
                      # If motion is tiny, ignore it to prevent drift?
                      # Or accumulate it? safer to ignore for "rock solid" static map
                      d_pose = np.identity(3) # Force zero
                      # But we might still want to add points to map if new area seen?
                      # No, if we didn't move, we see same thing.
                      pass

             except Exception as e:
                 print(f"ICP Error: {e}")
                 pass
        else:
             moved = True # First frame always updates
        
        # Extract refined delta
        dx = d_pose[0, 2]
        dy = d_pose[1, 2]
        dtheta = math.atan2(d_pose[1, 0], d_pose[0, 0])
        
        # 3. Update Global Pose
        gx = dx * math.cos(self.pose[2]) - dy * math.sin(self.pose[2])
        gy = dx * math.sin(self.pose[2]) + dy * math.cos(self.pose[2])
        
        self.pose[0] += gx
        self.pose[1] += gy
        self.pose[2] += dtheta
        
        # 4. Update Map & Reference
        # If we moved (or it's the first frame), update the map
        if moved or self.prev_dense_scan is None:
             c, s = math.cos(self.pose[2]), math.sin(self.pose[2])
             R_global = np.array([[c, -s], [s, c]])
             
             # Transform dense scan to global
             global_points = np.dot(dense_scan, R_global.T) + self.pose[:2]
             
             # PROBABILISTIC UPDATE (RAYCASTING)
             # Update cells along the ray to be FREE
             # Update end point to be OCCUPIED
             
             # Convert robot pose to grid coords
             rob_ix = int(round(self.pose[0] / self.GRID_RESOLUTION))
             rob_iy = int(round(self.pose[1] / self.GRID_RESOLUTION))
             
             for p in global_points:
                 gx, gy = p[0], p[1]
                 ix = int(round(gx / self.GRID_RESOLUTION))
                 iy = int(round(gy / self.GRID_RESOLUTION))
                 
                 # Raycast from robot to hit point
                 # 'bresenham_line' returns integers
                 cells = bresenham_line(rob_ix, rob_iy, ix, iy)
                 
                 # Mark free space (all except last one)
                 for i in range(len(cells) - 1):
                     cx, cy = cells[i]
                     key = (cx, cy)
                     if key in self.global_grid:
                         # DECREASE occupancy
                         # We use a float for probability-like behavior:
                         # 0.0 = Free, >1.0 = Occupied
                         self.global_grid[key][2] = max(0.1, self.global_grid[key][2] - 0.5) 
                     else:
                         # Register as Free Space (Seen, but empty)
                         # We store [cx, cy, count]
                         # Using 0.01 to indicate "we saw it, but it's empty"
                         # We need the real coordinates for the key roughly
                         self.global_grid[key] = [cx * self.GRID_RESOLUTION, cy * self.GRID_RESOLUTION, 0.1]
                 
                 # Mark occupied (last point)
                 key = (ix, iy)
                 if key in self.global_grid:
                     cell = self.global_grid[key]
                     # Fuse
                     if cell[2] < 50:
                         cell[0] += gx
                         cell[1] += gy
                         cell[2] += 1.0 # Add 1 for hit
                 else:
                     self.global_grid[key] = [gx, gy, 1.0]
                          
             self.prev_dense_scan = dense_scan_small # Update reference
             self.path.append(self.pose.copy())
             
    def get_map_points(self):
        """Reconstruct list of points [x,y] from the grid (Mean values)"""
        points = []
        for key, val in self.global_grid.items():
            # if val[2] > 1: # Optional filter: only confirmed points?
            points.append([val[0]/val[2], val[1]/val[2]])
        return points

    def get_pose(self):
        """Returns (x, y, theta_degrees)"""
        return self.pose[0], self.pose[1], math.degrees(self.pose[2])

    # GET TURN ANGLE
    def calculate_exploration_vector(self, cartesian_points):
        """
        Returns the angle (degrees) of the 'deepest' sector (most open space).
        Input: numpy array or list of [x, y] points (in robot local frame)
        """
        if len(cartesian_points) == 0:
            return None
            
        # Analyze in 10-degree sectors
        SECTOR_SIZE = 10
        num_sectors = 360 // SECTOR_SIZE
        sector_sums = [0.0] * num_sectors
        sector_counts = [0] * num_sectors
        
        # Convert Cart to Polar
        for p in cartesian_points:
            x, y = p[0], p[1]
            dist = math.sqrt(x*x + y*y)
            if dist > 0:
                angle_rad = math.atan2(y, x)
                angle_deg = math.degrees(angle_rad) % 360
                
                idx = int(angle_deg // SECTOR_SIZE)
                
                if 0 <= idx < num_sectors:
                    sector_sums[idx] += dist
                    sector_counts[idx] += 1
                    
        best_idx = -1
        max_avg = -1.0
        
        for i in range(num_sectors):
            if sector_counts[i] > 5: # Threshold
                avg = sector_sums[i] / sector_counts[i]
                if avg > max_avg:
                    max_avg = avg
                    best_idx = i
                    
        if best_idx != -1:
            # Return center angle
            # We want -180 to 180 range usually? 
            # Logic above uses 0-360.
            angle = best_idx * SECTOR_SIZE + (SECTOR_SIZE / 2)
            if angle > 180: angle -= 360
            return angle
        return None

    def get_turn_command(self):
        """
        Calculates the best direction to move into open space.
        Returns:
            angle_deg (float): The relative angle to turn (negative=right, positive=left).
                               Returns None if no map data.
        """
        map_pts = self.get_map_points()
        if len(map_pts) < 10:
            return None
            
        recent_map = np.array(map_pts)
        
        # Transform Global -> Local
        rob_x, rob_y, rob_theta = self.pose
        c, s = math.cos(rob_theta), math.sin(rob_theta)
        R_inv = np.array([[c, s], [-s, c]])
        
        pts_centered = recent_map - np.array([rob_x, rob_y])
        local_map_pts = np.dot(pts_centered, R_inv.T)
        
        target_angle_local = self.calculate_exploration_vector(local_map_pts)
        return target_angle_local

def normalize_angle(angle_deg):
    """Normalize angle to [-180, 180]"""
    while angle_deg > 180:
        angle_deg -= 360
    while angle_deg <= -180:
        angle_deg += 360
    return angle_deg



slam_system = SimpleSLAM()

# 46 degrees
# 133 degrees

# --- Network Lidar Wrapper ---
class NetworkRPLidar:
    def __init__(self, port=12345):
        self.port = port
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.sock.bind(('0.0.0.0', self.port))
        self.sock.listen(1)
        self.conn = None
        self.addr = None
        print(f"Network Lidar: Listening on 0.0.0.0:{self.port}...")
        print("Waiting for connection from Raspberry Pi...")
        self.conn, self.addr = self.sock.accept()
        print(f"Network Lidar: Connected by {self.addr}")

    def iter_measurements(self, max_buf_meas=3000):
        # We ignore max_buf_meas as we just stream what we get
        struct_fmt = '<Bff' # uchar, float, float
        struct_len = struct.calcsize(struct_fmt)
        
        while True:
            try:
                # Robustly read 'struct_len' bytes
                data = b''
                while len(data) < struct_len:
                    packet = self.conn.recv(struct_len - len(data))
                    if not packet:
                        return # Connection closed
                    data += packet
                    
                quality, angle, distance = struct.unpack(struct_fmt, data)
                # Yield tuple compatible with RPLidar library: (new_scan, quality, angle, distance)
                # We don't track new_scan bit in this simple protocol, passing False (0) is fine for our SLAM logic
                yield (0, quality, angle, distance) 
            except Exception as e:
                print(f"Network error: {e}")
                break
                
    def stop(self):
        if self.conn:
            self.conn.close()
        self.sock.close()
        
    def stop_motor(self):
        pass
        
    def disconnect(self):
        self.stop()
        
    def clear_input(self):
        pass
        
    def get_health(self):
        return "Network Mode"
        
    def get_info(self):
        return {"model": "Network", "serial": "Remote"}

# --- Lidar Sensor Class ---
class LidarSensor:
    def __init__(self, port=None, network_mode=False, network_port=12345):
        self.network_mode = network_mode
        
        if self.network_mode:
            print(f"Initializing Network Lidar on port {network_port}...")
            self.lidar = NetworkRPLidar(port=network_port)
        else:
            if port is None:
                port = self._detect_serial_port()
            
            # Note: self.lidar is initialized inside _detect_serial_port if detection succeeds,
            # otherwise we might need to initialize it here if specific port passed.
            if not hasattr(self, 'lidar'):
                 print(f"Attempting to connect to specific/fallback port: {port}")
                 self.lidar = RPLidar(port, timeout=5)
        
        # Robust initialization
        try:
            self.lidar.clear_input()
            time.sleep(0.5)
            print(self.lidar.get_info())
            print(self.lidar.get_health())
        except Exception as e:
            print(f"Initial connection error: {e}. Retrying reset...")
            try:
                self.lidar.stop()
                self.lidar.stop_motor()
                self.lidar.disconnect()
                time.sleep(1)
                self.lidar.connect()
                self.lidar.clear_input()
                print(self.lidar.get_info())
                print(self.lidar.get_health())
            except Exception as e2:
                 print(f"Retry failed: {e2}. Continuing anyway, scanning might fail.")
        self.scan_data = []               # Cumulative list of all scan points
        self.last_revolution_data = []    # Points from the last complete revolution
        self.port = None # Store the connected port
        self.current_revolution_points = []  # Points for the current revolution in progress
        self.running = True
        self.revolutions = 0
        self.last_angle = None            # To detect wrap-around in angle
        # Start the LiDAR scanning in its own thread
        self.thread = threading.Thread(target=self._scan_loop, daemon=True)
        self.thread.start()

    def _detect_serial_port(self):
        """
        Detects the serial port by trying to connect and read info from valid candidates.
        """
        ports = list(serial.tools.list_ports.comports())
        candidates = []
        
        # Filter for likely candidates on macOS/Linux
        for p in ports:
            if any(x in p.device for x in ["SLAB_USBtoUART", "usbserial", "USB", "ACM"]):
                candidates.append(p.device)
        
        # Prioritize SLAB_USBtoUART as it was confirmed working
        candidates.sort(key=lambda x: 0 if "SLAB_USBtoUART" in x else 1)
                
        print(f"Candidate ports: {candidates}")
        
        for port in candidates:
            print(f"Testing port: {port}...")
            for baud in [115200, 256000]:
                print(f"  Trying baudrate: {baud}...")
                try:
                    # Try to connect and get info (reduced timeout for faster checks)
                    temp_lidar = RPLidar(port, baudrate=baud, timeout=2)
                    # Force stop in case it's already spinning
                    try:
                        temp_lidar.stop()
                        temp_lidar.stop_motor()
                    except:
                        pass
                    temp_lidar.clear_input()
                    info = temp_lidar.get_info()
                    print(f"Success! Found LiDAR on {port} at {baud} baud: {info}")
                    temp_lidar.disconnect()
                    # Re-initialize the main lidar object with the correct baudrate
                    self.lidar = RPLidar(port, baudrate=baud, timeout=5)
                    self.port = port
                    return port
                except Exception as e:
                    print(f"  Failed on {port} @ {baud}: {e}")
                    if "Resource busy" in str(e) or "Errno 16" in str(e):
                        print(f"  --> PORT LOCKED: {port} is being used by another process or is stuck.")
                        print("      ACTION REQUIRED: Unplug and replug the LiDAR USB cable to reset the connection.")
                    try:
                        temp_lidar.disconnect()
                    except:
                        pass
                    continue
                
        print("Could not find a working LiDAR on any port. Defaulting to /dev/ttyUSB0")
        return '/dev/ttyUSB0'
    

    def _scan_loop(self):
        """
        Continuously fetch scan data in a separate thread.
        """
        print("scan looping", flush=True)
        # Ensure clean state before starting scans
        try:
             self.lidar.stop()
             self.lidar.stop_motor()
             time.sleep(1)
             self.lidar.start_motor()
             time.sleep(0.5)
        except:
             pass
             
        while self.running:
            try:
                # Track stats
                total_points_in_rev = 0
                valid_points_in_rev = 0
                rev_start_time = time.time()
                
                
                # Use iter_measurments for raw data access (avoid buffering/grouping latency)
                # Increasing buffer here to prevent overflow on Pi
                for new_scan, quality, angle, distance in self.lidar.iter_measurments(max_buf_meas=3000):
                    if not self.running:
                        break
                    
                    total_points_in_rev += 1
                    
                    # 'new_scan' is a boolean flag indicating start of a new revolution
                    if new_scan:
                        # Revolution complete
                        dt = time.time() - rev_start_time
                        rpm = 60/dt if dt > 0 else 0
                        rev_start_time = time.time()
                        
                        count = valid_points_in_rev
                        print(f"Rev {self.revolutions} | RPM: {rpm:.1f} | Points: {count}/{total_points_in_rev} ({(count/total_points_in_rev*100) if total_points_in_rev else 0:.1f}%)", flush=True)
                            
                        self.last_revolution_data = self.current_revolution_points.copy()
                        
                        self.current_revolution_points = []
                        valid_points_in_rev = 0
                        total_points_in_rev = 0
                        self.revolutions += 1
                    
                    if distance > 0:
                        valid_points_in_rev += 1
                        # Create ScanObject (adapted for new format)
                        # We use a dummy list [quality, angle, distance] to match ScanObject init
                        scan_obj = ScanObject([quality, angle, distance], revolution=self.revolutions)
                        self.scan_data.append(scan_obj)
                        self.current_revolution_points.append(scan_obj)
            except RPLidarException as e:
                print(f"LiDAR connection/scan error: {e}. Retrying...", flush=True)
                try:
                    self.lidar.stop()
                    self.lidar.disconnect()
                    self.lidar.connect()
                    self.lidar.clear_input()
                except Exception as recon_e:
                    print(f"Failed to reconnect: {recon_e}")
                    time.sleep(2)
            except Exception as e:
                print(f"Unexpected error in scan loop: {e}", flush=True)
                time.sleep(1)
    
    def get_scan(self):
        """Return the full list of scan objects."""
        return self.scan_data

    def getLastRevolutionData(self):
        return self.last_revolution_data
    
    def waitForNewRevolution(self):
        currentRevolution = self.revolutions
        
        while True:
            time.sleep(0.01)
            if currentRevolution != self.revolutions:
                return
        
    
    def stop(self):
        """Stop scanning and clean up the LiDAR sensor."""
        self.running = False
        try:
            self.thread.join()
        except:
            pass
        self.lidar.stop()
        self.lidar.stop_motor()
        self.lidar.disconnect()

class ScanObject:
    def __init__(self, scanBase, revolution=None):
        self.scanStrength = scanBase[0]
        self.angle = scanBase[1]
        self.distance = scanBase[2] # Keep in mm
        self.revolution = revolution

        self.angle = scanBase[1]
        self.distance = scanBase[2] # Keep in mm
        self.revolution = revolution


        

def export_map_to_image(slam, filename="slam_map.png"):
    """
    Exports the current global grid to a PNG image.
    Auto-crops to the explored area.
    """
    if not slam.global_grid:
        print("Map is empty, nothing to save.")
        return False
        
    # 1. Calculate Bounds
    # Keys are (ix, iy)
    min_x = min_y = float('inf')
    max_x = max_y = float('-inf')
    
    for (ix, iy) in slam.global_grid.keys():
        if ix < min_x: min_x = ix
        if ix > max_x: max_x = ix
        if iy < min_y: min_y = iy
        if iy > max_y: max_y = iy
        
    # Add padding (1 meter = 1000mm / 30mm res ~= 33 cells)
    padding = 10
    min_x -= padding
    max_x += padding
    min_y -= padding
    max_y += padding
    
    width_cells = max_x - min_x + 1
    height_cells = max_y - min_y + 1
    
    # Scale for output image (1 pixel per cell is too small, let's do 4x)
    px_scale = 4
    img_w = width_cells * px_scale
    img_h = height_cells * px_scale
    
    try:
        surf = pygame.Surface((img_w, img_h))
        # Fill Unknown (Black) - Matches walls, hides ray gaps
        surf.fill((0, 0, 0))
        
        # 2. Process Map (Morphological Closing)
        # Create dense array
        dense_map = np.zeros((width_cells, height_cells), dtype=bool)
        
        # Fill dense map with occupied cells
        for (ix, iy), val in slam.global_grid.items():
            weight = val[2]
            if weight >= 1.0:
                # Array indices [x, y] relative to min
                ax = ix - min_x
                ay = iy - min_y
                if 0 <= ax < width_cells and 0 <= ay < height_cells:
                    dense_map[ax, ay] = True
                    
        # Apply Binary Closing (Dilation then Erosion) to bridge gaps
        structure = np.ones((5, 5), dtype=int)
        closed_map = binary_closing(dense_map, structure=structure, iterations=2)
        
        # CONSTRAINT: Do not expand walls into known Free Space
        # Build dense free map
        dense_free = np.zeros((width_cells, height_cells), dtype=bool)
        for (ix, iy), val in slam.global_grid.items():
            if val[2] < 1.0: # Free
                ax = ix - min_x
                ay = iy - min_y
                if 0 <= ax < width_cells and 0 <= ay < height_cells:
                    dense_free[ax, ay] = True
                    
        # Remove pixels from closed_map that are actually known free space
        # This prevents "closing" a doorway or corridor that we have seen through
        closed_map = np.logical_and(closed_map, np.logical_not(dense_free))
        
        # 3. Draw Results
        
        # First Draw Free Space (White) - Unprocessed
        # We perform this first so walls overwrite it
        for (ix, iy), val in slam.global_grid.items():
            weight = val[2]
            if weight < 1.0:
                 sx = (ix - min_x) * px_scale
                 sy = (iy - min_y) * px_scale
                 pygame.draw.rect(surf, (255, 255, 255), (sx, sy, px_scale, px_scale))
                 
        # Draw Occupied Space (Black) - From Morphological Result
        # Iterate over the boolean array
        # This is fast enough for export
        it = np.nditer(closed_map, flags=['multi_index'])
        for is_occ in it:
            if is_occ:
                ax, ay = it.multi_index
                sx = ax * px_scale
                sy = ay * px_scale
                pygame.draw.rect(surf, (0, 0, 0), (sx, sy, px_scale, px_scale))
            
        # 4. Draw Path (Optional)
        if len(slam.path) > 1:
            scaled_path = []
            for px, py, _ in slam.path:
                 ix = int(px / slam.GRID_RESOLUTION)
                 iy = int(py / slam.GRID_RESOLUTION)
                 sx = (ix - min_x) * px_scale
                 sy = (iy - min_y) * px_scale
                 scaled_path.append((sx, sy))
            
            if len(scaled_path) > 1:
                pygame.draw.lines(surf, (0, 255, 0), False, scaled_path, 2)

        pygame.image.save(surf, filename)
        print(f"Map saved to {filename}")
        return True
    except Exception as e:
        print(f"Error saving map: {e}")
        return False

if __name__ == "__main__":
    # Parse Arguments
    parser = argparse.ArgumentParser(description="LiDAR SLAM System")
    parser.add_argument("--headless", action="store_true", help="Run without a graphical window (for Pi/SSH)")
    parser.add_argument("--port", type=str, default=None, help="Specific serial port for LiDAR (e.g. /dev/ttyUSB0)")
    parser.add_argument("--network", action="store_true", help="Run in network mode (receive data from Pi)")
    parser.add_argument("--net_port", type=int, default=12345, help="Network port for streaming (default: 12345)")
    args = parser.parse_args()

    if args.headless:
        # Set dummy video driver for Pygame (no window)
        os.environ["SDL_VIDEODRIVER"] = "dummy"
        print("Running in HEADLESS mode. No window will be shown.")

    # Initialize LiDAR
    try:
        lidar = LidarSensor(port=args.port, network_mode=args.network, network_port=args.net_port)
    except Exception as e:
        print(f"Failed to initialize LiDAR: {e}")
        sys.exit(1)
    time.sleep(2) # Allow time for spin up and connection
    
    # Try to slow down motor to increase point density (more samples per degree)
    # Default is often 1023 (max). 600-700 is a good target for A1/A2.
    # DISABLED FOR PI STABILITY (Buffer Overflows)
    # try:
    #     lidar.lidar.set_pwm(200)
    #     print("Set PWM to 660 for denser scanning")
    # except Exception as e:
    #     print(f"Could not set PWM (might not be supported): {e}")
    
    
    # Initialize Pygame
    pygame.init()
    WIDTH, HEIGHT = 1000, 800
    screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
    pygame.display.set_caption("LiDAR Radar View")
    
    # Fonts
    font_hud = pygame.font.SysFont("monospace", 16)
    font_label = pygame.font.SysFont("monospace", 12)
    
    # Console Input Listener (Thread)
    def console_listener(slam_sys):
        """
        Listens for keyboard input from the console (SSH friendly).
        """
        print("\n--- CONSOLE CONTROLS ---")
        print(" 's' + Enter: Save map to png")
        print(" 'q' + Enter: Quit")
        print("------------------------\n")
        
        while True:
            try:
                # Simple blocking input
                cmd = input().strip().lower()
                
                print(f"DEBUG: Received command '{cmd}'") # Debug print
                    
                if cmd == 's':
                    # Save Map
                    export_dir = "map_exports"
                    if not os.path.exists(export_dir):
                        os.makedirs(export_dir)
                    timestamp = time.strftime("%Y%m%d-%H%M%S")
                    filename = os.path.join(export_dir, f"slam_map_{timestamp}.png")
                    export_map_to_image(slam_sys, filename)
                elif cmd == 'q':
                    print("Quitting from console...")
                    # Post QUIT event to main loop
                    pygame.event.post(pygame.event.Event(pygame.QUIT))
                    break
            except EOFError:
                break
            except Exception as e:
                # print(f"Console error: {e}")
                pass
    
    # Start input thread
    input_thread = threading.Thread(target=console_listener, args=(slam_system,), daemon=True)
    input_thread.start()

    # --- Control Server Thread (for bi-directional CLI from Pi) ---
    def control_server_thread(slam_sys, port=12346):
        print(f"Control Server: Listening on port {port}")
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            server_sock.bind(('0.0.0.0', port))
            server_sock.listen(1)
            
            while True:
                conn, addr = server_sock.accept()
                # print(f"Control: Connected by {addr}")
                try:
                     while True:
                        data = conn.recv(1024)
                        if not data: break
                        
                        msg = data.decode('utf-8').strip()
                        response = "UNKNOWN"
                        
                        if msg == "GET_TURN":
                             # Calculate turn command
                             cmd = slam_sys.get_turn_command()
                             if cmd is None:
                                 response = "TURN None"
                             else:
                                 response = f"TURN {cmd:.2f}"
                        else:
                             response = f"ECHO {msg}"
                             
                        conn.sendall(response.encode('utf-8'))
                except Exception as e:
                    print(f"Control Error: {e}")
                finally:
                    conn.close()
        except Exception as e:
            print(f"Failed to bind control server: {e}")

    if args.network:
        # Start control server
        ctrl_thread = threading.Thread(target=control_server_thread, args=(slam_system,), daemon=True)
        ctrl_thread.start()

    # Colors (Sci-Fi Theme)
    # Colors (ROS / Occupancy Grid Theme)
    COLOR_BG = (120, 120, 120)   # Grey (Unknown)
    COLOR_FREE = (255, 255, 255) # White (Free)
    COLOR_OCCUPIED = (0, 0, 0)   # Black (Occupied)
    
    COLOR_GRID = (100, 100, 100) # Slightly darker grey
    COLOR_AXIS = (255, 0, 0)     # Red Axis
    COLOR_ROBOT = (0, 255, 0)    # Green Robot
    COLOR_TEXT = (0, 0, 0)       # Black Text

    # Visualization settings
    scale = 0.15  # Pixels per mm (0.15 = 150px per meter. Fits ~5m room on screen)
    offset_x = WIDTH // 2
    offset_y = HEIGHT // 2
    view_offset_x = 0
    view_offset_y = 0
    is_dragging = False
    last_mouse_pos = (0, 0)
    
    clock = pygame.time.Clock()
    running = True

    
    # Visualization buffer
    # We use point_buffer for raw "trails" (local frame)
    point_buffer = deque(maxlen=2000) 
    last_processed_revolution = -1
    
    # SLAM Mode Flag
    SHOW_SLAM_MAP = True
    
    # Stats for HUD
    current_rpm = 0.0
    current_valid_pct = 0.0
    
    # Exploration Vector Smoothing (EMA)
    exp_smooth_x = 0.0
    exp_smooth_y = 0.0
    
    # Save Notification
    last_save_time = 0
    save_msg = ""

    print("Radar UI started. Drag to Pan, Scroll to Zoom.")

    try:
        while running:
            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    WIDTH, HEIGHT = event.w, event.h
                    offset_x = WIDTH // 2
                    offset_y = HEIGHT // 2
                elif event.type == pygame.MOUSEWHEEL:
                    # Zoom in/out
                    zoom_factor = 1.1 if event.y > 0 else 0.9
                    scale *= zoom_factor
                    scale = max(0.01, min(scale, 10.0)) # Relaxed clamp to allow infinite zoom
                
                # Pan logic
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 1: # Left click
                        is_dragging = True
                        last_mouse_pos = event.pos
                elif event.type == pygame.MOUSEBUTTONUP:
                    if event.button == 1:
                        is_dragging = False
                elif event.type == pygame.MOUSEMOTION:
                    if is_dragging:
                        dx, dy = event.pos[0] - last_mouse_pos[0], event.pos[1] - last_mouse_pos[1]
                        view_offset_x += dx
                        view_offset_y += dy
                        last_mouse_pos = event.pos
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_s:
                        # Save Map
                        export_dir = "map_exports"
                        if not os.path.exists(export_dir):
                            os.makedirs(export_dir)
                            
                        timestamp = time.strftime("%Y%m%d-%H%M%S")
                        filename = os.path.join(export_dir, f"slam_map_{timestamp}.png")
                        
                        if export_map_to_image(slam_system, filename):
                             # Show feedback
                             last_save_time = time.time()
                             save_msg = f"Saved to {export_dir}/"
            
            # --- Rendering ---
            # center_x/y is now the WORLD ORIGIN (0,0) on screen
            map_origin_x = offset_x + view_offset_x
            map_origin_y = offset_y + view_offset_y

            if not args.headless:
                screen.fill(COLOR_BG)
    
                # 1. Draw Grid (Centered on World Origin)
                # Calculate grid step size
                steps = [500, 1000, 2000, 5000, 10000] # 0.5m, 1m, 2m, 5m, 10m
                
                # Find closest nice step
                visible_radius_mm = min(WIDTH, HEIGHT) / 2 / scale
                grid_step_mm = 1000
                for s in steps:
                    if visible_radius_mm / s >= 2.5: 
                       grid_step_mm = s
                
                # Draw circles centered on World Origin
                current_r_mm = grid_step_mm
                while True:
                    r_px = int(current_r_mm * scale)
                    if r_px > max(WIDTH, HEIGHT) * 2: # Stop if far off screen
                        break
                        
                    # Only draw if roughly visible? Pygame handles clipping.
                    pygame.draw.circle(screen, COLOR_GRID, (int(map_origin_x), int(map_origin_y)), r_px, 1)
                    
                    # Label along the axis
                    dist_m = current_r_mm / 1000
                    label_text = f"{dist_m:.1f}m" if dist_m % 1 != 0 else f"{int(dist_m)}m"
                    label = font_label.render(label_text, True, COLOR_GRID)
                    screen.blit(label, (map_origin_x + 5, map_origin_y - r_px - 15))
                    
                    current_r_mm += grid_step_mm
                
                # Draw Axes (World Frame)
                pygame.draw.line(screen, COLOR_AXIS, (map_origin_x - 10000, map_origin_y), (map_origin_x + 10000, map_origin_y), 1)
                pygame.draw.line(screen, COLOR_AXIS, (map_origin_x, map_origin_y - 10000), (map_origin_x, map_origin_y + 10000), 1)

            # 2. Process Data
            data_points = lidar.getLastRevolutionData()
            
            if data_points and data_points[0].revolution > last_processed_revolution:
                last_processed_revolution = data_points[0].revolution
                point_buffer.extend(data_points)
                # Update SLAM
                slam_system.process_scan(data_points)
            
            # Get Robot Pose
            rob_x, rob_y, rob_theta = slam_system.pose[0], slam_system.pose[1], slam_system.pose[2]
            
            # Robot Screen Position
            rob_sx = map_origin_x + rob_x * scale
            rob_sy = map_origin_y + rob_y * scale

            # 3. Draw Points with Fading (Raw Buffer)
            # These are relative to the robot. In Absolute view, we must transform them.
            # We assume they belong to the *current* pose for visualization (approx).
            if point_buffer:
                num_points = len(point_buffer)
                for i, point in enumerate(point_buffer):
                    dist = point.distance
                    if dist > 0:
                        # Local Angle
                        angle_rad_local = math.radians(point.angle)
                        # Global Angle = Robot Theta + Local Angle
                        angle_rad_global = rob_theta + angle_rad_local
                        
                        # Global Pos = Robot Pos + Rotated Vector
                        gx = rob_x + dist * math.cos(angle_rad_global)
                        gy = rob_y + dist * math.sin(angle_rad_global)
                        
                        # Screen Pos
                        sx = map_origin_x + gx * scale
                        sy = map_origin_y + gy * scale
                        
                        if -10 <= sx <= WIDTH+10 and -10 <= sy <= HEIGHT+10:
                            intensity = int(50 + (205 * (i / num_points))) 
                            color = (0, intensity, intensity) 
                            pygame.draw.circle(screen, color, (int(sx), int(sy)), 2)

            # 3.5. Draw SLAM Global Map (Occupancy Grid Style)
            if SHOW_SLAM_MAP:
                # Iterate only over visible cells ideally, but python dict iteration is fast enough for <50k cells
                # To look like a filled map, we draw rectangles (pixels), not circles
                
                # Pre-calculate screen center to avoid doing it in loop
                # We need to lock the dictionary or copy it to avoid runtime error if thread updates it
                # For now, we assume simple access is fine
                
                # Optimized Rendering: Draw "Free" first, then "Occupied"
                # Actually, iterating once is better.
                
                rects_free = []
                rects_occupied = []
                
                # 1. Collect Valid Cells
                # We collect dict keys (ix, iy) -> type (0=Free, 1=Occupied)
                # To optimize, we separate them into lists
                
                # Optimized collection (culling off-screen)
                grid_res = slam_system.GRID_RESOLUTION
                cols_free = {} # y -> list of x
                cols_occ = {}  # y -> list of x
                
                # Screen bounds in grid coords
                # map_origin + x * scale = 0  =>  x = -map_origin / scale
                min_vis_x = int((-map_origin_x / scale) / grid_res) - 2
                max_vis_x = int(((WIDTH - map_origin_x) / scale) / grid_res) + 2
                min_vis_y = int((-map_origin_y / scale) / grid_res) - 2
                max_vis_y = int(((HEIGHT - map_origin_y) / scale) / grid_res) + 2
                
                for key, val in slam_system.global_grid.items():
                    gx, gy = key[0], key[1]
                    
                    # Culling
                    if not (min_vis_x <= gx <= max_vis_x and min_vis_y <= gy <= max_vis_y):
                        continue
                        
                    weight = val[2]
                    target_dict = cols_free if weight < 1.0 else cols_occ
                    
                    if gy not in target_dict:
                        target_dict[gy] = []
                    target_dict[gy].append(gx)
                
                # 2. Greedy Meshing (Horizontal Merging) & Drawing
                grid_size_px = max(1, int(grid_res * scale))
                # Slight overlap to prevent gaps? No, standard Size is best.
                # Adding 1px sometimes helps with float rounding gaps, but depends on scale.
                
                def draw_merged_type(col_dict, color):
                    for y, xs in col_dict.items():
                        xs.sort()
                        if not xs: continue
                        
                        # Greedy merge
                        start_x = xs[0]
                        curr_x = xs[0]
                        
                        for x in xs[1:]:
                            if x == curr_x + 1:
                                # Contiguous
                                curr_x = x
                            else:
                                # Break
                                # Draw rect from start_x to curr_x (inclusive)
                                pixels_x = int(map_origin_x + start_x * grid_res * scale)
                                pixels_y = int(map_origin_y + y * grid_res * scale)
                                width = int((curr_x - start_x + 1) * grid_res * scale)
                                # Ensure minimal width due to float trunc
                                if width < grid_size_px: width = grid_size_px
                                
                                # Sometimes width needs +1 to avoid hairline gaps
                                pygame.draw.rect(screen, color, (pixels_x, pixels_y, width + 1, grid_size_px + 1))
                                
                                start_x = x
                                curr_x = x
                        
                        # Draw final segment
                        pixels_x = int(map_origin_x + start_x * grid_res * scale)
                        pixels_y = int(map_origin_y + y * grid_res * scale)
                        width = int((curr_x - start_x + 1) * grid_res * scale)
                        pygame.draw.rect(screen, color, (pixels_x, pixels_y, width + 1, grid_size_px + 1))

                draw_merged_type(cols_free, COLOR_FREE)
                draw_merged_type(cols_occ, COLOR_OCCUPIED)

                 # Draw Path
                         
                # Draw Path
                if len(slam_system.path) > 1:
                    path_points = []
                    for pose in slam_system.path:
                        px = map_origin_x + pose[0] * scale
                        py = map_origin_y + pose[1] * scale
                        path_points.append((px, py))
                    
                    if len(path_points) > 1:
                        pygame.draw.lines(screen, (0, 255, 0), False, path_points, 2)


            # 4. Draw Robot Marker (At calculated screen pos)
            pygame.draw.circle(screen, COLOR_ROBOT, (int(rob_sx), int(rob_sy)), 6)
            # Direction indicator
            arrow_len = 20
            ax = rob_sx + arrow_len * math.cos(rob_theta)
            ay = rob_sy + arrow_len * math.sin(rob_theta)
            pygame.draw.line(screen, COLOR_ROBOT, (int(rob_sx), int(rob_sy)), (int(ax), int(ay)), 2)

            # 5. Draw Exploration Vector (Yellow Line)
            # Use SLAM MAP for stability
            
            # Simple API call:
            target_angle_local = slam_system.get_turn_command()
            
            # NAVIGATION COMMAND
            turn_cmd_deg = 0.0

            if target_angle_local is not None:
                    # Smoothing (Local Frame)
                    t_rad_local = math.radians(target_angle_local)
                    tx, ty = math.cos(t_rad_local), math.sin(t_rad_local)
                    
                    alpha = 0.1 
                    if exp_smooth_x == 0 and exp_smooth_y == 0:
                        exp_smooth_x, exp_smooth_y = tx, ty
                    else:
                        exp_smooth_x = (1 - alpha) * exp_smooth_x + alpha * tx
                        exp_smooth_y = (1 - alpha) * exp_smooth_y + alpha * ty
                    
                    # Convert Smoothed Vector to Global Angle for drawing
                    local_angle_smooth_rad = math.atan2(exp_smooth_y, exp_smooth_x)
                    local_angle_smooth_deg = math.degrees(local_angle_smooth_rad)
                    
                    # Calculate Turn Command (Normalize to -180 to 180)
                    # Exploration Vector is already in Robot Frame: 0 deg = Forward (X)
                    # We need to turn to face it.
                    turn_cmd_deg = normalize_angle(local_angle_smooth_deg)
                    
                    global_draw_angle = rob_theta + local_angle_smooth_rad
                    
                    # Draw Line from Robot
                    vec_len = 150 
                    ex = rob_sx + vec_len * math.cos(global_draw_angle)
                    ey = rob_sy + vec_len * math.sin(global_draw_angle)
                    
                    pygame.draw.line(screen, (255, 255, 0), (rob_sx, rob_sy), (ex, ey), 4)
                    
                    label_target = font_label.render(f"GOAL ({turn_cmd_deg:.1f}°)", True, (255, 255, 0))
                    screen.blit(label_target, (ex + 10, ey))
                


            # 6. Draw HUD
            hud_y = 10
            line_height = 20
            
            # Format Turn Command
            turn_str = "N/A"
            if 'turn_cmd_deg' in locals() and target_angle_local is not None:
                d = turn_cmd_deg
                if abs(d) < 5:
                    turn_str = "FORWARD"
                elif d > 0:
                    turn_str = f"LEFT {abs(d):.1f}°" # Standard: +Angle is Left (CCW)
                else: 
                    turn_str = f"RIGHT {abs(d):.1f}°"
            
            texts = [
                f"FPS: {clock.get_fps():.1f}",
                f"Buffer: {len(point_buffer)} pts",
                f"Scale: {scale:.2f} px/mm",
                f"Status: Connected ({lidar.port})",
                f"CMD: {turn_str}",
                "Controls: Drag=Pan, Scroll=Zoom"
            ]
            
            # Draw semi-transparent background for HUD
            # Calculate required size
            max_w = 0
            total_h = len(texts) * line_height + 10
            for t in texts:
                w, h = font_hud.size(t)
                max_w = max(max_w, w)
            
            hud_surf = pygame.Surface((max_w + 20, total_h), pygame.SRCALPHA)
            hud_surf.fill((0, 0, 0, 150)) # Black with alpha
            screen.blit(hud_surf, (5, 5))
            
            hud_y = 10
            for text_str in texts:
                s = font_hud.render(text_str, True, (255, 255, 255))
                screen.blit(s, (15, hud_y))
                hud_y += line_height
                
            # Draw Save Notification
            if time.time() - last_save_time < 3.0: # Show for 3 seconds
                 save_surf = font_hud.render(save_msg, True, (0, 255, 0))
                 screen.blit(save_surf, (WIDTH // 2 - save_surf.get_width() // 2, HEIGHT - 50))

            # Update display
            if not args.headless:
                pygame.display.flip()
            # If headless, we still need to tick clock to not consume 100% CPU
            clock.tick(30)
            
    except KeyboardInterrupt:
        pass
    finally:
        print("Stopping LiDAR...")
        lidar.stop()
        pygame.quit()