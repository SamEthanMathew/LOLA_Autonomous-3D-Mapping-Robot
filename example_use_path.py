#!/usr/bin/env python3
"""
Example script demonstrating how to load and use the path planner output.
This shows how to integrate the planned paths into a robot navigation system.
"""

import json
from pathlib import Path
import math


def load_path(json_path):
    """
    Load a path from the JSON output file.
    
    Args:
        json_path: Path to the JSON file
        
    Returns:
        Dictionary with path data
    """
    with open(json_path, 'r') as f:
        return json.load(f)


def calculate_heading(from_point, to_point):
    """
    Calculate heading angle from one waypoint to the next.
    
    Args:
        from_point: (x, y) tuple in mm
        to_point: (x, y) tuple in mm
        
    Returns:
        Heading angle in degrees (0° = East, 90° = North)
    """
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    angle_rad = math.atan2(dy, dx)
    angle_deg = math.degrees(angle_rad)
    return angle_deg


def calculate_distance(from_point, to_point):
    """
    Calculate distance between two waypoints.
    
    Args:
        from_point: (x, y) tuple in mm
        to_point: (x, y) tuple in mm
        
    Returns:
        Distance in millimeters
    """
    dx = to_point[0] - from_point[0]
    dy = to_point[1] - from_point[1]
    return math.sqrt(dx**2 + dy**2)


def generate_navigation_commands(path_data):
    """
    Generate robot navigation commands from a path.
    
    Args:
        path_data: Dictionary loaded from JSON
        
    Returns:
        List of navigation commands
    """
    waypoints_mm = path_data['waypoints_mm']
    commands = []
    
    for i in range(len(waypoints_mm) - 1):
        current = waypoints_mm[i]
        next_wp = waypoints_mm[i + 1]
        
        # Calculate segment properties
        distance = calculate_distance(current, next_wp)
        heading = calculate_heading(current, next_wp)
        
        command = {
            'waypoint_index': i,
            'from': current,
            'to': next_wp,
            'distance_mm': round(distance, 2),
            'distance_m': round(distance / 1000, 3),
            'heading_deg': round(heading, 2),
            'action': 'drive'
        }
        
        commands.append(command)
    
    # Add final command
    commands.append({
        'waypoint_index': len(waypoints_mm) - 1,
        'action': 'stop',
        'position': waypoints_mm[-1]
    })
    
    return commands


def main():
    """Example usage of the path planner output."""
    
    # Find the most recent path file
    output_dir = Path('path_outputs')
    json_files = list(output_dir.glob('path_coords_*.json'))
    
    if not json_files:
        print("No path files found in path_outputs/")
        print("Run path_planner.py first to generate a path.")
        return
    
    # Use most recent file
    latest_path = max(json_files, key=lambda p: p.stat().st_mtime)
    print(f"Loading path: {latest_path.name}\n")
    
    # Load path data
    path_data = load_path(latest_path)
    
    # Display path information
    print("=" * 70)
    print("PATH INFORMATION")
    print("=" * 70)
    print(f"Source Map:      {path_data['source_map']}")
    print(f"Generated:       {path_data['timestamp']}")
    print(f"Waypoints:       {path_data['num_waypoints']}")
    print(f"Total Length:    {path_data['path_length_m']} m")
    print(f"Planning Time:   {path_data.get('planning_time_seconds', 'N/A')} s")
    print()
    
    # Display waypoints
    print("WAYPOINTS (metric coordinates):")
    print("-" * 70)
    for i, (x, y) in enumerate(path_data['waypoints_mm']):
        if i == 0:
            label = "START"
        elif i == len(path_data['waypoints_mm']) - 1:
            label = "END"
        else:
            label = f"WP{i}"
        print(f"  {label:6s}  x={x:8.2f} mm, y={y:8.2f} mm  ({x/1000:.3f} m, {y/1000:.3f} m)")
    print()
    
    # Generate navigation commands
    commands = generate_navigation_commands(path_data)
    
    print("NAVIGATION COMMANDS:")
    print("-" * 70)
    for i, cmd in enumerate(commands):
        if cmd['action'] == 'drive':
            print(f"  Segment {i+1}:")
            print(f"    Distance:  {cmd['distance_m']:.3f} m")
            print(f"    Heading:   {cmd['heading_deg']:.1f}°")
        else:
            print(f"  Final: STOP at ({cmd['position'][0]:.1f}, {cmd['position'][1]:.1f}) mm")
    print()
    
    print("=" * 70)
    print("INTEGRATION EXAMPLE")
    print("=" * 70)
    print("""
# Example robot control integration:

import example_use_path

# Load path
path_data = example_use_path.load_path('path_outputs/your_path.json')
commands = example_use_path.generate_navigation_commands(path_data)

# Execute navigation
for cmd in commands:
    if cmd['action'] == 'drive':
        distance_m = cmd['distance_m']
        heading_deg = cmd['heading_deg']
        
        # Your robot control code here:
        # robot.turn_to(heading_deg)
        # robot.drive_forward(distance_m)
        # robot.wait_until_arrived()
        
        print(f"Driving {distance_m:.2f}m at heading {heading_deg:.1f}°")
    else:
        # robot.stop()
        print("Arrived at destination!")
""")


if __name__ == '__main__':
    main()



