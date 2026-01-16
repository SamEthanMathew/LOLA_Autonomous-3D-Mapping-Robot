import sys
import os
import time
import socket
import struct
import argparse
import threading
from rplidar import RPLidar

CONFIG_FILE = "last_ip.txt"

def get_saved_ip():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r') as f:
            return f.read().strip()
    return None

def save_ip(ip):
    with open(CONFIG_FILE, 'w') as f:
        f.write(ip)

def control_client_thread(host, port=12346):
    """
    Connects to the Control Server on the laptop.
    Listens for user Input on the Pi (Enter key) -> Sends GET_TURN -> Prints Response.
    """
    print(f"--- Control Client (Port {port}) ---")
    print("Press ENTER to query current Turn Command.")
    print("Type 'q' to quit.")
    
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sock.connect((host, port))
    except Exception as e:
        print(f"Control Client Connection Failed: {e}")
        return

    while True:
        try:
            cmd = input() # Blocking wait for Enter
            if cmd.strip().lower() == 'q':
                break
                
            # Send Request
            try:
                sock.sendall(b"GET_TURN")
                response = sock.recv(1024).decode('utf-8')
                print(f"Laptop Response: {response}")
            except Exception as e:
                print(f"Control Comms Error: {e}")
                break
                
        except EOFError:
            break
            
    sock.close()


def run_sender(host, port, device_port):
    # Save IP for next time
    save_ip(host)

    print(f"--- LiDAR TCP Sender ---")
    print(f"Target: {host}:{port}")
    print(f"Device: {device_port}")
    
    # 1. Start Control Thread (Interactive)
    # We do this before LiDAR to allow UI to be ready, but LiDAR init might print stuff.
    # Actually, we need to run LiDAR in main thread because it might need signal handling?
    # No, let's run LiDAR in main thread. Control thread handles stdin.
    
    ctrl_thread = threading.Thread(target=control_client_thread, args=(host, port + 1), daemon=True)
    ctrl_thread.start()
    
    # 2. Connect to LiDAR
    try:
        lidar = RPLidar(device_port, timeout=5)
        # info = lidar.get_info()
        # print(f"LiDAR Connected: {info}")
        # health = lidar.get_health()
        # print(f"Health: {health}")
        lidar.clear_input()
        time.sleep(1)
    except Exception as e:
        print(f"Failed to connect to LiDAR: {e}")
        return

    # 3. Connect to Server (Laptop) - Data Channel
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
    try:
        print(f"Connecting Data Stream to {host}:{port}...")
        sock.connect((host, port))
        print("Connected!")
    except Exception as e:
        print(f"Failed to connect data stream: {e}")
        lidar.stop()
        lidar.disconnect()
        return

    # 4. Stream Loop
    try:
        try:
            iterator = lidar.iter_measurments(max_buf_meas=3000)
        except AttributeError:
             iterator = lidar.iter_measurements(max_buf_meas=3000)
        count = 0
        last_print = time.time()
        
        print("Streaming data... (Press Enter for Turn Cmd)")
        
        for new_scan, quality, angle, distance in iterator:
            # Protocol V2: Sync(0xA5) + NewScan(1) + Quality(1) + Angle(4) + Distance(4) = 11 bytes
            pay_content = struct.pack('<BBff', int(new_scan), quality, angle, distance)
            payload = b'\xa5' + pay_content
            try:
                sock.sendall(payload)
                count += 1
                if count % 100 == 0: os.write(1, b'.') # Print dot for progress
            except BrokenPipeError:
                print("Data Connection lost.")
                break
                
            if time.time() - last_print > 5.0: # Print less frequently to not spam CLI
                # print(f"Sent {count} points (last 5s)")
                count = 0
                last_print = time.time()
                
    except KeyboardInterrupt:
        print("Stopping...")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Cleaning up...")
        lidar.stop()
        lidar.stop_motor()
        lidar.disconnect()
        sock.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stream LiDAR data over TCP")
    
    saved_ip = get_saved_ip()
    
    parser.add_argument("--host", type=str, default=saved_ip, help=f"IP address of laptop (default: {saved_ip})")
    parser.add_argument("--port", type=int, default=12345, help="Port to connect to (default: 12345)")
    parser.add_argument("--device", type=str, default='/dev/ttyUSB0', help="LiDAR serial port")
    
    args = parser.parse_args()
    
    if args.host is None:
        print("Error: No host IP provided and no saved IP found.")
        print("Usage: python3 lidar_sender.py --host <IP>")
        sys.exit(1)
    
    run_sender(args.host, args.port, args.device)
