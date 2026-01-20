import airsim
import time
import csv
import math
import threading
import os
import queue
import random

client = airsim.MultirotorClient()
client.confirmConnection()
client.enableApiControl(True)
client.armDisarm(True)

filename = "astroid_flight_dataset.csv"
print(f"Data will be saved to: {filename}")

with open(filename, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["Timestamp", "Vx (m/s)", "Vy (m/s)", "Vz (m/s)", "Power (W)", "Altitude (m)"])

data_queue = queue.Queue()
recording = True

def record_data():
    while recording or not data_queue.empty():
        try:
            data = data_queue.get(timeout=1)
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(data)
        except queue.Empty:
            continue

recording_thread = threading.Thread(target=record_data, daemon=True)
recording_thread.start()

def generate_astroid_path(radius, vh_speed, vv_speed, duration, center_x=0, center_y=0, start_z=-5.0, points_per_sec=10):
    path = []
    num_points = int(duration * points_per_sec)
    dt = 1.0 / points_per_sec
    
    perimeter = 6.0 * radius
    if vh_speed < 0.1: vh_speed = 0.1
    
    time_per_cycle = perimeter / vh_speed
    omega = 2 * math.pi / time_per_cycle
    
    for i in range(num_points + 1):
        t_sim = i * dt
        theta = omega * t_sim
        
        x = center_x + radius * math.pow(math.cos(theta), 3)
        y = center_y + radius * math.pow(math.sin(theta), 3)
        
        z = start_z - (vv_speed * t_sim) 
        
        path.append(airsim.Vector3r(x, y, z))
    return path

def simulate_power(rotor_states):
    voltage = 15.2 + 0.3 * (2 * random.random() - 1)
    try:
        current = 0.0
        for rotor in rotor_states.rotors:
            speed = rotor['speed']
            speed = min(speed, 900)
            effective_speed = speed * (1 + random.uniform(-0.05, 0.05))
            current += (effective_speed ** 3) * 0.75e-7
    except:
        current = 0.0
    efficiency = 0.8 + random.uniform(-0.02, 0.02)
    return voltage * current * efficiency

sample_freq = 100 
sample_interval = 1 / sample_freq
total_missions = 200 

try:
    for mission_id in range(1, total_missions + 1):
        print(f"\nStarting Mission {mission_id}/{total_missions}...")

        client.reset()
        client.enableApiControl(True)
        client.armDisarm(True)
        client.takeoffAsync().join()
        
        client.hoverAsync().join()
        time.sleep(1)
        start_pos = client.getMultirotorState().kinematics_estimated.position
        start_z = start_pos.z_val
        center_x = start_pos.x_val
        center_y = start_pos.y_val

        wind_x = random.uniform(-2.0, 2.0)
        wind_y = random.uniform(-2.0, 2.0)
        client.simSetWind(airsim.Vector3r(wind_x, wind_y, 0))
        
        vh_speed = random.uniform(5.0, 12.0)
        vv_speed = random.uniform(-1.0, 1.0) 
        radius = random.uniform(30.0, 80.0)
        duration_mission = random.uniform(40.0, 70.0) 

        if abs(vv_speed) < 0.1: vv_speed = 0.1
        total_speed = math.sqrt(vh_speed**2 + vv_speed**2)

        print(f"Starting Astroid flight: Vh={vh_speed:.2f} m/s, Vv={vv_speed:.2f} m/s, R={radius:.2f}m")

        path = generate_astroid_path(radius, vh_speed, vv_speed, duration_mission, 
                                     center_x, center_y, start_z)

        client.moveOnPathAsync(path, 
                             velocity=total_speed, 
                             drivetrain=airsim.DrivetrainType.ForwardOnly,
                             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0))

        start_time = time.time()
        while time.time() - start_time < duration_mission:
            state = client.getMultirotorState()
            kin = state.kinematics_estimated
            
            if state.collision.has_collided:
                print("Collision detected! (High agility maneuver collision expected, data truncated)")
                break
            
            power = simulate_power(client.getRotorStates())
            
            data_queue.put([
                time.time(), 
                kin.linear_velocity.x_val, 
                kin.linear_velocity.y_val,
                -kin.linear_velocity.z_val, 
                power, 
                -kin.position.z_val
            ])
            
            if not state.landed_state == airsim.LandedState.Flying:
                 break
            time.sleep(sample_interval)

        client.hoverAsync().join()
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)
        print(f"Mission {mission_id} complete")

except KeyboardInterrupt:
    print("User interrupted...")
finally:
    recording = False
    recording_thread.join()
    client.reset()
    client.enableApiControl(False)
