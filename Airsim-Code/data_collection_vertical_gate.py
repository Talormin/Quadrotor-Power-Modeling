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

filename = "vertical_gate_dataset.csv"
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

def generate_vertical_gate_path(start_pos, ascent_height, transit_length, step_size=0.5):
    path = []
    x0, y0, z0 = start_pos.x_val, start_pos.y_val, start_pos.z_val
    
    z_target = z0 - ascent_height
    
    z_steps = int(ascent_height / step_size)
    for i in range(z_steps + 1):
        z = z0 - (i * step_size)
        path.append(airsim.Vector3r(x0, y0, z))
        
    x_steps = int(transit_length / step_size)
    for i in range(1, x_steps + 1):
        x = x0 + (i * step_size)
        path.append(airsim.Vector3r(x, y0, z_target))
        
    current_x = x0 + (x_steps * step_size)
    
    for i in range(1, z_steps + 1):
        z = z_target + (i * step_size)
        path.append(airsim.Vector3r(current_x, y0, z))
        
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
        
        target_speed = random.uniform(1.0, 5.0)
        ascent_height = random.uniform(10.0, 40.0)
        transit_length = random.uniform(5.0, 10.0)
        
        total_distance = ascent_height * 2 + transit_length
        estimated_time = total_distance / target_speed
        timeout_duration = estimated_time * 1.5 + 10

        print(f"Starting vertical gate flight: Speed={target_speed:.2f} m/s, Height={ascent_height:.1f}m")

        path = generate_vertical_gate_path(start_pos, ascent_height, transit_length)

        client.moveOnPathAsync(path, 
                             velocity=target_speed, 
                             drivetrain=airsim.DrivetrainType.ForwardOnly,
                             yaw_mode=airsim.YawMode(is_rate=False, yaw_or_rate=0))

        start_time = time.time()
        while time.time() - start_time < timeout_duration:
            state = client.getMultirotorState()
            kin = state.kinematics_estimated
            
            if state.collision.has_collided:
                print("Collision detected!")
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
            
            if (time.time() - start_time > 5.0) and (-kin.position.z_val < 0.5):
                print("Approaching ground, mission complete.")
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
