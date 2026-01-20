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

filename = "vertical_climb_data.csv"
vz = -1

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

recording_thread = threading.Thread(target=record_data)
recording_thread.daemon = True
recording_thread.start()

client.takeoffAsync().join()
client.hoverAsync().join()
print("Takeoff complete. Hovering for 2 seconds...")
time.sleep(2)

print("Starting vertical climb and data recording...")

client.simSetWind(airsim.Vector3r(2, 1, 0))

duration = 30
start_time = time.time()
client.moveByVelocityAsync(vx=0, vy=0, vz=vz, duration=duration)

try:
    while time.time() - start_time < duration:
        state = client.getMultirotorState()
        kinematics = state.kinematics_estimated
        rotor_states = client.getRotorStates()

        vx = kinematics.linear_velocity.x_val
        vy = kinematics.linear_velocity.y_val
        vz = -kinematics.linear_velocity.z_val
        altitude = -kinematics.position.z_val

        voltage = 15.2 + (0.3 * (2 * random.random() - 1))
        try:
            current = 0.0
            for rotor in rotor_states.rotors:
                speed = rotor['speed']
                speed = min(speed, 900)
                effective_speed = speed * (1 + random.uniform(-0.05, 0.05))
                current += (effective_speed ** 3) * 0.75* 1e-7
        except Exception:
            current = 0.0
        
        efficiency = 0.8 + random.uniform(-0.02, 0.02)
        power = voltage * current * efficiency
       
        data_queue.put([time.time(), vx, vy, vz, power, altitude])
        time.sleep(0.01)

finally:
    client.hoverAsync().join()
    client.landAsync().join()
    recording = False
    recording_thread.join()
    print("Data collection complete. File saved:", filename)
