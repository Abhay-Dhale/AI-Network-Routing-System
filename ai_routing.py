import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import random
import matplotlib.pyplot as plt

# -----------------------------
# TRAINING DATA GENERATION
# -----------------------------

data = []

for _ in range(1000):
    paths = []
    
    for i in range(3):
        latency = random.randint(10, 150)
        packet_loss = random.uniform(0, 5)
        bandwidth = random.randint(10, 100)
        
        score = latency * 0.5 + packet_loss * 10 - bandwidth * 0.2
        paths.append((latency, packet_loss, bandwidth, score))
    
    best_path = np.argmin([p[3] for p in paths])
    
    row = []
    for p in paths:
        row.extend(p[:3])
    
    row.append(best_path)
    data.append(row)

columns = [
    "lat1","loss1","bw1",
    "lat2","loss2","bw2",
    "lat3","loss3","bw3",
    "best_path"
]

df = pd.DataFrame(data, columns=columns)

X = df.drop("best_path", axis=1)
y = df["best_path"]

model = DecisionTreeClassifier()
model.fit(X, y)

print("✅ AI Routing Model Ready!\n")

# -----------------------------
# REAL-TIME SIMULATION
# -----------------------------

latencies = []

for i in range(5):  # simulate 5 time intervals
    print(f"\n🔄 Simulation Round {i+1}")
    
    inputs = []
    current_lat = []
    
    for p in range(1, 4):
        lat = random.randint(10, 150)
        loss = random.uniform(0, 5)
        bw = random.randint(10, 100)
        
        print(f"Path {p}: Lat={lat}ms Loss={round(loss,2)}% BW={bw}Mbps")
        
        inputs.extend([lat, loss, bw])
        current_lat.append(lat)
    
    prediction = model.predict([inputs])
    
    print("🚀 AI Selected Best Path:", prediction[0] + 1)
    latencies.append(current_lat)

# -----------------------------
# PLOT GRAPH
# -----------------------------

latencies = np.array(latencies)

plt.plot(latencies[:,0])
plt.plot(latencies[:,1])
plt.plot(latencies[:,2])

plt.title("Path Latency Comparison")
plt.xlabel("Time Interval")
plt.ylabel("Latency (ms)")
plt.legend(["Path 1","Path 2","Path 3"])
plt.show()





