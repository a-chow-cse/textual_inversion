import os
import subprocess

class_index=[ 78]

for i in class_index:
    i=i

    """command="CUDA_VISIBLE_DEVICES=6 python knn.py --class_num "+ str(i) + " --make_cluster 1"
    print("Running (Making Cluster): ", command)
    process = subprocess.Popen(command, shell=True)
    process.wait()
    """

    command="python ./utils/misc/knn.py --class_num "+ str(i) + " --make_cluster 0"
    print("Running: ", command)
    process = subprocess.Popen(command, shell=True)
    process.wait()