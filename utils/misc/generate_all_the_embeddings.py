import os
import subprocess

class_index=[64, 33, 65, 67, 37, 69, 70, 71, 42, 75, 76, 19, 52, 83, 22, 25, 93,99,100]

for i in class_index:
    i=i-1

    command="CUDA_VISIBLE_DEVICES=6 python knn.py --class_num "+ str(i) + " --make_cluster 1"
    print("Running (Making Cluster): ", command)
    process = subprocess.Popen(command, shell=True)
    process.wait()

    command="python knn.py --class_num "+ str(i) + " --make_cluster 0 "+" --choose_random 30"
    print("Running(Choose Random 30): ", command)
    process = subprocess.Popen(command, shell=True)
    process.wait()