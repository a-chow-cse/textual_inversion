import torch
import os
from torchvision import datasets, transforms

directory_path = "./logs/"
subdirectories = [os.path.join(directory_path, d) for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]

cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)
class_names= cifar100_train.classes
class_names=['road', 'palm_tree', 'snake', 'bicycle', 'cloud', 'table', 'train', 'rabbit', 'shrew', 'skyscraper']

steps=[49,99,149]
cluster_names=["cluster_0", "cluster_1", "cluster_2","cluster_3", "cluster_4", "cluster_5","cluster_6", "cluster_7", "cluster_8", "cluster_9"]

output_pth_path = 'cifar_trained_embeddings_task_0_cluster_10_step49_99_149_not_clip.pth'
embeddings_to_save={}


for class_name in class_names:

    step_with_embeddings={}

    for step in steps:

        embeddings_list=[]

        for cluster_name in cluster_names:

            str_to_find="_"+class_name+"_5_"+cluster_name
            folder_names = [d for d in subdirectories if str_to_find in os.path.basename(d).lower()]

            if len(folder_names)!=1: #if there is no folder with that class name and cluster number
                print("wrong"+class_name+ cluster_name)
            else:
                embeddings_path=folder_names[0]+"/checkpoints/"
                embedding= [f for f in os.listdir(embeddings_path) if os.path.isfile(os.path.join(embeddings_path, f)) and ("-"+str(step)) in f]

                if len(embedding)!=1: #if any step embedding is missing to generate
                    print("No embedding for step ", step, " for class ", class_name, " cluster ", cluster_name)
                else:
                    state_dict_emb= torch.load(embeddings_path+"/"+embedding[0])
                    embeddings_list.append(state_dict_emb)

        step_with_embeddings[step]=embeddings_list
    
    embeddings_to_save[class_name]=step_with_embeddings

"""
for key, value in embeddings_to_save.items():
        print(f'{key}:' )
        """
torch.save(embeddings_to_save,output_pth_path)
file_size = os.path.getsize(output_pth_path)
print(f"The size of the file '{output_pth_path}' is {file_size} bytes.")
