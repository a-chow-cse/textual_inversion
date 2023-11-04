import argparse
import os
import shutil
import random
from PIL import Image

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import subprocess


def encode_images_parallel(train_dataloader, num_gpus,clip_model,preprocess,folder_name):
    
    encoded_images = []
    allImgFeature=None

    with torch.no_grad():

        for batch in train_dataloader:   
            images=[(image) for image, label in batch]

            preprocessed_images = []
            for image in images:
                preprocessed_image = preprocess(image).unsqueeze(0).to("cuda")
                preprocessed_images.append(preprocessed_image)

            image_features = clip_model.encode_image(torch.cat(preprocessed_images, dim=0).to("cuda"))
    

            if allImgFeature==None:
                allImgFeature=image_features
            else:
                allImgFeature=torch.cat((allImgFeature, image_features), dim=0)

            
            encoded_images=encoded_images + images

    n_clusters=3
    #print(allImgFeature.shape," allImgFeature shape")
    allImgFeature /= allImgFeature.norm(dim=-1, keepdim=True)
    allImgFeature_np = allImgFeature.cpu().numpy()
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++", n_init=10, max_iter=1000)
    cluster_labels = kmeans.fit_predict(allImgFeature_np)

    for j in range(n_clusters):
        print("Saving cluster ",j+1)
        path_to_save=folder_name+"/cluster_"+str(j)
        os.makedirs(path_to_save)
        for i in range(len(cluster_labels)):
            if cluster_labels[i]==j:
                encoded_images[i].save(path_to_save+"/image_"+str(i)+".png")

def custom_collate(batch):
    return batch

if __name__ == "__main__":

    torch.manual_seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="Cluster images using CLIP embeddings and plot the distribution.")
    parser.add_argument("--data_path", type=str, help="Path to the folder containing images")
    parser.add_argument("--class_num", type=int, default=1,help="which class to cluster of cifar_100")
    parser.add_argument("--make_cluster", type=int, default=1,help="purpose: cluster making of a dataset")
    parser.add_argument("--choose_random", type=int, default=30,help="how many images for each cluster")
    parser.add_argument("--show_cluster", type=int,default=0, help="Path to the folder containing images")
    args = parser.parse_args()


    if args.show_cluster ==1:
        clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

        cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)
        class_name= cifar100_train.classes[args.class_num]
        filtered_cifar100_train = torch.utils.data.Subset(cifar100_train, [i for i in range(len(cifar100_train)) if cifar100_train.targets[i] == args.class_num])
        
        print("Class Name: ",class_name, " Number of images: ",len(filtered_cifar100_train))
        folder_name=os.getcwd()+"/data/"+str(args.class_num)+"_"+class_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
        else:
            shutil.rmtree(folder_name)
            os.makedirs(folder_name)
        train_dataloader = DataLoader(filtered_cifar100_train, batch_size=64, shuffle=True,worker_init_fn=np.random.seed(42),collate_fn=custom_collate)
        encoded_images = encode_images_parallel(train_dataloader,4,clip_model,preprocess,folder_name)

        destination_path_o=folder_name+"/current_image"

        n_clusters=3
        m=args.choose_random

        for i in range(n_clusters):
            path=folder_name+"/cluster_"+str(i)
            destination_path=destination_path_o+str(i)
            
            if os.path.exists(destination_path):
                #shutil.rmtree(destination_path)
                #os.makedirs(destination_path)
                print("Already cluster exists")
            else:
                image_paths = os.listdir(path)
                selected_image_paths = random.sample(image_paths, m)

                os.makedirs(destination_path)
                for image_path in selected_image_paths:
                    shutil.copy(path+"/"+image_path, destination_path+"/"+image_path)

            shutil.rmtree(path)

            image_files = [f for f in os.listdir(destination_path) if f.endswith('.png')]

            images = []

            for image_file in image_files:
                image = Image.open(os.path.join(destination_path, image_file))
                images.append(image)

            width, height = images[0].size

            images_per_row = 10
            num_rows = len(images) // images_per_row + (len(images) % images_per_row > 0)
            combined_width = width * images_per_row
            combined_height = height * num_rows
            combined_image = Image.new('RGB', (combined_width, combined_height))

            for j, image in enumerate(images):
                row = j // images_per_row
                col = j % images_per_row
                combined_image.paste(image, (col * width, row * height))

            combined_image.save("./combined_image_cluster"+str(i)+".png")


        

    elif args.make_cluster==1:
        clip_model, preprocess = clip.load("ViT-B/32", device="cuda")

        cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)
        class_name= cifar100_train.classes[args.class_num]
        filtered_cifar100_train = torch.utils.data.Subset(cifar100_train, [i for i in range(len(cifar100_train)) if cifar100_train.targets[i] == args.class_num])
        
        print("Class Name: ",class_name, " Number of images: ",len(filtered_cifar100_train))
        folder_name=os.getcwd()+"/data/"+str(args.class_num)+"_"+class_name
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)
            train_dataloader = DataLoader(filtered_cifar100_train, batch_size=64, shuffle=True,worker_init_fn=np.random.seed(42),collate_fn=custom_collate)
            encoded_images = encode_images_parallel(train_dataloader,4,clip_model,preprocess,folder_name)
        #else:
        #    shutil.rmtree(folder_name)
        #    os.makedirs(folder_name)

        
    else:

        cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)
        class_name= cifar100_train.classes[args.class_num]
        folder_name=os.getcwd()+"/data/"+str(args.class_num)+"_"+class_name

        destination_path_o=folder_name+"/current_image"

        n_clusters=3
        m=args.choose_random

        for i in range(n_clusters):
            path=folder_name+"/cluster_"+str(i)
            destination_path=destination_path_o+str(i)
            
            if os.path.exists(destination_path):
                #shutil.rmtree(destination_path)
                #os.makedirs(destination_path)
                print("Already cluster exists")
            else:
                image_paths = os.listdir(path)
                selected_image_paths = random.sample(image_paths, m)

                os.makedirs(destination_path)
                for image_path in selected_image_paths:
                    shutil.copy(path+"/"+image_path, destination_path+"/"+image_path)
                #print(image_path)

            init_word_to_put=    class_name.split("_")[-1]
            if init_word_to_put=="cockroach" or init_word_to_put=="caterpillar":
                init_word_to_put="insect"
            elif init_word_to_put=="chimpanzee":
                init_word_to_put="animal"
            elif init_word_to_put=="sunflower" or init_word_to_put=="tulip":
                init_word_to_put="flower"
            elif init_word_to_put=="porcupine" or init_word_to_put=="possum" or init_word_to_put=="raccoon":
                init_word_to_put="animal"
            elif init_word_to_put=="flatfish":
                init_word_to_put="fish"

            
            """

            train_command = "python ../main.py --base ../configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t \
            --actual_resume /local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt -n "+ "\""+class_name+\
            "_"+str(m)+"_cluster_"+str(i)+"\" --gpus 0,7, --data_root "+destination_path+ " --init_word "+init_word_to_put
            print("commands_to_run: cluster:",i)
            print(train_command)
            process = subprocess.Popen(train_command, shell=True)
            process.wait()
            shutil.rmtree(path)
            """

