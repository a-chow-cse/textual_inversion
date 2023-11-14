import argparse
import os
import shutil
import random
from PIL import Image

import torch
import torchvision
import torchmetrics
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import clip
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import numpy as np
import subprocess
import re

from convs.cifar_resnet import resnet32
import torch
from torch import nn
import json
from convs.linears import SimpleLinear


class BaseNet(nn.Module):
    def __init__(self, args, pretrained):
        super(BaseNet, self).__init__()

        self.convnet = resnet32()
        self.fc = None

    @property
    def feature_dim(self):
        return self.convnet.out_dim

    def extract_vector(self, x):
        return self.convnet(x)["features"]

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        """
        {
            'fmaps': [x_1, x_2, ..., x_n],
            'features': features
            'logits': logits
        }
        """
        out.update(x)

        return out

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

class IncrementalNet(BaseNet):
    def __init__(self, args, pretrained, gradcam=False):
        super().__init__(args, pretrained)
        self.gradcam = gradcam
        if hasattr(self, "gradcam") and self.gradcam:
            self._gradcam_hooks = [None, None]
            self.set_gradcam_hook()

    def update_fc(self, nb_classes):
        fc = self.generate_fc(self.feature_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc


    def generate_fc(self, in_dim, out_dim):
        fc = SimpleLinear(in_dim, out_dim)

        return fc

    def forward(self, x):
        x = self.convnet(x)
        out = self.fc(x["features"])
        out.update(x)
        if hasattr(self, "gradcam") and self.gradcam:
            out["gradcam_gradients"] = self._gradcam_gradients
            out["gradcam_activations"] = self._gradcam_activations

        return out

def load_json(settings_path):
    with open(settings_path) as data_file:
        param = json.load(data_file)

    return param



def encode_images_parallel(train_dataloader, num_gpus,clip_model,preprocess,folder_name,n_clusters):
    
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


def encoder_images_model_feature(train_dataloader,model,folder_name,n_clusters,choose_random):
    encoded_images = []
    allImgFeature=None
    confidences=[]
    probabilities=[]
    class_number=9 #TODO: change according to class in task

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), 
    ])

    with torch.no_grad():

        for batch in train_dataloader:   
            images=[(image) for image, label in batch]

            preprocessed_images = []
            for image in images:
                preprocessed_image = transform(image).unsqueeze(0).to("cuda")
                preprocessed_images.append(preprocessed_image)   

            #print(preprocessed_images)
            image_features = model.extract_vector(torch.cat(preprocessed_images, dim=0))    
            confidence=model(torch.cat(preprocessed_images, dim=0))["logits"]
            
            probabilities=probabilities+ torch.softmax(confidence, dim=1)[:, class_number].tolist()
            #print(len(probabilities),"probabilities shape")#,probabilities)
            

            if allImgFeature==None:
                allImgFeature=image_features
            else:
                allImgFeature=torch.cat((allImgFeature, image_features), dim=0)

            
            encoded_images=encoded_images + images

    #exit()
    allImgFeature /= allImgFeature.norm(dim=-1, keepdim=True)
    allImgFeature_np = allImgFeature.cpu().numpy()
    

    kmeans = KMeans(n_clusters=n_clusters, random_state=0, init="k-means++", n_init=10, max_iter=1000)
    cluster_labels = kmeans.fit_predict(allImgFeature_np)


    for j in range(n_clusters):
        print("Saving cluster ",j+1)
        path_to_save=folder_name+"/cluster_"+str(j)
        os.makedirs(path_to_save)

        this_cluster_confidence=[]
        position=[]

        for i in range(len(cluster_labels)):
            if cluster_labels[i]==j:
                this_cluster_confidence.append(probabilities[i])
                position.append(i)
        print(sorted(this_cluster_confidence, reverse=True)[:choose_random])
        top_5_indices=sorted(range(len(sorted(this_cluster_confidence, reverse=True))), key=lambda k: sorted(this_cluster_confidence, reverse=True)[k])[:choose_random]
        for k in top_5_indices:
            encoded_images[position[k]].save(path_to_save+"/image_"+str(position[k])+".png")



def custom_collate(batch):
    return batch


def selectTop100PerClass(train_dataloader, num_gpus,clip_model,preprocess,folder_name, class_name,top_k):
    encoded_images = []
    allImgFeature=None

    with torch.no_grad():

        for batch in train_dataloader:   
            images=[(image) for image, label in batch]
            #print(images)

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

        class_label_embedding = clip_model.encode_text(clip.tokenize([class_name]).to("cuda"))#.repeat(len(encoded_images), 1)

        allImgFeature /= allImgFeature.norm(dim=-1, keepdim=True)
        class_label_embedding /= class_label_embedding.norm(dim=-1, keepdim=True)

        similarity_scores = torch.mm(allImgFeature, class_label_embedding.t())
        top_k_indices = torch.topk(similarity_scores, top_k, dim=0).indices

        return similarity_scores
        #for i in top_k_indices:
            #print("index: ",i[0].item()," similarity scoreS: ",similarity_scores[i[0].item()] )
            #encoded_images[i[0].item()].save(folder_name+"/image_"+str(i[0].item())+".png")

        #return clip_score


if __name__ == "__main__":

    torch.manual_seed(42)
    random.seed(42)

    parser = argparse.ArgumentParser(description="Cluster images using CLIP embeddings and plot the distribution.")
    parser.add_argument("--data_path", type=str, help="Path to the folder containing images")
    parser.add_argument("--class_num", type=int, default=11,help="which class to cluster of cifar_100")
    parser.add_argument("--make_cluster", type=int, default=1,help="purpose: cluster making of a dataset")
    parser.add_argument("--choose_random", type=int, default=100,help="how many images for each cluster")
    parser.add_argument("--show_cluster", type=int,default=0, help="how the cluster of current class and cluster number")
    parser.add_argument("--choose_1_cluster_top100Image", type=int,default=0, help="choose 100 images based on clip scores from the class to make a cluster")
    args = parser.parse_args()
    n_clusters=10
    model_for_cluster="model_extractor"

    if args.choose_1_cluster_top100Image==1:
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
        selectTop100PerClass(train_dataloader,4,clip_model,preprocess,folder_name,class_name,50)



    elif args.make_cluster==1:

        cifar100_train = datasets.CIFAR100(root='./data', train=True, download=False)
        class_name= cifar100_train.classes[args.class_num]
        filtered_cifar100_train = torch.utils.data.Subset(cifar100_train, [i for i in range(len(cifar100_train)) if cifar100_train.targets[i] == args.class_num])
        
        print("Class Name: ",class_name, " Number of images: ",len(filtered_cifar100_train))
        folder_name=os.getcwd()+"/data_c1_img500/"+str(args.class_num)+"_"+class_name

        train_dataloader = DataLoader(filtered_cifar100_train, batch_size=2, shuffle=True,worker_init_fn=np.random.seed(42),collate_fn=custom_collate)


        if os.path.exists(folder_name):
            shutil.rmtree(folder_name)
        os.makedirs(folder_name)

        if model_for_cluster=="clip":
            clip_model, preprocess = clip.load("ViT-B/32", device="cuda")
            encoded_images = encode_images_parallel(train_dataloader,4,clip_model,preprocess,folder_name,n_clusters)

            top_k=args.choose_random
            destination_path_o=folder_name+"/current_image"
            similarity_scores=selectTop100PerClass(train_dataloader, 2,clip_model,preprocess,folder_name, class_name,top_k)

            for i in range(n_clusters):
                path=folder_name+"/cluster_"+str(i)
                destination_path=destination_path_o+str(i)

                if os.path.exists(destination_path):
                    print("Already cluster exists")
                else:
                    os.makedirs(destination_path)
                image_paths = os.listdir(path)

                scores=[]
                for image_path in image_paths:
                    print(re.search(r'(\d+)\.png', image_path).group(1),image_path)
                    scores.append(similarity_scores[int(re.search(r'(\d+)\.png', image_path).group(1))])

                scores_tensor = torch.tensor(scores, dtype=torch.float)
                top_k_indices = torch.topk(scores_tensor, top_k, dim=0).indices
                print(top_k_indices)

                
                for idx in top_k_indices:
                    shutil.copy(path+"/"+image_paths[idx], destination_path+"/"+image_paths[idx])
        else:
            param = load_json('./utils/misc/exps/diff_replay.json')
            model = IncrementalNet(param, False)
            model.update_fc(10)
            model.to("cuda")

            weights = torch.load("./utils/misc/exps/logs/diff_replay/cifar100/0/10/0.pkl", map_location="cuda")
            m, u = model.load_state_dict(weights['model_state_dict'], False)
            encoded_images = encoder_images_model_feature(train_dataloader,model,folder_name,n_clusters,args.choose_random)
                


        #else:
        #    shutil.rmtree(folder_name)
        #    os.makedirs(folder_name)
        #

    else:

        cifar100_train = datasets.CIFAR100(root='./data', train=True, download=False)
        class_name= cifar100_train.classes[args.class_num]
        folder_name=os.getcwd()+"/data_c10_img5/"+str(args.class_num)+"_"+class_name

        destination_path_o=folder_name+"/cluster_"

        m=args.choose_random

        for i in range(2,n_clusters):
            destination_path=destination_path_o+str(i)

            init_word_to_put= class_name.split("_")[-1]
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
            elif init_word_to_put=="shrew":
                init_word_to_put="mammal"

            init_word_to_put="realistic"

            
            train_command = "python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune_style_c3_img5.yaml -t \
            --actual_resume /local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt -n "+ "\""+class_name+\
            "_"+str(m)+"_cluster_"+str(i)+"\" --gpus 6, --data_root "+destination_path+ " --init_word "+init_word_to_put
            print("commands_to_run: cluster:",i)
            print(train_command)
            process = subprocess.Popen(train_command, shell=True)
            process.wait()
            #shutil.rmtree(path)


#python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume /local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt -n "boy_10_c1" --gpus 4,7, --data_root ./data/11_boy/cluster_0/--init_word boy
