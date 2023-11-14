from torchvision import datasets, transforms
from torch.utils.data import Subset
import torch
import random
import numpy as np
from torch.utils.data import DataLoader
from misc.convs.cifar_resnet import resnet32
from torch import nn
import json
from misc.convs.linears import SimpleLinear
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import manifold


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

# scale and move the coordinates so they fit [0; 1] range
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
 
    starts_from_zero = x - np.min(x)
 
    return starts_from_zero / value_range



def custom_collate(batch):
    return batch

def train(train_dataloader,model):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(), 
    ])

    all_features=None
    Labels=None

    for batch in train_dataloader:
        
        images=[(image) for image, label in batch]
        labels=[(label) for image, label in batch]
        if Labels is None:
            Labels=labels
        else:
            Labels=np.concatenate((Labels, labels), axis=0)

        preprocessed_images = []
        for image in images:
            preprocessed_image = transform(image).unsqueeze(0).to("cuda")
            preprocessed_images.append(preprocessed_image)   

        image_features = model.extract_vector(torch.cat(preprocessed_images, dim=0)).detach().cpu().numpy()
        if all_features is None:
            all_features=image_features
        else:
            all_features=np.concatenate((all_features, image_features), axis=0)

    return all_features, Labels
def main():
    
    seed = 10
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    random_sample=20
    batch_size = 10

    #task_0=[68, 56, 78,  8, 23, 84, 90, 65, 74, 76]
    #task_0=[68, 56, 78,  8, 23]
    task_0=[84, 90, 65, 74, 76]
    
    """

    task_0_map = {68:'road',
            56: 'palm_tree',
            78: 'snake',
            8: 'bicycle',
            23: 'cloud',
            84: 'table',
            90: 'train',
            65: 'rabbit',
            74: 'shrew',
            76: 'skyscraper'}

    """

    #task_0_map = {68:'road',56: 'palm_tree',78: 'snake',8: 'bicycle',23: 'cloud'}
    task_0_map = {84: 'table',90: 'train',65: 'rabbit',74: 'shrew',76: 'skyscraper'}
    #legend_labels = ['road', 'palm_tree', 'snake', 'bicycle', 'cloud', 'table', 'train', 'rabbit', 'shrew', 'skyscraper']
    legend_labels = ['table', 'train', 'rabbit', 'shrew', 'skyscraper']
    #legend_labels = ['road', 'palm_tree', 'snake', 'bicycle', 'cloud']

    generated_data_path="./25kImages/"
    colors_ = sns.color_palette('Set1', len(task_0))
    palette = {task_0_map[task]: colors_[i] for i, task in enumerate(task_0)}


    param = load_json('./utils/misc/exps/diff_replay.json')
    model = IncrementalNet(param, False)
    model.update_fc(20)
    model.to("cuda")
    model.eval()

    weights = torch.load("./utils/misc/exps/logs/diff_replay/cifar100/0/10/0.pkl", map_location="cuda")
    #weights = torch.load("./utils/misc/exps/1_ti_10c1.pkl", map_location="cuda")
    #weights = torch.load("./utils/misc/exps/ft_ti_10c1.pkl", map_location="cuda")

    m, u = model.load_state_dict(weights['model_state_dict'], False)

    """ CIFAR Original DATA """
    cifar100_train = datasets.CIFAR100(root='./data', train=True, download=False)  
    final_indices=[]
    for t_i in task_0:
        indices = [i for i in range(len(cifar100_train)) if cifar100_train.targets[i]==t_i]
        indices=random.sample(indices, random_sample)
        final_indices=final_indices+indices

    cifar100_subset = torch.utils.data.Subset(cifar100_train, final_indices)
    train_dataloader = DataLoader(cifar100_subset, batch_size=batch_size, shuffle=False,worker_init_fn=np.random.seed(seed),collate_fn=custom_collate)
    all_features, Labels= train(train_dataloader,model)

    """ Generated DATA """
    g_dataset=datasets.ImageFolder(root=generated_data_path)

    final_indices=[]
    for t_i in range(len(task_0)):
        indices = [i for i in range(len(g_dataset)) if g_dataset.targets[i]==t_i]
        indices=random.sample(indices, random_sample)
        final_indices=final_indices+indices

    g_dataset_subset = torch.utils.data.Subset(g_dataset, final_indices)
    train_dataloader = DataLoader(g_dataset_subset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate)
    all_Gfeatures, GLabels= train(train_dataloader,model)
    GLabels=[task_0[i] for i in GLabels]

    """Total features """
    totalFeatures=np.concatenate((all_features, all_Gfeatures), axis=0)

    """T-SNE"""
    tsne = TSNE(n_components=2).fit_transform(totalFeatures)
    each_sec_len=random_sample*len(task_0)

    tsne_1 = tsne[:each_sec_len]
    tsne_2 = tsne[-each_sec_len:]

    cps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                       data=np.column_stack((tsne_1, 
                                            Labels[:(each_sec_len)])))
    cps_df.loc[:, 'target'] = Labels.astype(int)
    cps_df.loc[:, 'target'] = cps_df.target.map(task_0_map)

    Gcps_df = pd.DataFrame(columns=['CP1', 'CP2', 'target'],
                       data=np.column_stack((tsne_2, 
                                            GLabels[:(each_sec_len)])))
    Gcps_df.loc[:, 'target'] = GLabels
    Gcps_df.loc[:, 'target'] = Gcps_df.target.map(task_0_map)

    print("GCPS DF",Gcps_df.head())

    colors = [palette[i] for i in cps_df['target']]
    colors_1 = [palette[i] for i in Gcps_df['target']]

    scatter1 = plt.scatter(cps_df['CP1'], cps_df['CP2'], c=colors, marker='o')
    scatter2 = plt.scatter(Gcps_df['CP1'], Gcps_df['CP2'], c=colors_1, marker='^')

    legend_handles = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, markersize=10) for color in colors_]
    

    plt.legend(handles=legend_handles, labels=legend_labels,bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.subplots_adjust(right=0.8)

    plt.savefig("tsne.jpeg", bbox_inches='tight')

        

if __name__ == "__main__":
    main()


