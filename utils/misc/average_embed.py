import os
import torch

directory_path = "./logs/"
subdirectories = [os.path.join(directory_path, d) for d in os.listdir(directory_path) if os.path.isdir(os.path.join(directory_path, d))]
word="ray"

folder_names = [d for d in subdirectories if word in os.path.basename(d).lower()]
absolute_paths = [os.path.abspath(folder_name) for folder_name in folder_names]

emb_499=None
emb_999=None
emb_1499=None

for folder_name in absolute_paths:
    checkpoint_path= folder_name+"/checkpoints/"
    print(checkpoint_path)
    emb_499_path=checkpoint_path+"embeddings_gs-49.pt"
    emb_999_path=checkpoint_path+"embeddings_gs-99.pt"
    emb_1499_path=checkpoint_path+"embeddings_gs-149.pt"

    if os.path.exists(emb_499_path):
        print(emb_499_path)
        n= torch.load(emb_499_path)
        if emb_499==None:
            emb_499=n
        else:
            emb_499['string_to_param']['*']=torch.nn.Parameter(torch.add(emb_499['string_to_param']['*'],n['string_to_param']['*']))
            print(emb_499['string_to_param']['*'][:,:3])
    if os.path.exists(emb_999_path):
        print(emb_999_path)
        n= torch.load(emb_999_path)
        if emb_999==None:
            emb_999=n
        else:
            emb_999['string_to_param']['*']=torch.nn.Parameter(torch.add(emb_999['string_to_param']['*'],n['string_to_param']['*']))
    if os.path.exists(emb_1499_path):
        print(emb_1499_path)
        n= torch.load(emb_1499_path)
        if emb_1499==None:
            emb_1499=n
        else:
            emb_1499['string_to_param']['*']=torch.nn.Parameter(torch.add(emb_1499['string_to_param']['*'],n['string_to_param']['*']))
    

emb_499['string_to_param']['*'] = torch.nn.Parameter(emb_499['string_to_param']['*'] /3)
emb_999['string_to_param']['*'] = torch.nn.Parameter(emb_999['string_to_param']['*'] /3)
emb_1499['string_to_param']['*'] = torch.nn.Parameter(emb_1499['string_to_param']['*'] /3)

torch.save(emb_499, "../emb_avg_499.pt")
torch.save(emb_999, "../emb_avg_999.pt")
torch.save(emb_1499, "../emb_avg_1499.pt")
  