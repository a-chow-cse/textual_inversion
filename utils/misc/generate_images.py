import argparse
import os
import shutil
import random

if __name__ == "__main__":

    templates = [
        'a photo of a *.',
        'a blurry photo of a *.',
        'a black and white photo of a *.',
        'a low contrast photo of a *.',
        'a high contrast photo of a *.',
        'a bad photo of a *.',
        'a good photo of a *.',
        'a photo of a small *.',
        'a photo of a big *.',
        'a photo of the *.',
        'a blurry photo of the *.',
        'a black and white photo of the *.',
        'a low contrast photo of the *.',
        'a high contrast photo of the *.',
        'a bad photo of the *.',
        'a good photo of the *.',
        'a photo of the small *.',
        'a photo of the big *.',
    ]

    parser = argparse.ArgumentParser(description="Generate images based on embeddings")
    parser.add_argument("--emb_path", type=str, help="Path to the folder containing images")
    args = parser.parse_args()

    directory_path=args.emb_path
    snake_folders = [folder for folder in os.listdir(directory_path)
                 if os.path.isdir(os.path.join(directory_path, folder)) and 'snake' in folder]

    arr=[49,99,149]
    for folder in snake_folders:
        folder_path = os.path.join(directory_path, folder)
        for i in arr:
            for prmpt in templates:
                generating_command_ori="CUDA_VISIBLE_DEVICES=0 python scripts/txt2img.py --ddim_eta 0.0 \
                --n_samples 4 --n_iter 2 --scale 5.0 --ddim_steps 200 --ckpt_path /local/scratch/chowdhury.150/model.ckpt\
                 --outdir "

                generating_command=generating_command_ori+" ../outputs_cifar_prompts/txt2img-"+str(i)
                generating_command+=" --embedding_path "+ folder_path+"/checkpoints/embeddings_gs-"+str(i)+".pt"

                generating_command+=" --prompt \""+prmpt+"\" "
                print(generating_command)
                os.system(generating_command)
                print(i)
        #shutil.rmtree(" ../outputs_cifar_prompts/txt2img-"+str(i)+"/samples/")
    exit()
    shutil.rmtree(args.emb_path+"/configs/")
    shutil.rmtree(args.emb_path+"/images/")
    shutil.copy(args.emb_path+"/testtube/version_0/metrics.csv",args.emb_path+"/metrics.csv")
    shutil.rmtree(args.emb_path+"/testtube/")

    #os.system(command)

