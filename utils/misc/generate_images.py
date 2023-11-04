import argparse
import os
import shutil
import random

if __name__ == "__main__":


    parser = argparse.ArgumentParser(description="Generate images based on embeddings")
    parser.add_argument("--emb_path", type=str, help="Path to the folder containing images")
    args = parser.parse_args()

    generating_command_ori="CUDA_VISIBLE_DEVICES=6 python scripts/txt2img.py --ddim_eta 0.0 \
    --n_samples 8 --n_iter 2 --scale 5.0 --ddim_steps 200 --ckpt_path /local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt\
     --prompt \"a photo of *\" --outdir "

    arr=[499,999,1499,1999]
    #arr=[499]
    for i in arr:
        generating_command=generating_command_ori+" ../outputs/txt2img-"+str(i)
        generating_command+=" --embedding_path "+ args.emb_path+"/checkpoints/embeddings_gs-"+str(i)+".pt"
        print(generating_command)
        os.system(generating_command)
        print(i)
        shutil.rmtree("../outputs/txt2img-"+str(i)+"/samples/")

    shutil.rmtree(args.emb_path+"/configs/")
    shutil.rmtree(args.emb_path+"/images/")
    shutil.copy(args.emb_path+"/testtube/version_0/metrics.csv",args.emb_path+"/metrics.csv")
    shutil.rmtree(args.emb_path+"/testtube/")

    print("creating loss graphs: ")
    command= "python utils/plot_losses.py "+ args.emb_path+"/metrics.csv"
    print(command)
    #os.system(command)

