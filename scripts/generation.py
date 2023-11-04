import argparse, os, sys, glob
import torch
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from tqdm import tqdm, trange
from einops import rearrange
from torchvision.utils import make_grid

from ldm.util import instantiate_from_config
from ldm.models.diffusion.ddim import DDIMSampler
from ldm.models.diffusion.plms import PLMSSampler

def setup_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--prompt",
        type=str,
        nargs="?",
        default="a photo of *",
        help="the prompt to render"
    )

    parser.add_argument(
        "--outdir",
        type=str,
        nargs="?",
        help="dir to write results to",
        default="outputs/txt2img-samples"
    )
    parser.add_argument(
        "--ddim_steps",
        type=int,
        default=50,
        help="number of ddim sampling steps",
    )

    parser.add_argument(
        "--plms",
        action='store_true',
        help="use plms sampling",
    )

    parser.add_argument(
        "--ddim_eta",
        type=float,
        default=0.0,
        help="ddim eta (eta=0.0 corresponds to deterministic sampling",
    )
    parser.add_argument(
        "--n_iter",
        type=int,
        default=1,
        help="sample this often",
    )

    parser.add_argument(
        "--H",
        type=int,
        default=256,
        help="image height, in pixel space",
    )

    parser.add_argument(
        "--W",
        type=int,
        default=256,
        help="image width, in pixel space",
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for the given prompt",
    )

    parser.add_argument(
        "--scale",
        type=float,
        default=10.0,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )

    parser.add_argument(
        "--ckpt_path", 
        type=str, 
        default="/local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt", 
        help="Path to pretrained ldm text2img model")

    parser.add_argument(
        "--embedding_path", 
        type=str, 
        help="Path to a pre-trained embedding manager checkpoint")
    
    parser.add_argument(
        "--embedding_ckpt", 
        type=str,
        default="./utils/cifar_trained_embeddings.pth", 
        help="Path to a pre-trained all embedding model pth ")

    parser.add_argument(
        "--step", 
        type=int,
        default=0,
        choices=[499, 999, 1499], 
        help="global step for embeddings")

    parser.add_argument(
        "--class_name", 
        type=str,
        default= "" ,
        help="class_name to consider for generation from model")

    parser.add_argument(
        "--cluster", 
        type=int,
        default=-1,
        choices=[0, 1, 2] ,
        help="global step for embeddings")

    
    return parser

def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model

def generate_images_and_returnNumpyArray(prompt, 
    ddim_steps, model, all_embeddings_ckpt, class_name, step,cluster,n_samples,n_iter,
    scale=10.0,ddim_eta=0.0,H=256,W=256):

    sampler = DDIMSampler(model)
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(n_samples * [""])
            for n in trange(n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(n_samples * [prompt])
                shape = [4, H//8, W//8]
                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc,
                                                 eta=ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for i in range(len(x_samples_ddim)):
                    x_samples_ddim[i]= 255. * rearrange(x_samples_ddim[i].cpu().numpy(), 'c h w -> h w c')



def load_embeddings_diffusion_model(embedding_ckpt_path,device, config_path):
    config = OmegaConf.load(config_path)  
    model = load_model_from_config(config, embedding_ckpt_path)
    all_embeddings_ckpt= torch.load(embedding_ckpt_path,map_location = device )

    return model, all_embeddings_ckpt

def generate_images_save(model, opt):

    if opt.plms:
        sampler = PLMSSampler(model)
    else:
        sampler = DDIMSampler(model)

    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    prompt = opt.prompt

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    all_samples=list()
    with torch.no_grad():
        with model.ema_scope():
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(opt.n_samples * [""])
            for n in trange(opt.n_iter, desc="Sampling"):
                c = model.get_learned_conditioning(opt.n_samples * [prompt])
                shape = [4, opt.H//8, opt.W//8]
                samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                                 conditioning=c,
                                                 batch_size=opt.n_samples,
                                                 shape=shape,
                                                 verbose=False,
                                                 unconditional_guidance_scale=opt.scale,
                                                 unconditional_conditioning=uc,
                                                 eta=opt.ddim_eta)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, min=0.0, max=1.0)

                for x_sample in x_samples_ddim:
                    x_sample = 255. * rearrange(x_sample.cpu().numpy(), 'c h w -> h w c')
                    Image.fromarray(x_sample.astype(np.uint8)).save(os.path.join(sample_path, f"{base_count:04}.jpg"))
                    base_count += 1
                all_samples.append(x_samples_ddim)


    # additionally, save as grid
    grid = torch.stack(all_samples, 0)
    grid = rearrange(grid, 'n b c h w -> (n b) c h w')
    grid = make_grid(grid, nrow=opt.n_samples)

    # to image
    grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
    Image.fromarray(grid.astype(np.uint8)).save(os.path.join(outpath, f'{prompt.replace(" ", "-")}.jpg'))


if __name__ == "__main__":
    
    parser = setup_parser()
    opt = parser.parse_args()

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    if device=='cpu':
        print("Need a gpu to run, cuda not available!")
        exit()

    config = OmegaConf.load("configs/latent-diffusion/txt2img-1p4B-eval_with_tokens.yaml")  
    model = load_model_from_config(config, opt.ckpt_path)
    
    if '*' not in opt.prompt:
        print("There is no * in prompt. Direct diffusion Image generation")
        opt.outdir="outputs/"+opt.prompt+"_ddim"+str(opt.ddim_steps)
    elif opt.embedding_path:
        model.embedding_manager.load(opt.embedding_path)
    elif opt.embedding_ckpt:
        all_embeddings_ckpt= torch.load(opt.embedding_ckpt,map_location = device )
        if opt.class_name=="":
            print("Please provide class name to choose the embedding from the pretrained embeddings")
            exit()

        if opt.step=="":
            print("Please provide which step to choose the embedding from the pretrained embeddings choices[499,999,1499]")
            exit()

        if opt.cluster=="":
            print("Please provide which cluster to choose the embedding from the pretrained embeddings choices[0,1,2]")
            exit()

        selected_embedding=all_embeddings_ckpt[opt.class_name][opt.step][opt.cluster]
        #print(selected_embedding)
        model.embedding_manager.load_from_script(selected_embedding)
        opt.outdir="outputs/"+opt.class_name+"_"+str(opt.step)+"_"+str(opt.cluster)+"_"+"ddim"+str(opt.ddim_steps)

    model = model.to(device)
    generate_images_save(model, opt)


    
 
