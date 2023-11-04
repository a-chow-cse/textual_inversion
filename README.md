## **Setup**

Run:

```
conda env create -f environment.yaml

conda activate ldm

mkdir -p models/ldm/text2img-large/

wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## **get Cifar-100 fine-grained class** 
Work has been done for cifar-100 dataset

```
python utils/get_cifar_information.py
```

## **Pretrained Embeddings Path**
Path: `./utils/cifar_trained_embeddings.pth`

How is it stored:
`{'apple':{ 499: [pt1,pt2,pt3], 999:[pt1,pt2,pt3], 1499:[pt1,pt2,pt3] }, 'aquarium_fish':{...}}`

```
├── class name 1              
   ├── 499
      ├── cluster 0 pt
      ├── cluster 1 pt
      ├── cluster 2 pt
   ├── 999
      ├── cluster 0 pt
      ├── cluster 1 pt
      ├── cluster 2 pt         
   └── 1499 
      ├── cluster 0 pt
      ├── cluster 1 pt
      ├── cluster 2 pt               
├── class name 2                    
   ├── ...         
```

## **Generate images**
- **Prompt with pretrained embedding pth for class name=boy, step=499, cluster=0 ddim=50**
```
CUDA_VISIBLE_DEVICES=7 python scripts/generation.py --embedding_ckpt ./utils/cifar_trained_embeddings.pth --class_name boy --step 499 --cluster 0 --ddim_steps 200 --prompt "a photo of *"
```
saved in outputs/boy_499_0_ddim200/
![Example Image](./outputsDemo/boy_499_0_ddim200/a-photo-of-*.jpg)


- **Prompt with pretrained embedding pth for class name=aquarium_fish, step=499, cluster=0**
```
CUDA_VISIBLE_DEVICES=7 python scripts/generation.py --embedding_ckpt ./utils/cifar_trained_embeddings.pth --class_name aquarium_fish --step 499 --cluster 0 --ddim_steps 50 --prompt "a photo of *"
```
saved in outputs/aquarium_fish_499_0_ddim50/
![Example Image](./outputsDemo/aquarium_fish_499_0_ddim50/a-photo-of-*.jpg)



- **Prompt without pretrained embedding, with custom prompt**
```
CUDA_VISIBLE_DEVICES=7 python scripts/generation.py --embedding_ckpt ./utils/cifar_trained_embeddings.pth --ddim_steps 50 --prompt "A photo of apple"
```
saved in outputs/A photo of apple_ddim50/
![Example Image](./outputsDemo/A%20photo%20of%20apple_ddim50/A-photo-of-apple.jpg)

```
CUDA_VISIBLE_DEVICES=7 python scripts/generation.py --embedding_ckpt ./utils/cifar_trained_embeddings.pth --ddim_steps 50 --prompt "A photo of boy"
```
saved in outputs/A photo of apple_ddim50/
![Example Image](./outputsDemo/A%20photo%20of%20boy_ddim50/A-photo-of-boy.jpg)

## **using generation from a code**
In case to intregate with another code, images can be returned as numpy array without saving as images in disk. 
#TODO: have to check by using the methods

**To load the model and embeddings from the checkpoint**
`def load_embeddings_diffusion_model(embedding_ckpt_path,device, config_path)` from generation.py

**To get images as numpy array**

`def generate_images_and_returnNumpyArray(prompt, 
    ddim_steps, model, all_embeddings_ckpt, class_name, step,cluster,n_samples,n_iter,
    scale=10.0,ddim_eta=0.0,H=256,W=256):` from generation.py

- prompt =" a photo of * " (* will mean embedding vector)
- class_name = cifar100 fine_grained labels
- step = (499,999,1499)
- cluster = (0,1,2)
- n_samples = how many images to get per sampling
- n_iter = how many samples to do from the model. Each sampling will provide slight different images for randomness
- ddim_eta = higher value will give lower quality image but with high speed (0-0.5)
- scale = lower the value, less guidance from the embedding vector. usual good value(7-13)
### **Step demo**
class name = boy, cluster=0, ddim=50
- step 499
![Example Image](./outputsDemo/boy_499_0_ddim50/a-photo-of-*.jpg)
- step 999
![Example Image](./outputsDemo/boy_999_0_ddim50/a-photo-of-*.jpg)
- step 1499
![Example Image](./outputsDemo/boy_1499_0_ddim50/a-photo-of-*.jpg)

### **ddim Demo**
class name = boy, cluster=0, step=499
- ddim 50
![Example Image](./outputsDemo/boy_499_0_ddim50/a-photo-of-*.jpg)
- ddim 200
![Example Image](./outputsDemo/boy_499_0_ddim200/a-photo-of-*.jpg)

### **cluster Demo**
class name = boy, ddim=50, step=499
- cluster 0
![Example Image](./outputsDemo/boy_499_0_ddim50/a-photo-of-*.jpg)
- cluster 1
![Example Image](./outputsDemo/boy_499_1_ddim50/a-photo-of-*.jpg)
- cluster 2
![Example Image](./outputsDemo/boy_499_2_ddim50/a-photo-of-*.jpg)

### **Scale Demo**
class name = boy, cluster=0, ddim=200, step=1499
- scale 12
![Example Image](./outputsDemo/boy_1499_0_ddim200_scale_diff/a-photo-of-*_scale12.jpg)
- scale 10
![Example Image](./outputsDemo/boy_1499_0_ddim200/a-photo-of-*.jpg)
- scale 6
![Example Image](./outputsDemo/boy_1499_0_ddim200_scale_diff/a-photo-of-*_scale6.jpg)
- scale 4
![Example Image](./outputsDemo/boy_1499_0_ddim200_scale_diff/a-photo-of-*scale4.jpg)
- scale 4
![Example Image](./outputsDemo/boy_1499_0_ddim200_scale_diff/a-photo-of-*_scale1.jpg)


### **ddim_eta Demo**
class name = boy, cluster=0, ddim=200, step=1499
- ddim_eta 0.0
![Example Image](./outputsDemo/boy_1499_0_ddim200/a-photo-of-*.jpg)
- ddim_eta 0.25
![Example Image](./outputsDemo/boy_1499_0_ddim200_ddimEta_0.25/a-photo-of-*.jpg)


## **To see the selected images from a cluster of a class**

```
CUDA_VISIBLE_DEVICES=6 python utils/misc/knn.py --class_num 11 --show_cluster 1
```
output will be in textual)inversion folder named `combined_image_cluster0.png`, `combined_image_cluster1.png` and `combined_image_cluster2.png`

- cluster 0
![Example Image](./outputsDemo/combined_image_cluster0.png)
- cluster 1
![Example Image](./outputsDemo/combined_image_cluster1.png)
- cluster 2
![Example Image](./outputsDemo/combined_image_cluster2.png)

**END**
* * *
* * *
***


## Image Generation from one single embedding
```
CUDA_VISIBLE_DEVICES=3 python scripts/txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 2 --scale 10.0 --ddim_steps 50 --ckpt_path /local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt --prompt "a photo of apple" --embedding_path ./utils/logs/current_image22023-10-30T15-50-24_apple_3_cluster_2/checkpoints/embeddings_gs-499.pt
```
Run from textual_inversion dir
`python generate_images.py --embedding_path ./utils/logs/.../ <before checkpoints>`


# An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion

[![arXiv](https://img.shields.io/badge/arXiv-2208.01618-b31b1b.svg)](https://arxiv.org/abs/2208.01618)

[[Project Website](https://textual-inversion.github.io/)]

> **An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion**<br>
> Rinon Gal<sup>1,2</sup>, Yuval Alaluf<sup>1</sup>, Yuval Atzmon<sup>2</sup>, Or Patashnik<sup>1</sup>, Amit H. Bermano<sup>1</sup>, Gal Chechik<sup>2</sup>, Daniel Cohen-Or<sup>1</sup> <br>
> <sup>1</sup>Tel Aviv University, <sup>2</sup>NVIDIA

>**Abstract**: <br>
> Text-to-image models offer unprecedented freedom to guide creation through natural language.
  Yet, it is unclear how such freedom can be exercised to generate images of specific unique concepts, modify their appearance, or compose them in new roles and novel scenes.
  In other words, we ask: how can we use language-guided models to turn <i>our</i> cat into a painting, or imagine a new product based on <i>our</i> favorite toy?
  Here we present a simple approach that allows such creative freedom.
  Using only 3-5 images of a user-provided concept, like an object or a style, we learn to represent it through new "words" in the embedding space of a frozen text-to-image model.
  These "words" can be composed into natural language sentences, guiding <i>personalized</i> creation in an intuitive way.
  Notably, we find evidence that a <i>single</i> word embedding is sufficient for capturing unique and varied concepts.
  We compare our approach to a wide range of baselines, and demonstrate that it can more faithfully portray the concepts across a range of applications and tasks.

## Description
This repo contains the official code, data and sample inversions for our Textual Inversion paper. 

## Updates
**29/08/2022** Merge embeddings now supports SD embeddings. Added SD pivotal tuning code (WIP), fixed training duration, checkpoint save iterations.
**21/08/2022** Code released!

## TODO:
- [x] Release code!
- [x] Optimize gradient storing / checkpointing. Memory requirements, training times reduced by ~55%
- [x] Release data sets
- [ ] Release pre-trained embeddings
- [ ] Add Stable Diffusion support

## Setup

Our code builds on, and shares requirements with [Latent Diffusion Models (LDM)](https://github.com/CompVis/latent-diffusion). To set up their environment, please run:

```
conda env create -f environment.yaml
conda activate ldm
```

You will also need the official LDM text-to-image checkpoint, available through the [LDM project page](https://github.com/CompVis/latent-diffusion). 

Currently, the model can be downloaded by running:

```
mkdir -p models/ldm/text2img-large/
wget -O models/ldm/text2img-large/model.ckpt https://ommer-lab.com/files/latent-diffusion/nitro/txt2img-f8-large/model.ckpt
```

## Usage

### Inversion

To invert an image set, run:

```
python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml 
               -t 
               --actual_resume /path/to/pretrained/model.ckpt 
               -n <run_name> 
               --gpus 0, 
               --data_root /path/to/directory/with/images
               --init_word <initialization_word>
```
`python main.py --base configs/latent-diffusion/txt2img-1p4B-finetune.yaml -t --actual_resume /local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt -n "apple_cifar100_500_train" --gpus 0, --data_root ../label_0_images_3/ --init_word photo `

where the initialization word should be a single-token rough description of the object (e.g., 'toy', 'painting', 'sculpture'). If the input is comprised of more than a single token, you will be prompted to replace it.

Please note that `init_word` is *not* the placeholder string that will later represent the concept. It is only used as a beggining point for the optimization scheme.

In the paper, we use 5k training iterations. However, some concepts (particularly styles) can converge much faster.

To run on multiple GPUs, provide a comma-delimited list of GPU indices to the --gpus argument (e.g., ``--gpus 0,3,7,8``)

Embeddings and output images will be saved in the log directory.

See `configs/latent-diffusion/txt2img-1p4B-finetune.yaml` for more options, such as: changing the placeholder string which denotes the concept (defaults to "*"), changing the maximal number of training iterations, changing how often checkpoints are saved and more.

**Important** All training set images should be upright. If you are using phone captured images, check the inputs_gs*.jpg files in the output image directory and make sure they are oriented correctly. Many phones capture images with a 90 degree rotation and denote this in the image metadata. Windows parses these correctly, but PIL does not. Hence you will need to correct them manually (e.g. by pasting them into paint and re-saving) or wait until we add metadata parsing.

### Generation

To generate new images of the learned concept, run:
```
python scripts/txt2img.py --ddim_eta 0.0 
                          --n_samples 8 
                          --n_iter 2 
                          --scale 10.0 
                          --ddim_steps 50 
                          --embedding_path /path/to/logs/trained_model/checkpoints/embeddings_gs-5049.pt 
                          --ckpt_path /path/to/pretrained/model.ckpt 
                          --prompt "a photo of *"
```
` python scripts/txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 2 --scale 10.0 --ddim_steps 50 --embedding_path /home/chowdhury.150/Projects/cl_dm/textual_inversion/logs/label_0_images_32023-10-26T02-00-02_apple_cifar100_500_train/checkpoints/embeddings_gs-6099.pt --ckpt_path /local/scratch/chowdhury.150/ldm/text2img-large/model.ckpt --prompt "a photo of *" `
where * is the placeholder string used during inversion.

` python merge_embeddings.py --manager_ckpts /home/chowdhury.150/Projects/cl_dm/textual_inversion/logs/label_0_images2023-10-26T03-06-02_apple_cifar100_500_train_all/checkpoints/embeddings_gs-5499.pt /home/chowdhury.150/Projects/cl_dm/textual_inversion/logs/label_0_images2023-10-26T03-06-02_apple_cifar100_500_train_all/checkpoints/embeddings_gs-5999.pt --output_path /home/chowdhury.150/Projects/cl_dm/textual_inversion/logs/label_0_images2023-10-26T03-06-02_apple_cifar100_500_train_all/checkpoints/merged_embeddings.pt `
### Merging Checkpoints

LDM embedding checkpoints can be merged into a single file by running:

```
python merge_embeddings.py 
--manager_ckpts /path/to/first/embedding.pt /path/to/second/embedding.pt [...]
--output_path /path/to/output/embedding.pt
```

For SD embeddings, simply add the flag: `-sd` or `--stable_diffusion`.

If the checkpoints contain conflicting placeholder strings, you will be prompted to select new placeholders. The merged checkpoint can later be used to prompt multiple concepts at once ("A photo of * in the style of @").

### Pretrained Models / Data

Datasets which appear in the paper are being uploaded [here](https://drive.google.com/drive/folders/1d2UXkX0GWM-4qUwThjNhFIPP7S6WUbQJ). Some sets are unavailable due to image ownership. We will upload more as we recieve permissions to do so.

Pretained models coming soon.

## Stable Diffusion

Stable Diffusion support is a work in progress and will be completed soon™.

## Tips and Tricks
- Adding "a photo of" to the prompt usually results in better target consistency.
- Results can be seed sensititve. If you're unsatisfied with the model, try re-inverting with a new seed (by adding `--seed <#>` to the prompt).


## Citation

If you make use of our work, please cite our paper:

```
@misc{gal2022textual,
      doi = {10.48550/ARXIV.2208.01618},
      url = {https://arxiv.org/abs/2208.01618},
      author = {Gal, Rinon and Alaluf, Yuval and Atzmon, Yuval and Patashnik, Or and Bermano, Amit H. and Chechik, Gal and Cohen-Or, Daniel},
      title = {An Image is Worth One Word: Personalizing Text-to-Image Generation using Textual Inversion},
      publisher = {arXiv},
      year = {2022},
      primaryClass={cs.CV}
}
```

## Results
Here are some sample results. Please visit our [project page](https://textual-inversion.github.io/) or read our paper for more!

![](img/teaser.jpg)

![](img/samples.jpg)

![](img/style.jpg)