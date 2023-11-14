import os
import random
from torchvision import datasets, transforms
from PIL import Image

cifar100_train = datasets.CIFAR100(root='./data', train=True, download=False)

class_num = 78

save_folder = f'./cifar_class_{class_num}'
os.makedirs(save_folder, exist_ok=True)
class_indices = [i for i in range(len(cifar100_train)) if cifar100_train.targets[i] == class_num]

random_indices = random.sample(class_indices, 250)
transform = transforms.Compose([transforms.ToPILImage()])

for idx in random_indices:
    image, label = cifar100_train[idx]
    image.save(os.path.join(save_folder, f'image_{idx}.png'))

print(f"Random 100 images from class {class_num} saved in {save_folder}")