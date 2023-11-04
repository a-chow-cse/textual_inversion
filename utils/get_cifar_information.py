from torchvision import datasets, transforms

cifar100_train = datasets.CIFAR100(root='./data', train=True, download=True)
print("Classes: ",cifar100_train.classes)