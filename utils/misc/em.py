"""
import torch
torch.cuda.empty_cache()
import sys

a= torch.load("./logs//current_image02023-10-31T02-50-30_apple_30_cluster_0/checkpoints/embeddings_gs-1499.pt")

print(a.keys())
print(a['string_to_token'].keys())
print(a['string_to_token']['*'].element_size()*a['string_to_token']['*'].nelement())
print(sys.getsizeof(a['string_to_token']))
print(sys.getsizeof(a))

print(a['string_to_param'].keys())
print(a['string_to_param']['*'])
print(sys.getsizeof(a['string_to_param']['*']))
print(a['string_to_param']['*'].element_size()*a['string_to_param']['*'].nelement())
"""
word="skunk"
from transformers import CLIPTokenizer
tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
tokens = tokenizer.tokenize(word)
print(tokens)

from transformers import BertTokenizerFast
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")
tokens = tokenizer(word, truncation=True, max_length=77, return_length=True)
print(tokens)

#lawn_mower == vehicle
#hamster= mammal
#shrew=mammal
#skunk=mammal