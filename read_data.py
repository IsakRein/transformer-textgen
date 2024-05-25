import torch.nn as nn
import torch.nn.functional as F
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import torch.optim as optim
import pickle
import sys
import json
from gensim.models import Word2Vec
import os
import re
from torcheval.metrics.text import Perplexity
from spellchecker import SpellChecker

val_loss_transformer_char_test_embed_128 = torch.load(
    'model_data/transformer_char_test_embed_128/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_256 = torch.load(
    'model_data/transformer_char_test_embed_256/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512 = torch.load(
    'model_data/transformer_char_test_embed_512/val_losses.pth')[:11]
# val_loss_transformer_char_test_embed_512_8_heads = torch.load(
#     'model_data/transformer_char_test_embed_512_8_heads/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512_seq_128 = torch.load(
    'model_data/transformer_char_test_embed_512_seq_128/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512_seq_128_4_layers = torch.load(
    'model_data/transformer_char_test_embed_512_seq_128_4_layers/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512_seq_128_10_layers = torch.load(
    'model_data/transformer_char_test_embed_512_seq_128_10_layers/val_losses.pth')[:11]


print(val_loss_transformer_char_test_embed_128[10])
print(val_loss_transformer_char_test_embed_256[10])
print(val_loss_transformer_char_test_embed_512[10])
# print(val_loss_transformer_char_test_embed_512_8_heads[10])
print(val_loss_transformer_char_test_embed_512_seq_128[10])
print(val_loss_transformer_char_test_embed_512_seq_128_4_layers[10])
print(val_loss_transformer_char_test_embed_512_seq_128_10_layers[10])

print(val_loss_transformer_char_test_embed_512_seq_128)
print(val_loss_transformer_char_test_embed_512_seq_128_10_layers)
