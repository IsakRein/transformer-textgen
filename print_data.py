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

print(torch.load('model_data/rnn_char_example/val_losses.pth'))
print(torch.load('model_data/lstm_char_example/val_losses.pth'))

