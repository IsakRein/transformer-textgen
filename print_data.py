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

val_loss_lstm_char_128_hidden = torch.load('model_data/lstm_char_128_hidden/val_losses.pth')[:-1]
val_loss_lstm_char_256_hidden = torch.load('model_data/lstm_char_256_hidden/val_losses.pth')[:-1]
val_loss_lstm_char_512_hidden = torch.load('model_data/lstm_char_512_hidden/val_losses.pth')[:-1]
val_loss_lstm_char_128_hidden_two_layer = torch.load('model_data/lstm_char_128_hidden_two_layer/val_losses.pth')[:-1]
val_loss_lstm_char_256_hidden_two_layer = torch.load('model_data/lstm_char_256_hidden_two_layer/val_losses.pth')[:-1]
val_loss_lstm_char_512_hidden_two_layer = torch.load('model_data/lstm_char_512_hidden_two_layer/val_losses.pth')[:-1]

val_loss_lstm_char_lr_01_bz_8 = torch.load('model_data/lstm_char_lr_01_bz_8/val_losses.pth')[:-1]
val_loss_lstm_char_lr_001_bz_8 = torch.load('model_data/lstm_char_lr_001_bz_8/val_losses.pth')[:-1]
val_loss_lstm_char_lr_0001_bz_8 = torch.load('model_data/lstm_char_lr_0001_bz_8/val_losses.pth')[:-1]
val_loss_lstm_char_lr_01_bz_16 = torch.load('model_data/lstm_char_lr_01_bz_16/val_losses.pth')[:-1]
val_loss_lstm_char_lr_001_bz_16 = torch.load('model_data/lstm_char_lr_001_bz_16/val_losses.pth')[:-1]
val_loss_lstm_char_lr_0001_bz_16 = torch.load('model_data/lstm_char_lr_0001_bz_16/val_losses.pth')[:-1]

val_loss_rnn_char_baseline = torch.load('model_data/rnn_char_baseline/val_losses.pth')[:26]


x_values = range(0,len(val_loss_lstm_char_128_hidden) * 1000, 1000)
plt.title("Validation loss vs update iteration")
plt.plot(x_values, val_loss_lstm_char_128_hidden, label="128x1")
plt.plot(x_values, val_loss_lstm_char_256_hidden, label="256x1")
plt.plot(x_values, val_loss_lstm_char_512_hidden, label="512x1")
plt.plot(x_values, val_loss_lstm_char_128_hidden_two_layer, label="128x2")
plt.plot(x_values, val_loss_lstm_char_256_hidden_two_layer, label="256x2")
plt.plot(x_values, val_loss_lstm_char_512_hidden_two_layer, label="512x2")
plt.plot(x_values, val_loss_rnn_char_baseline, label="rnn")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.savefig("images/layer_size")
plt.clf()


x_values = range(0,len(val_loss_lstm_char_128_hidden) * 1000, 1000)
plt.title("Validation loss vs update iteration")
plt.plot(x_values, val_loss_lstm_char_lr_01_bz_8, label="lr=0.1 bz=8")
plt.plot(x_values, val_loss_lstm_char_lr_001_bz_8, label="lr=0.01 bz=8")
plt.plot(x_values, val_loss_lstm_char_lr_0001_bz_8, label="lr=0.001 bz=8")
plt.plot(x_values, val_loss_lstm_char_lr_01_bz_16, label="lr=0.1 bz=16")
plt.plot(x_values, val_loss_lstm_char_lr_001_bz_16, label="lr=0.01 bz=16")
plt.plot(x_values, val_loss_lstm_char_lr_0001_bz_16, label="lr=0.001 bz=16")
plt.plot(x_values, val_loss_rnn_char_baseline, label="rnn")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.savefig("images/lr_bz")
plt.clf()

