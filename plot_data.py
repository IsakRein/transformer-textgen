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
val_perplexity_lstm_char_128_hidden = torch.load('model_data/lstm_char_128_hidden/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_256_hidden = torch.load('model_data/lstm_char_256_hidden/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_512_hidden = torch.load('model_data/lstm_char_512_hidden/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_128_hidden_two_layer = torch.load('model_data/lstm_char_128_hidden_two_layer/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_256_hidden_two_layer = torch.load('model_data/lstm_char_256_hidden_two_layer/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_512_hidden_two_layer = torch.load('model_data/lstm_char_512_hidden_two_layer/val_perplexity.pth')[:-1]

val_loss_lstm_char_lr_01_bz_8 = torch.load('model_data/lstm_char_lr_01_bz_8/val_losses.pth')[:-1]
val_loss_lstm_char_lr_001_bz_8 = torch.load('model_data/lstm_char_lr_001_bz_8/val_losses.pth')[:-1]
val_loss_lstm_char_lr_0001_bz_8 = torch.load('model_data/lstm_char_lr_0001_bz_8/val_losses.pth')[:-1]
val_loss_lstm_char_lr_01_bz_16 = torch.load('model_data/lstm_char_lr_01_bz_16/val_losses.pth')[:-1]
val_loss_lstm_char_lr_001_bz_16 = torch.load('model_data/lstm_char_lr_001_bz_16/val_losses.pth')[:-1]
val_loss_lstm_char_lr_0001_bz_16 = torch.load('model_data/lstm_char_lr_0001_bz_16/val_losses.pth')[:-1]
val_perplexity_lstm_char_lr_01_bz_8 = torch.load('model_data/lstm_char_lr_01_bz_8/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_lr_001_bz_8 = torch.load('model_data/lstm_char_lr_001_bz_8/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_lr_0001_bz_8 = torch.load('model_data/lstm_char_lr_0001_bz_8/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_lr_01_bz_16 = torch.load('model_data/lstm_char_lr_01_bz_16/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_lr_001_bz_16 = torch.load('model_data/lstm_char_lr_001_bz_16/val_perplexity.pth')[:-1]
val_perplexity_lstm_char_lr_0001_bz_16 = torch.load('model_data/lstm_char_lr_0001_bz_16/val_perplexity.pth')[:-1]


val_loss_rnn_char_baseline = torch.load('model_data/rnn_char_baseline/val_losses.pth')[:26]
val_perplexity_rnn_char_baseline = torch.load('model_data/rnn_char_baseline/val_perplexity.pth')[:26]



val_loss_transformer_char_test_embed_128 = torch.load('model_data/transformer_char_test_embed_128/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_256 = torch.load('model_data/transformer_char_test_embed_256/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512 = torch.load('model_data/transformer_char_test_embed_512/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512_seq_32 = torch.load('model_data/transformer_char_test_embed_512_seq_32/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512_seq_128 = torch.load('model_data/transformer_char_test_embed_512_seq_128/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512_seq_128_4_layers = torch.load('model_data/transformer_char_test_embed_512_seq_128_4_layers/val_losses.pth')[:11]
val_loss_transformer_char_test_embed_512_seq_128_10_layers = torch.load('model_data/transformer_char_test_embed_512_seq_128_10_layers/val_losses.pth')[:11]

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
plt.savefig("images/lstm_layer_size_loss")
plt.clf()

x_values = range(0,len(val_loss_lstm_char_128_hidden) * 1000, 1000)
plt.title("Validation perplexity vs update iteration")
plt.plot(x_values, val_perplexity_lstm_char_128_hidden, label="128x1")
plt.plot(x_values, val_perplexity_lstm_char_256_hidden, label="256x1")
plt.plot(x_values, val_perplexity_lstm_char_512_hidden, label="512x1")
plt.plot(x_values, val_perplexity_lstm_char_128_hidden_two_layer, label="128x2")
plt.plot(x_values, val_perplexity_lstm_char_256_hidden_two_layer, label="256x2")
plt.plot(x_values, val_perplexity_lstm_char_512_hidden_two_layer, label="512x2")
plt.plot(x_values, val_perplexity_rnn_char_baseline, label="rnn")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Perplexity")
plt.legend()
plt.savefig("images/lstm_layer_size_perplexity")
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
plt.savefig("images/lstm_lr_bz_loss")
plt.clf()

x_values = range(0,len(val_loss_lstm_char_128_hidden) * 1000, 1000)
plt.title("Validation perplexity vs update iteration")
plt.plot(x_values, val_perplexity_lstm_char_lr_01_bz_8, label="lr=0.1 bz=8")
plt.plot(x_values, val_perplexity_lstm_char_lr_001_bz_8, label="lr=0.01 bz=8")
plt.plot(x_values, val_perplexity_lstm_char_lr_0001_bz_8, label="lr=0.001 bz=8")
plt.plot(x_values, val_perplexity_lstm_char_lr_01_bz_16, label="lr=0.1 bz=16")
plt.plot(x_values, val_perplexity_lstm_char_lr_001_bz_16, label="lr=0.01 bz=16")
plt.plot(x_values, val_perplexity_lstm_char_lr_0001_bz_16, label="lr=0.001 bz=16")
plt.plot(x_values, val_perplexity_rnn_char_baseline, label="rnn")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("Perplexity")
plt.legend()
plt.savefig("images/lstm_lr_bz_perplexity")
plt.clf()



x_values = range(0,len(val_loss_transformer_char_test_embed_128) * 1000, 1000)
plt.title("Validation loss vs update iteration")
plt.plot(x_values, val_loss_transformer_char_test_embed_128, label="n_embed = 128")
plt.plot(x_values, val_loss_transformer_char_test_embed_256, label="n_embed = 256")
plt.plot(x_values, val_loss_transformer_char_test_embed_512, label="n_embed = 512")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.savefig("images/transformer_n_embed")
plt.clf()

x_values = range(0,len(val_loss_transformer_char_test_embed_128) * 1000, 1000)
plt.title("Validation loss vs update iteration")
plt.plot(x_values, val_loss_transformer_char_test_embed_512_seq_32, label="n_embed = 512, seq_length=32")
plt.plot(x_values, val_loss_transformer_char_test_embed_512, label="n_embed = 512, seq_length=64")
plt.plot(x_values, val_loss_transformer_char_test_embed_512_seq_128, label="n_embed = 512, seq_length=128")

plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.savefig("images/transformer_n_embed_vs_seq")
plt.clf()


x_values = range(0,len(val_loss_transformer_char_test_embed_128) * 1000, 1000)
plt.title("Validation loss vs update iteration")
plt.plot(x_values, val_loss_transformer_char_test_embed_512_seq_128_4_layers, label="n_embed=512,seq_length=128,layers=4")
plt.plot(x_values, val_loss_transformer_char_test_embed_512_seq_128, label="n_embed = 512, seq_length=128, layers = 6")
plt.plot(x_values, val_loss_transformer_char_test_embed_512_seq_128_10_layers, label="n_embed=512,seq_length=128,layers=10")
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel("loss")
plt.legend()
plt.savefig("images/transformer_layer_comp")
plt.clf()
