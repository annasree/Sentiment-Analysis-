# Sentiment-Analysis-
This project try to do the sentiment analysis of Amazon customer review using BERT-Style LLM. Data is taken from kaggle
# import necessory modules
import torch
import string
import numpy as np
import pandas as pd
import torch.nn as nn
from google.colab import drive
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from transformers import BertTokenizer
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from transformers import BertForSequenceClassification
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
