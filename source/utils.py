import argparse
import os
import json
import gzip
import pickle
import time
import numpy as np
import pandas as pd

import sys
sys.path.append("..")

#import warnings
#warnings.filterwarnings('ignore')

#import matplotlib.pyplot as plt
from datetime import datetime
from tqdm import tqdm

import nltk
#nltk.download('poplular')
from nltk.tokenize import word_tokenize, sent_tokenize
#from nltk.corpus import stopwords
#from nltk.stem.porter import PorterStemmer

from transformers import BertTokenizer, BertModel
import torch


