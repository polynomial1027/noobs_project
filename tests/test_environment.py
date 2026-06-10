import sys
import numpy
import pandas
import scipy
import sklearn
import statsmodels
import matplotlib
import seaborn
import cv2
import skimage
import pygame
import torch
from PIL import Image

print("Python:", sys.version)
print("numpy:", numpy.__version__)
print("pandas:", pandas.__version__)
print("scipy:", scipy.__version__)
print("scikit-learn:", sklearn.__version__)
print("statsmodels:", statsmodels.__version__)
print("opencv:", cv2.__version__)
print("scikit-image:", skimage.__version__)
print("pygame:", pygame.version.ver)
print("torch:", torch.__version__)
print("torch cuda available:", torch.cuda.is_available())
print("Environment check passed.")