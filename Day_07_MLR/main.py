import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import fetch_california_housing
df = fetch_california_housing()
print(df.DESCR)
