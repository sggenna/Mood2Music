# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.17.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import kagglehub
import os

# %%
# load dataset from kaggle
path = kagglehub.dataset_download("divyansh22/summer-olympics-medals")

print(os.getcwd())

# %%
df = pd.read_csv("olympics_data.csv", encoding="latin1")
df.head(10)
