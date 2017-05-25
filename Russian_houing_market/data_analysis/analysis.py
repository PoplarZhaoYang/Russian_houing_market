
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import model_selection, proprocessing
import xgboost as xgb
color = sns.color_palette()

%matplotlib inline #IPython magic函数,图像嵌入notebook中

pd.options.mode.chained_assignment = None #配置链式赋值,不警告
pd.set_option('display.max_columns', 500) #最大显示列数设置




