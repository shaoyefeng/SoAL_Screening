import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
import pickle

df = pd.read_pickle(r"E:\LC_screening\all\center\_driver\ALL.pickle", compression='gzip')[['geno', 'cir_per_second']]
grouped_df = df.groupby('geno', sort=False)
meandf = grouped_df.mean().add_prefix('mean_')
for i in range(0, len(meandf)):
    if i % 3 == 1:
        meandf.iloc[i] = meandf.iloc[i] / meandf.iloc[i-1] - 1
    elif i % 3 == 2:
        meandf.iloc[i] = meandf.iloc[i] / meandf.iloc[i-2] - 1
    else:
        pass
control = np.arange(0, len(meandf), 3)
meandf = meandf.drop(meandf.index[control])


fig, ax = plt.subplots(figsize=(20, 10))
plt.subplots_adjust(top=0.99, bottom=0.14, left=0.05, right=0.99)

xlabels = meandf.index.tolist()
a = np.arange(0, 66)
b = np.arange(2, 66, 3)
x = np.setdiff1d(a, b)
y = meandf['mean_cir_per_second']
c = [a for a in "rb" * 22]

# sns.barplot(x=x, y=y, palette=c, alpha=0.5)
plt.bar(x=x, height=y, color=c, alpha=0.5)
plt.xticks(x, xlabels, fontsize=7, rotation=90)

ax.set_ylabel('mean_cir_per_second (normalized_difference)')
ax.set_xlim(-1, 66)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# plt.show()
plt.savefig('E:/all/center/normalized_difference_center.png')