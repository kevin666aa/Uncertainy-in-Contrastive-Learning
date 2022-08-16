import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt


y = np.arange(1, 0, -0.07)
fig = plt.figure()
plt.plot(y, linewidth=4)
# plt.xlabel('intensity', fontsize = 20)
# plt.ylabel('aaa', fontsize = 20)
# plt.ylim(ylimits[i])
plt.xticks(fontsize= 15)
plt.yticks(fontsize= 15)
plt.subplots_adjust(top = 0.97, bottom = 0.08, right = 0.98, left = 0.1, 
            hspace = 0, wspace = 0)
plt.margins(0,0)
plt.savefig(f'/vol/bitbucket/yw2621/SimCLR/figures/test.pdf')