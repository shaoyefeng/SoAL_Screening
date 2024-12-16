from fpt_analysis import plot_line_by_info
import matplotlib.pyplot as plt
import pickle
import os

datapath = r'E:\LC21\center\_etho'

plt.figure()
ax = plt.gca()
ax.set_xlabel('Time(s)')
ax.set_ylabel('Circling per minute')

color = ["r", "g", "b"]
i = 0
for a in os.listdir(datapath):
    if a.endswith("center_dist.pickle"):
        b = os.path.join(datapath, a)
        c = color[i]
        df = pickle.load(open(b, 'rb'))
        fig_info = df[2]
        fig_info[-3] = c
        plot_line_by_info(ax, fig_info, label=a.split("_")[0])
        ax.legend()
        i += 1
# plt.show()
figpath = os.path.join(datapath, 'center_circling.png')
plt.savefig(figpath)

# df = pickle.load(open(r"D:\Lab\Project\Screening\LC25\center\_etho\CSXTrp31_etho_center_dist.pickle",'rb'))
# fig_info = df[2]
# fig_info[-3] = 'red'
# plot_line_by_info(ax, fig_info, label='CS X Trp 31')
# ax.legend()
#
# df = pickle.load(open(r"D:\Lab\Project\Screening\LC25\center\_etho\LC25XTrp31_etho_center_dist.pickle",'rb'))
# fig_info = df[2]
# fig_info[-3] = 'green'
# plot_line_by_info(ax, fig_info, label='LC25 X Trp 31')
# ax.legend()
#
# df = pickle.load(open(r"D:\Lab\Project\Screening\LC25\center\_etho\LC25XShi31_etho_center_dist.pickle",'rb'))
# fig_info = df[2]
# fig_info[-3] = 'blue'
# plot_line_by_info(ax, fig_info, label='LC25 X Shi 31')
# ax.legend()



