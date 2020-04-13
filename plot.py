import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import matplotlib as mpl

mpl.rc('font', family='Garamond')
counter_without_history = np.asarray(
    [0.10546740000000021,
     0.3371098999999993,
     2.2849944000000066,
     9.955656600000008,
     20.602998800000005,
     144.46911600000004])
counter_history = np.asarray(
    [0.09691070000000046,
     0.3359525999999997,
     2.549793599999999,
     8.310589899999988,
     25.590754599999947,
     100.56323509999987])

counter_fast = np.asarray(
    [0.07219020000000032,
     0.20095899999999967,
     0.6836049000000011,
     3.622343800000003,
     17.62905929999999,
     76.94946349999991]
)
counter_multi = np.asarray(
    [13.546170299999996,
     13.0215605,
     14.50015730000003,
     16.221036100000006,
     19.790862800000028,
     91.27405599999996])
counter_half_fast = np.asarray(
    [0.08201779999999737,
     0.32945390000000074,
     1.3495251000000046,
     5.999991900000008,
     18.58495070000003,
     59.92163530000012])
inter_history = (interpolate.CubicSpline(np.linspace(0, 6, 6), counter_history))(np.linspace(0, 6, 600))
inter_without_history = (interpolate.CubicSpline(np.linspace(0, 6, 6), counter_without_history))(np.linspace(0, 6, 600))
inter_fast = (interpolate.CubicSpline(np.linspace(0, 6, 6), counter_fast))(np.linspace(0, 6, 600))
inter_multi = (interpolate.CubicSpline(np.linspace(0, 6, 6), counter_multi))(np.linspace(0, 6, 600))
inter_half_fast = (interpolate.CubicSpline(np.linspace(0, 6, 6), counter_half_fast))(np.linspace(0, 6, 600))

plt.figure(figsize=(12, 9))
# plt.plot(np.linspace(0, 6, 6), counter_history, label="With History", linewidth=2.5, alpha=0.7)
plt.plot(np.linspace(0, 6, 600), inter_history, label="With History Interpolation", linewidth=2.5, alpha=0.7)
# plt.plot(np.linspace(0, 6, 6), counter_without_history, label="Without History", linewidth=2.5, alpha=0.7)
plt.plot(np.linspace(0, 6, 600), inter_without_history, label="Without History Interpolation", linewidth=2.5, alpha=0.7)
# plt.plot(np.linspace(0, 6, 6), counter_fast, label="Fast", linewidth=2.5, alpha=0.7)
plt.plot(np.linspace(0, 6, 600), inter_fast, label="Fast Interpolation", linewidth=2.5, alpha=0.7)
# plt.plot(np.linspace(0, 6, 6), counter_multi, label="MultiCore", linewidth=2.5, alpha=0.7)
plt.plot(np.linspace(0, 6, 600), inter_multi, label="MultiCore Interpolation", linewidth=2.5, alpha=0.7)
# plt.plot(np.linspace(0, 6, 6), counter_half_fast, label="Half Fast", linewidth=2.5, alpha=0.7)
plt.plot(np.linspace(0, 6, 600), inter_half_fast, label="Half Fast Interpolation", linewidth=2.5, alpha=0.7)
plt.legend()
# argmax = np.argmax(inter) / 100
# max = np.max(inter)
# plt.scatter([argmax, ], [max, ], 100, color='red', alpha=0.7)
# plt.plot([argmax, argmax], [0, max], color='red', linewidth=1.5, linestyle="--", alpha=0.7)
# plt.plot([0, argmax], [max, max], color='red', linewidth=1.5, linestyle="--", alpha=0.7)
ax = plt.gca()
ax.spines["right"].set_color("none")
ax.spines["top"].set_color("none")

# plt.annotate(r'Longest Total Time',
#              xy=(argmax, max), xycoords='data',
#              xytext=(50, -50), textcoords='offset points', fontsize=16,
#              arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))

plt.xlabel("Search Depth", fontsize=16)
plt.ylabel("Total Time (s)", fontsize=16)
plt.legend(loc='upper right', prop={"size": "14"})
plt.yticks(np.linspace(0, 150, 16, endpoint=True))
plt.xticks(np.linspace(0, 10, 11, endpoint=True))
plt.grid()
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor="none", alpha=0.5))
