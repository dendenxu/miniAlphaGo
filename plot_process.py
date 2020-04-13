import matplotlib.pyplot as plt
import numpy as np
from scipy import interpolate
import matplotlib as mpl

mpl.rc('font', family='Garamond')
counter_black = np.asarray(
    [  # 0.30591330000000005,
        # 0.44735809999999976,
        # 1.8774962,
        # 4.0305789,
        # 2.9003426999999995,
        # 3.4301890999999998,
        # 4.462001700000002,
        # 5.012544600000005,
        # 6.530969200000001,
        # 14.476920499999999,
        # 8.077299199999999,
        # 10.4651784,
        # 11.220239300000003,
        # 22.79562150000001,
        # 17.256274400000024,
        # 29.028554799999995,
        # 29.6418391,
        # 25.301124000000016,
        # 24.508708400000017,
        # 5.560382799999957,
        # 1.9159836999999698,
        # 0.9684512000000041,
        # 0.3543061000000307,
        # 0.22160230000002912,
        # 0.10739430000000993,
        # 0.03452529999998433,
        # 0.0052319000000125016,
        0.0017914999999675274,
        0.4600400000000011,
        1.7333701000000001,
        3.7559451,
        2.8742006000000018,
        3.604536799999998,
        4.745733700000002,
        5.2534180999999975,
        6.7274569,
        15.042072700000006,
        8.29106680000001,
        6.257546500000004,
        8.611326199999993,
        38.08395590000001,
        10.949538700000005,
        7.834566200000012,
        13.82002270000001,
        8.855895300000014,
        2.486822200000006,
        0.8097903999999971,
        0.6526192000000037,
        0.3101494999999943,
        0.2763114999999914,
        0.17363869999999793,
        0.16929939999999988,
        0.03682889999998906,
        0.023178799999982402,
        0.001932699999997567,
    ])
counter_white = np.asarray(
    [  # 0.36037050000000015,
        #  1.1919163,
        #  2.1388210999999995,
        #  2.388910899999999,
        #  5.208125299999999,
        #  5.5536753999999995,
        #  5.796886100000002,
        #  3.846306299999995,
        #  4.037858499999999,
        #  2.2909645999999952,
        #  2.4705905,
        #  2.7985618000000017,
        #  3.500963999999996,
        #  3.1381733999999994,
        #  3.2720701999999733,
        #  2.5106186999999807,
        #  2.221976000000012,
        #  2.983878500000003,
        #  1.502042700000004,
        #  1.5094435999999973,
        #  0.8131455000000187,
        #  1.0436693000000332,
        #  0.39271649999994906,
        #  0.0007204999999999018,
        #  0.27751429999995025,
        #  0.12524530000001732,
        #  0.045250000000010004,
        #  0.0006169999999769971,
        #  0.0026899999999727697,
        #  0.0010300999999799387,
        0.3479342000000001,
        1.2180229000000011,
        2.152190299999999,
        2.4658362999999994,
        6.6617745999999975,
        7.257855299999996,
        7.894012199999999,
        4.5733557000000005,
        5.803474400000013,
        5.259165799999991,
        4.466288800000001,
        6.334664400000008,
        4.610168200000004,
        6.679588699999982,
        3.821889700000014,
        0.0009660999999994146,
        4.7896338000000185,
        2.828656499999994,
        1.1874569000000008,
        1.141919999999999,
        0.7351628000000119,
        0.0007659999999987122,
        0.3793456999999876,
        0.38169529999998986,
        0.23529460000000313,
        0.20510789999997314,
        0.04508960000001139,
        0.0033376000000089334,
        0.0006200999999919077,
        0.0010665000000074087,
    ])
black_size = counter_black.shape[0]
white_size = counter_white.shape[0]
inter_black = (interpolate.CubicSpline(np.linspace(0, black_size, black_size), counter_black))(
    np.linspace(0, black_size, black_size * 100))
inter_white = (interpolate.CubicSpline(np.linspace(0, white_size, white_size), counter_white))(
    np.linspace(0, white_size, white_size * 100))
plt.figure(figsize=(12, 9))
plt.plot(np.linspace(0, black_size, black_size * 100), inter_black, label="Simple AI Interpolation", linewidth=2.5,
         alpha=0.7)
plt.plot(np.linspace(0, white_size, white_size * 100), inter_white, label="Ours Interpolation", linewidth=2.5,
         alpha=0.7)
plt.plot(np.linspace(0, black_size, black_size), counter_black, label="Simple AI", linewidth=2.5,
         alpha=0.7)
plt.plot(np.linspace(0, white_size, white_size), counter_white, label="Ours", linewidth=2.5,
         alpha=0.7)
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
plt.xlabel("i th Move", fontsize=16)
plt.ylabel("Move Time", fontsize=16)
plt.legend(loc='upper right', prop={"size": "14"})
# plt.yticks(np.linspace(0, 150, 16, endpoint=True))
# plt.xticks(np.linspace(0, 10, 11, endpoint=True))
plt.grid()
ax.xaxis.set_ticks_position("bottom")
ax.spines["bottom"].set_position(("data", 0))
ax.yaxis.set_ticks_position("left")
ax.spines["left"].set_position(("data", 0))
for label in ax.get_xticklabels() + ax.get_yticklabels():
    label.set_fontsize(16)
    label.set_bbox(dict(facecolor='white', edgecolor="none", alpha=0.5))
