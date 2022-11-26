import os
import numpy as np
import ipdb as pdb
import math
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties  # 导入FontProperties
from matplotlib.patches import  ConnectionPatch



C_max = 100  #客户数
T_max = 1600
Q_max = 2000
T_timeunit = np.arange(T_max)

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float, axis=0)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def output_avg(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs_0 = []
    avg_rs_1 = []
    avg_rs_2 = []
    avg_rs_3 = []
    avg_rs_4 = []
    avg_rs_5 = []
    avg_rs_6 = []
    avg_rs_7 = []
    avg_rs_8 = []
    for name in fileList:

        path = dir_path + name
        res = np.load(path)
        temp_rs_0 = np.array(res['arr_0'])
        temp_rs_1 = np.array(res['arr_1'])
        temp_rs_2 = np.array(res['arr_2'])
        temp_rs_3 = np.array(res['arr_3'])
        temp_rs_4 = np.array(res['arr_4'])
        temp_rs_5 = np.array(res['arr_5'])
        temp_rs_6 = np.array(res['arr_6'])
        temp_rs_7 = np.array(res['arr_7'])
        temp_rs_8 = np.array(res['arr_8'])
        avg_rs_0.append(temp_rs_0)
        avg_rs_1.append(temp_rs_1)
        avg_rs_2.append(temp_rs_2)
        avg_rs_3.append(temp_rs_3)
        avg_rs_4.append(temp_rs_4)
        avg_rs_5.append(temp_rs_5)
        avg_rs_6.append(temp_rs_6)
        avg_rs_7.append(temp_rs_7)
        avg_rs_8.append(temp_rs_8)

    avg_rs_0 = moving_average(np.mean(avg_rs_0, axis=0, keepdims=True)[0],10)
    avg_rs_1 = np.mean(avg_rs_1, axis=0, keepdims=True)[0]
    avg_rs_2 = np.mean(avg_rs_2, axis=0, keepdims=True)[0]
    avg_rs_3 = np.mean(avg_rs_3, axis=0, keepdims=True)[0]
    avg_rs_4 = np.mean(avg_rs_4, axis=0, keepdims=True)[0]
    avg_rs_5 = np.mean(avg_rs_5, axis=0, keepdims=True)[0]
    avg_rs_6 = np.mean(avg_rs_6, axis=0, keepdims=True)[0]
    avg_rs_7 = np.mean(avg_rs_7, axis=0, keepdims=True)[0]
    avg_rs_8 = np.mean(avg_rs_8, axis=0, keepdims=True)[0]
    return avg_rs_0, avg_rs_1, avg_rs_2, avg_rs_3, avg_rs_4, avg_rs_5, avg_rs_6, avg_rs_7, avg_rs_8
    
res_path = 'Data/'
CN_accuracy,_,_,_,_,_,_,_,_ = output_avg(res_path)
plt.figure(1)
plt.plot(np.arange(len(CN_accuracy)), CN_accuracy, color='r', label='CN_accuracy')
plt.xlabel('Data amount', fontsize=40)
plt.ylabel('Expected accuracy (%)', fontsize=40)
plt.tick_params(labelsize=35)
plt.ylim(0, 100)
plt.xlim(0, 2000)
plt.grid(True)


_, T_queue_proposed,_,_,_,_,_,_,_ = output_avg(res_path)
_,_, T_queue_random,_,_,_,_,_,_ = output_avg(res_path)
_,_,_, T_compare_queue1,_,_,_,_,_ = output_avg(res_path)
_,_,_,_, T_compare_queue2,_,_,_,_ = output_avg(res_path)
plt.figure(2)
line5 = plt.axhline(y=Q_max, color='k', linestyle='--', linewidth=3.0)
plt.ylim(0, Q_max * 5)
plt.xlim(0, T_max)

# print(T_queue_proposed)
line1, = plt.plot(np.arange(len(T_queue_proposed)), T_queue_proposed[:], label='Proposed')
line2, = plt.plot(np.arange(len(T_queue_random)), T_queue_random[:], label='Random')
line3, = plt.plot(np.arange(len(T_compare_queue1)), T_compare_queue1[:], label='Maximum')
line4, = plt.plot(np.arange(len(T_compare_queue2)), T_compare_queue2[:], label='Static')

plt.setp(line1, color='magenta', linewidth=3.0)
plt.setp(line2, color='b', linewidth=1.5)
plt.setp(line3, color='red', linewidth=3.0)
plt.setp(line4, color='lime', linewidth=3.0)
plt.legend(handles=(line1, line2, line3, line4, line5), labels=('Proposed scheme', 'Random selection scheme', 'Maximum selection scheme', 'Static selection scheme', 'Maximum length'), prop={'size':30})
plt.xlabel('Time slot (50ms)', fontsize=40)
plt.ylabel('Queue backlog (MB)', fontsize=40)
plt.tick_params(labelsize=35)
plt.grid(True)
# plt.setp(line4, color='k', linewidth=1.0)

# plt.xlabel('Queue Backlog')
# plt.ylabel('Expected Accuracy')

_,_,_,_,_, Total_Data_Proposed,_,_,_ = output_avg(res_path)
_,_,_,_,_,_, Total_Data_Random,_,_ = output_avg(res_path)
plt.figure(3)
Select = ('Proposed')
Random = ('Random')
bar_width = 0.7
# Total_Data_Proposed = Total_Data_Proposed / 10
# Total_Data_Random = Total_Data_Random / 10
plt.bar(Select, height=Total_Data_Proposed, width=bar_width, color='magenta')
plt.bar(Random, height=Total_Data_Random, width=bar_width, color='b')
plt.xlim(-0.7, 1.7)
# plt.legend()  
plt.tick_params(labelsize=35)
plt.ylabel('Total number of selected vehicles', fontsize=40)
print(Total_Data_Random)
print(Total_Data_Proposed)

_,_,_,_,_,_,_, s_star_proposed,_ = output_avg(res_path)
_,_,_,_,_,_,_,_, s_star_random= output_avg(res_path)
plt.figure(figsize=(1000,5))
plt.xlabel('Time slot (50ms)', fontsize=40)
plt.ylabel('The optimal number of \n selected vehicles', fontsize=40)
plt.title('Proposed scheme', fontsize=40)
plt.tick_params(labelsize=35)
time_slot = np.arange(len(s_star_proposed))
plt.plot(time_slot, s_star_proposed, color='magenta', linewidth=1)
# plt.legend()

plt.figure(figsize=(1000,5))
plt.xlabel('Time slot (50ms)', fontsize=40)
plt.ylabel('The optimal number of \n selected vehicles', fontsize=40)
plt.title('Random selection scheme', fontsize=40)
plt.tick_params(labelsize=35)
time_slot = np.arange(len(s_star_random))
plt.plot(time_slot, s_star_random, color='b', linewidth=1)


def output_avg_2(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs_0 = []
    avg_rs_1 = []
    for name in fileList:
        path = dir_path + name
        res = np.load(path)
        temp_rs_0 = np.array(res['arr_0'])
        temp_rs_1 = np.array(res['arr_1'])
        avg_rs_0.append(temp_rs_0)
        avg_rs_1.append(temp_rs_1)
    return avg_rs_0, avg_rs_1

res_path_2 = 'Train_data1/'
Accuracy_1,_ = output_avg_2(res_path_2)
_, Loss_1 = output_avg_2(res_path_2)


b = np.ones((9,1407))
for i in range(9):
    for j in range(1300):
        b[i][j] = Accuracy_1[i][j]

c = np.zeros((9,1407))
for k in range(9):
    for l in range(1300):
        c[k][l] = Loss_1[k][l]
    
d = []
d = moving_average(np.mean(b, axis=0, keepdims=True)[0],10)
# d = np.mean(b, axis=0, keepdims=True)[0]
# print(len(c))
sum_2=np.sum(d)
avg_2=sum_2/len(d)
print(len(d))

abc = 0
abcd = 29
d_1 = []
for i in range(abcd):
    if abc<=1300:
        d_1.append(d[abc])
        abc = abc + 50
    else:
        d_1.append(d[1397])            

e = []
e = moving_average(np.mean(c, axis=0, keepdims=True)[0],10)
# e = np.mean(c, axis=0, keepdims=True)[0]

abc_1 = 0
e_1 = []
for i in range(abcd):
    if abc_1<=1300:
        e_1.append(e[abc_1])
        abc_1 = abc_1 + 50
    else:
        e_1.append(e[1397])  



def output_avg_3(dir):
    dir_path = dir
    fileList = os.listdir(dir_path) 
    fileList = [name for name in fileList if '.npz' in name]
    avg_rs = []
    for name in fileList:
        path = dir_path + name
        res = np.load(path)
        temp_rs = np.array(res['arr_0'])
        avg_rs.append(temp_rs)
    return avg_rs

res_path_3 = 'Static_selection/'
Accuracy_2,_ = output_avg_2(res_path_3)
_, Loss_2 = output_avg_2(res_path_3)

p = []
p = moving_average(np.mean(Accuracy_2, axis=0, keepdims=True)[0],10)
q = []
q = moving_average(np.mean(Loss_2, axis=0, keepdims=True)[0],10)
abc_2 = 0
p_1 = []
for i in range(abcd):
    if abc_2<=1300:
        p_1.append(p[abc_2])
        abc_2 = abc_2 + 50
    else:
        p_1.append(p[1390]) 


abc_3 = 0
q_1 = []
for i in range(abcd):
    if abc_3<=1300:
        q_1.append(q[abc_3])
        abc_3 = abc_3 + 50
    else:
        q_1.append(q[1390]) 
# plt.figure(6)
# plt.xlim(-100, 2200)
# plt.xlabel('Time Slot (50ms)')
# plt.ylabel('Training Accuracy')
# plt.title('Control Algorithm + Weight Selection', fontsize=20)
# client_number = np.arange(len(d))
# plt.plot(client_number, d, color='b', linewidth=1)
# plt.axhline(y=1, color='k', linestyle='--', linewidth=1)

# plt.figure(7)
# plt.xlim(-100, 2200)
# plt.xlabel('Time Slot (50ms)')
# plt.ylabel('Loss')
# plt.title('Control Algorithm + Weight Selection', fontsize=20)
# client_number = np.arange(len(e))
# plt.plot(client_number, e, color='b', linewidth=1)
# plt.axhline(y=0, color='k', linestyle='--', linewidth=1)



# res_path_3 = 'Train_data2/'
# Accuracy,_ = output_avg_2(res_path_3)
# _, Loss = output_avg_2(res_path_3)
# # print(Accuracy)

# f = np.ones((2,1328))
# for i in range(2):
#     for j in range(1323):
#         b[i][j] = Accuracy[i][j]

# g = np.ones((2,1328))
# for k in range(2):
#     for l in range(1323):
#         c[k][l] = Loss[k][l]

# h = []
# h = moving_average(np.mean(f, axis=0, keepdims=True)[0],10)
# # d = np.mean(b, axis=0, keepdims=True)[0]
# # print(len(c))

# m = []
# m = moving_average(np.mean(g, axis=0, keepdims=True)[0],10)
# # e = np.mean(c, axis=0, keepdims=True)[0]

# plt.figure(8)
# plt.xlim(-100, 1700)
# plt.xlabel('Time Slot (50ms)')
# plt.ylabel('Training Accuracy')
# plt.title('Control Algorithm + Random Selection', fontsize=20)
# client_number = np.arange(len(h))
# plt.plot(client_number, h, color='red', linewidth=1)
# plt.axhline(y=1, color='k', linestyle='--', linewidth=1)

# plt.figure(9)
# plt.xlim(-100, 1700)
# plt.xlabel('Time Slot (50ms)')
# plt.ylabel('Loss')
# plt.title('Control Algorithm + Random Selection', fontsize=20)
# client_number = np.arange(len(m))
# plt.plot(client_number, m, color='red', linewidth=1)
# plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

# _, T_queue_proposed,_,_,_,_,_,_,_ = output_avg(res_path)
# _,_, T_queue_random,_,_,_,_,_,_ = output_avg(res_path)
# _,_,_, T_compare_queue1,_,_,_,_,_ = output_avg(res_path)
# _,_,_,_, T_compare_queue2,_,_,_,_ = output_avg(res_path)
# plt.figure(2)
# plt.axhline(y=Q_max, color='k', linestyle='--', linewidth=3.0)
# plt.ylim(0, Q_max * 5)
# plt.xlim(0, T_max)

# # print(T_queue_proposed)
# line1, = plt.plot(np.arange(len(T_queue_proposed)), T_queue_proposed[:], label='Proposed')
# line2, = plt.plot(np.arange(len(T_queue_random)), T_queue_random[:], label='Random')
# line3, = plt.plot(np.arange(len(T_compare_queue1)), T_compare_queue1[:], label='full')
# line4, = plt.plot(np.arange(len(T_compare_queue2)), T_compare_queue2[:], label='static')

# plt.setp(line1, color='magenta', linewidth=3.0)
# plt.setp(line2, color='b', linewidth=1.5)
# plt.setp(line3, color='red', linewidth=3.0)
# plt.setp(line4, color='lime', linewidth=3.0)
# plt.legend(handles=(line1, line2, line3, line4), labels=('Control Algorithm + Weight Selection', 'Control Algorithm + Random Selection', 'Full Selection', 'Static Selection'), prop={'size':30})
# plt.xlabel('Time Slot (50ms)', fontsize=20)
# plt.ylabel('Queue Backlog (MB)', fontsize=20)
# plt.grid(True)

res_path_2 = 'Train_data2/'
# print(output_avg_2(res_path_2))
Accuracy_1,_ = output_avg_2(res_path_2)
_, Loss_1 = output_avg_2(res_path_2)
# print(len(Accuracy_1[0]))

f = np.ones((9,1146))
for i in range(9):
    for j in range(1024):
        f[i][j] = Accuracy_1[i][j]

g = np.zeros((9,1146))
for k in range(9):
    for l in range(1024):
        g[k][l] = Loss_1[k][l]

h = []
h = moving_average(np.mean(f, axis=0, keepdims=True)[0],10)
sum_1=np.sum(h)
avg_1=sum_1/len(h)
print(len(h))

abcd_1 = 24
abc_4 = 0
h_1 = []
for i in range(abcd_1):
    if abc_4<=1100:
        h_1.append(h[abc_4])
        abc_4 = abc_4 + 50
    else:
        h_1.append(h[1136]) 

# d = np.mean(b, axis=0, keepdims=True)[0]
# print(len(c))

o = []
o = moving_average(np.mean(g, axis=0, keepdims=True)[0],10)
# e = np.mean(c, axis=0, keepdims=True)[0]

abc_5 = 0
o_1 = []
for i in range(abcd_1):
    if abc_5<=1100:
        o_1.append(o[abc_5])
        abc_5 = abc_5 + 50
    else:
        o_1.append(o[1136]) 

# plt.figure(8)
# plt.xlim(-100, 1500)
# plt.xlabel('Time slot (50ms)', fontsize=40)
# plt.ylabel('Training accuracy', fontsize=40)
# plt.title('Control Algorithm + Random Selection', fontsize=40)
client_number_1 = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150,  1200, 1250,  1300, 1350, 1400]
client_number_2 = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1150,  1200, 1250,  1300, 1350, 1391]
client_number_3 = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000, 1050, 1100, 1137]
x_0 = [100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600]
# line1, = plt.plot(client_number_1, d_1, color='magenta', marker='o', linewidth=1.5)
# line2, = plt.plot(client_number_3, h_1, color='b', marker='s', linewidth=1.5)
# line3, = plt.plot(client_number_2, p_1, color='g', marker='^', linewidth=1.5)

# plt.setp(line1, color='magenta', linewidth=1)
# plt.setp(line2, color='b', linewidth=1)
# plt.setp(line3, color='g', linewidth=1)
# plt.legend(handles=(line1, line2, line3), labels=('Proposed scheme', 'Random selection scheme', 'Static selection scheme'), prop={'size':25})
# plt.tick_params(labelsize=35)
# plt.axhline(y=1, color='k', linestyle='--', linewidth=1)
# # plt.grid(True)

def zone_and_linked(ax,axins,zone_left,zone_right,x,y,linked='bottom',
                    x_ratio=0.05,y_ratio=0.05):
    """缩放内嵌图形，并且进行连线
    ax:         调用plt.subplots返回的画布。例如： fig,ax = plt.subplots(1,1)
    axins:      内嵌图的画布。 例如 axins = ax.inset_axes((0.4,0.1,0.4,0.3))
    zone_left:  要放大区域的横坐标左端点
    zone_right: 要放大区域的横坐标右端点
    x:          X轴标签
    y:          列表，所有y值
    linked:     进行连线的位置，{'bottom','top','left','right'}
    x_ratio:    X轴缩放比例
    y_ratio:    Y轴缩放比例
    """
    xlim_left = x[zone_left]-(x[zone_right]-x[zone_left])*x_ratio
    xlim_right = x[zone_right]+(x[zone_right]-x[zone_left])*x_ratio

    y_data = np.hstack([yi[zone_left:zone_right] for yi in y])
    ylim_bottom = np.min(y_data)-(np.max(y_data)-np.min(y_data))*y_ratio
    ylim_top = np.max(y_data)+(np.max(y_data)-np.min(y_data))*y_ratio

    axins.set_xlim(xlim_left, xlim_right)
    axins.set_ylim(ylim_bottom, ylim_top)

    ax.plot([xlim_left,xlim_right,xlim_right,xlim_left,xlim_left],
            [ylim_bottom,ylim_bottom,ylim_top,ylim_top,ylim_bottom],"black")

    if linked == 'bottom':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_left,ylim_bottom)
        xyA_2, xyB_2 = (xlim_right,ylim_top), (xlim_right,ylim_bottom)
    elif  linked == 'top':
        xyA_1, xyB_1 = (xlim_left,ylim_bottom), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_right,ylim_top)
    elif  linked == 'left':
        xyA_1, xyB_1 = (xlim_right,ylim_top), (xlim_left,ylim_top)
        xyA_2, xyB_2 = (xlim_right,ylim_bottom), (xlim_left,ylim_bottom)
    elif  linked == 'right':
        xyA_1, xyB_1 = (xlim_left,ylim_top), (xlim_right,ylim_top)
        xyA_2, xyB_2 = (xlim_left,ylim_bottom), (xlim_right,ylim_bottom)
        
    con = ConnectionPatch(xyA=xyA_1,xyB=xyB_1,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)
    con = ConnectionPatch(xyA=xyA_2,xyB=xyB_2,coordsA="data",
                          coordsB="data",axesA=axins,axesB=ax)
    axins.add_artist(con)

fig, ax = plt.subplots(1,1,figsize=(12, 7))
# client_number_1 = np.arange(len(d_1))
# client_number_2 = np.arange(len(h_1))
# client_number_3 = np.arange(len(p_1))
ax.plot(client_number_1, d_1, color='magenta', marker='o', linewidth=1.5, label='Proposed scheme', alpha=0.7)
ax.plot(client_number_3, h_1, color='b', marker='s', linewidth=1.5, label='Random selection scheme', alpha=0.7)
ax.plot(client_number_2, p_1, color='g', marker='^', linewidth=1.5, label='Static selection scheme', alpha=0.7)
ax.legend(loc='right')
plt.xlim(-100, 1500)
plt.tick_params(labelsize=35)
plt.xlabel('Time slot (50ms)', fontsize=40)
plt.ylabel('Training accuracy', fontsize=40)
plt.legend(labels=('Proposed scheme', 'Random selection scheme', 'Static selection scheme'), loc='right', prop={'size':20})
plt.axhline(y=1, color='k', linestyle='--', linewidth=1)

axins = ax.inset_axes((0.4, 0.1, 0.4, 0.3))
axins.plot(client_number_1, d_1, color='magenta', marker='o', linewidth=1.5, label='Proposed scheme', alpha=0.7)
axins.plot(client_number_3, h_1, color='b', marker='s', linewidth=1.5, label='Random selection scheme', alpha=0.7)
axins.plot(client_number_2, p_1, color='g', marker='^', linewidth=1.5, label='Static selection scheme', alpha=0.7)
zone_and_linked(ax, axins, 2, 13, client_number_1, [d_1, h_1, p_1], 'right')


# plt.figure(9)
# plt.xlim(-100, 1500)
# plt.xlabel('Time slot (50ms)', fontsize=40)
# plt.ylabel('Loss', fontsize=40)
# # plt.title('Control Algorithm + Random Selection', fontsize=20)
# line1, = plt.plot(client_number_1, e_1, color='magenta', marker='o', linewidth=1.5)
# line2, = plt.plot(client_number_3, o_1, color='b', marker='s', linewidth=1.5)
# line3, = plt.plot(client_number_2, q_1, color='g', marker='^', linewidth=1.5)
# plt.setp(line1, color='magenta', linewidth=1)
# plt.setp(line2, color='b', linewidth=1)
# plt.setp(line3, color='g', linewidth=1)
# plt.legend(handles=(line1, line2, line3), labels=('Proposed scheme', 'Random selection scheme', 'Static selection scheme'), prop={'size':25})
# plt.tick_params(labelsize=35)
# plt.axhline(y=0, color='k', linestyle='--', linewidth=1)
# # plt.grid(True)

fig, ax = plt.subplots(1 ,1,figsize=(12, 7))
ax.plot(client_number_1, e_1, color='magenta', marker='o', linewidth=1.5, label='Proposed scheme', alpha=0.7)
ax.plot(client_number_3, o_1, color='b', marker='s', linewidth=1.5, label='Random selection scheme', alpha=0.7)
ax.plot(client_number_2, q_1, color='g', marker='^', linewidth=1.5, label='Static selection scheme', alpha=0.7)
ax.legend(loc='right')
plt.xlim(-100, 1500)
plt.tick_params(labelsize=35)
plt.xlabel('Time slot (50ms)', fontsize=40)
plt.ylabel('Loss', fontsize=40)
plt.legend(labels=('Proposed scheme', 'Random selection scheme', 'Static selection scheme'), loc='upper right', prop={'size':20})
plt.axhline(y=0, color='k', linestyle='--', linewidth=1)

axins = ax.inset_axes((0.4, 0.1, 0.4, 0.3))
axins.plot(client_number_1, e_1, color='magenta', marker='o', linewidth=1.5, label='Proposed scheme', alpha=0.7)
axins.plot(client_number_3, o_1, color='b', marker='s', linewidth=1.5, label='Random selection scheme', alpha=0.7)
axins.plot(client_number_2, q_1, color='g', marker='^', linewidth=1.5, label='Static selection scheme', alpha=0.7)
zone_and_linked(ax, axins, 2, 13, client_number_1, [e_1, o_1, q_1], 'right')


plt.show()