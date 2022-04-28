import numpy as np
import random
import matplotlib.pyplot as plt
import csv
import math
import scipy.stats as stats



C_max = 100  #客户数
T_max = 1600
Q_max = 2000
BS_high = 50

# Comm
f = 2.5 * (10 ** 9)
Pt = 20  # dBm 功率
exponentN = 3

# Data
datasize = 1  # MegaByte 平均数据大小
dataPerClient = 1000   #总训练数据100000 每个客户接受1000
datachunk = 10   #单位时间传输10

static_selection_client_number = int(C_max / 24)  #static选择方法

V = 10 ** 10

# Client number data amount
CN_data_amount = np.arange(C_max+1)   #0-100数组，步长为1
CN_data_amount = CN_data_amount * datachunk

CN_accuracy = np.arange(Q_max+1, dtype='f')
l_rate = 100
d_rate = -0.3
for i in range(1, Q_max+1):
    CN_accuracy[i] =100 - l_rate * (pow(CN_accuracy[i], d_rate))
T_timeunit = np.arange(T_max)
# Q_unit = np.arange(Q_max+1)

# comm
C_distance = np.zeros(C_max)
C_receive = np.zeros(C_max)
C_comm = np.zeros(C_max)
C_comm = np.full(C_max, 1, dtype=int)

# Data
C_data_proposed = np.zeros(C_max)
C_data_proposed = np.full(C_max, dataPerClient, dtype=int)
C_data_random = C_data_proposed.copy()

   
 
#Energy
C_E = np.array([random.randint(50, 100) for x in range(C_max)])
C_E_1 = C_E


#Survivability
dhk = np.array([random.randint(0, 1000) for x in range(C_max)])
D = [1000 for i in range(100)]
Uhk = []
x, y = 15, 0.7
lower, upper = x - 2 * y, x + 2 * y
z = stats.truncnorm((lower - x) / y, (upper - x) / y, loc = x, scale = y)
Uhk = z.rvs(100)
# print(Uhk)
C_S = ((D - dhk) / Uhk)
C_S_random = C_S.copy()
C_S_static = C_S.copy()

C_priority = np.zeros(C_max)
C_fairness_random = np.zeros(C_max)
C_fairness_proposed = np.zeros(C_max)

T_client_selection = np.arange(C_max+1)

T_queue_proposed = np.zeros(T_max)
T_queue_random = np.zeros(T_max)
T_compare_queue1 = np.zeros(T_max)
T_compare_queue2 = np.zeros(T_max)
T_client_choice_proposed = np.zeros(T_max)
T_client_choice_random = np.zeros(T_max)
T_accuracy_proposed = np.zeros(T_max)
T_accuracy_random = np.zeros(T_max)
T_departure = np.zeros(T_max)
T_alive_client_proposed = np.zeros(T_max)
T_alive_client_random = np.zeros(T_max)
s_star_proposed = []
s_star_random = []
s_star_static = []
Total_Data_Proposed = 0
Total_Data_Random = 0
copy_T_queue_proposed = []
copy_T_queue_random = []
epsilon = np.zeros(C_max)
a_E = np.zeros(C_max)
r = np.zeros(C_max)
E_consume = np.zeros(C_max)
T_power = np.zeros(C_max)


for t in range(T_max):


    departure = random.random()
    if departure < 0.95:
        departure = 10 * datachunk * random.random()
    else:
        departure = 0


    max_choice_client = 0
    max_choice_data = 0

    Bo = (3 * (10**8))/f
    for i in range(C_max):
        if dhk[i] >= 0:
            m = math.sqrt((BS_high**2)+(500-dhk[i])**2)
            cos = abs(500-dhk[i])/m
            f_d = (Uhk[i]*cos)/Bo
        else:
            f_d = 0     
        alpha = 5 * (10 ** 6)
        sigma_2 = 7.5
        C_distance[i] = abs(500-dhk[i])
        epsilon[i] =1.238*(10**(-6))*(C_distance[i]*C_distance[i])+(-0.001807)*C_distance[i]+1.002
        a_E[i] = (epsilon[i] ** (Uhk[i]/C_distance[i]))*(10**10)
        r[i] = sigma_2*a_E[i]    
        T_power[i] = (r[i]/alpha)**2
        C_receive[i] = T_power[i] - (20 * np.log10(f+f_d) + 10 * exponentN * np.log10(C_distance[i]) - 28)
        C_comm[i] = C_receive[i] * random.random()
    C_comm_min = min(C_comm)
    if C_comm_min < 0:
        C_comm = C_comm - C_comm_min
    dhk = dhk - (Uhk/500)

    for i in range(C_max):
        alpha = 5 * (10 ** 6)
        sigma_2 = 7.5
        epsilon[i] =1.238*(10**(-6))*(C_distance[i]*C_distance[i])+(-0.001807)*C_distance[i]+1.002
        a_E[i] = (epsilon[i] ** (Uhk[i]/C_distance[i]))*(10**10)
        r[i] = sigma_2*a_E[i]
        E_consume[i] = abs((C_data_proposed[i]*(r[i]/alpha))/alpha)
    C_E = C_E - np.around(E_consume[i])
    C_E[C_E < 0] = 0

# Proposed Number Decision
    if t == 0:
        max_choice_client_Proposed = C_max
        max_choice_data_Proposed = max_choice_client_Proposed * datachunk

        max_choice_client_Random = C_max
        max_choice_data_Random = max_choice_client_Random * datachunk

        T_queue_proposed[t] = 0
        T_queue_random[t] = 0
        T_compare_queue1[t] = 0
        T_compare_queue2[t] = 0
    else:
        val_proposed = -9999999999
        val_temp_proposed = val_proposed - 1
        for i in range(C_max + 1):
            if 0 < (T_queue_proposed[t - 1] + CN_data_amount[i]) and (T_queue_proposed[t - 1] + CN_data_amount[i]) < Q_max + 1:
                val_temp_proposed = V * CN_accuracy[(int)(T_queue_proposed[t - 1] + CN_data_amount[i])] + T_queue_proposed[t - 1] * CN_data_amount[i]
                if val_temp_proposed > val_proposed:
                    val_proposed = val_temp_proposed
                    max_choice_client_Proposed = i
                    # print('1', i)
        if T_queue_proposed[t-1] == 0 and (t-1)!=0:
            copy_T_queue_proposed.append(t-1)
     

        val_random = -9999999999
        val_temp_random = val_random - 1
        for i in range(C_max + 1):
            if 0 < (T_queue_random[t - 1] + CN_data_amount[i]) and (T_queue_random[t - 1] + CN_data_amount[i]) < Q_max + 1:
                val_temp_random = V * CN_accuracy[(int)(T_queue_random[t - 1] + CN_data_amount[i])] + T_queue_random[t - 1] * CN_data_amount[i]
                if val_temp_random > val_random:
                    val_random = val_temp_random
                    max_choice_client_Random = i
        if T_queue_random[t-1] == 0 and (t-1)!=0: 
            copy_T_queue_random.append(t-1)



# Proposed Selection
        num_alive_client_Proposed = 0
        for x in zip(C_S, C_data_proposed):
            if x[0] > 0 and x[1] > 0:
                num_alive_client_Proposed += 1
        max_choice_client_Proposed = min(max_choice_client_Proposed, num_alive_client_Proposed)
        # print(max_choice_client_Proposed)
        alive_client_Proposed = np.argwhere((C_S > 0) & (C_data_proposed > 0))
        alive_client_Proposed = alive_client_Proposed.reshape(num_alive_client_Proposed)
        s_star_proposed.append(max_choice_client_Proposed)
        for i in range(C_max):
            if (C_S[i] <= 0) or (C_data_proposed[i] <= 0) or (C_E[i] <= 0):
                C_priority[i] = 0
            else:
                C_priority[i] = (C_data_proposed[i] * C_comm[i]) / (C_E[i] * C_S[i])
        client_choice_Proposed = C_priority.argsort()[::-1][:max_choice_client_Proposed]
        for i in client_choice_Proposed:
            C_data_proposed[i] -= datachunk
            C_fairness_proposed[i] += 1
            Total_Data_Proposed += datachunk

# Random Select
        num_alive_client_Random = 0
        for x in zip(C_S_random, C_data_random):
            if x[0] > 0 and x[1] > 0:
                num_alive_client_Random += 1
        max_choice_client_Random = min(max_choice_client_Random, num_alive_client_Random)
        alive_client_Random = np.argwhere((C_S_random > 0) & (C_data_random > 0))
        alive_client_Random = alive_client_Random.reshape(num_alive_client_Random)
        client_choice_Random = random.sample(list(alive_client_Random), max_choice_client_Random) 
        s_star_random.append(max_choice_client_Random)
        for i in client_choice_Random:
            C_data_random[i] -= datachunk
            C_fairness_random[i] += 1
            Total_Data_Random += datachunk

# Static Select
        num_alive_client_Static = 0
        for x in C_S_static:
            if x > 0:
                num_alive_client_Static += 1
        static_selection_client_number = min(static_selection_client_number, num_alive_client_Static)
        s_star_static.append(static_selection_client_number)
        for i in range(C_max):
            C_S[i] -= 0.05
            C_S_random[i] -= 0.05
            C_S_static[i] -= 0.05

        max_choice_data_Proposed = max_choice_client_Proposed * datachunk
        max_choice_data_Random = max_choice_client_Random * datachunk

        T_queue_proposed[t] = max(T_queue_proposed[t-1] + max_choice_data_Proposed - departure, 0)
        T_queue_random[t] = max(T_queue_random[t-1] + max_choice_data_Random - departure, 0)
        T_compare_queue1[t] = T_compare_queue1[t-1] + (C_max * datachunk) - departure
        T_compare_queue2[t] = max(T_compare_queue2[t-1] + static_selection_client_number * datachunk - departure, 0)
        T_client_choice_proposed[t] = max_choice_client_Proposed
        T_client_choice_random[t] = max_choice_client_Random
        T_accuracy_proposed[t] = CN_accuracy[(int)(T_queue_proposed[t])]
        T_accuracy_random[t] = CN_accuracy[(int)(T_queue_random[t])]
        T_departure[t] = departure
        T_alive_client_proposed[t] = num_alive_client_Proposed
        T_alive_client_random[t] = num_alive_client_Random

f1 = open('Proposed_power.csv', 'w')
wr = csv.writer(f1)
wr.writerow(T_queue_proposed)

f2 = open('Random_power.csv', 'w')
wr = csv.writer(f2)
wr.writerow(T_queue_random)

f3 = open('Full_power.csv', 'w')
wr = csv.writer(f3)
wr.writerow(T_compare_queue1)

f4 = open('Static_power.csv', 'w')
wr = csv.writer(f4)
wr.writerow(T_compare_queue2)

f5 = open('Fairness_proposed.csv', 'w')
wr = csv.writer(f5)
wr.writerow(C_fairness_proposed)

f6 = open('Fairness_random.csv', 'w')
wr = csv.writer(f6)
wr.writerow(C_fairness_random)

f1.close()
f2.close()
f3.close()
f4.close()
f5.close()
f6.close()

interval = 3



g = copy_T_queue_proposed[0]
h = copy_T_queue_random[0]
print(len(s_star_proposed))
print(len(s_star_random))
print(h)
print(g)

# name = 'Data/' + 'six'
# np.savez(name, CN_accuracy, T_queue_proposed, T_queue_random, T_compare_queue1, T_compare_queue2, Total_Data_Proposed / 10, Total_Data_Random / 10, s_star_proposed, s_star_random)


print("Total Data : ", dataPerClient * C_max)
print("Total_Data_Proposed : ", Total_Data_Proposed)
print("Total_Data_Random : ", Total_Data_Random)


def return_s_star_proposed():
    return s_star_proposed,g

def return_s_star_random():
    return s_star_random,h

plt.figure(2)
plt.axhline(y=Q_max, color='k', linestyle='--', linewidth=3.0)
plt.ylim(0, Q_max * 5)
plt.xlim(0, T_max)

line1, = plt.plot(np.arange(len(T_queue_proposed)), T_queue_proposed[:], label='Proposed')
line2, = plt.plot(np.arange(len(T_queue_random)), T_queue_random[:], label='Random')
line3, = plt.plot(np.arange(len(T_compare_queue1)), T_compare_queue1[:], label='full')
line4, = plt.plot(np.arange(len(T_compare_queue2)), T_compare_queue2[:], label='static')

plt.setp(line1, color='magenta', linewidth=3.0)
plt.setp(line2, color='b', linewidth=1.5)
plt.setp(line3, color='red', linewidth=3.0)
plt.setp(line4, color='lime', linewidth=3.0)
plt.legend(handles=(line1, line2, line3, line4), labels=('Control Algorithm + Weight Selection', 'Control Algorithm + Random Selection', 'Full Selection', 'Static Selection'), prop={'size':30})
plt.xlabel('Time Slot (50ms)', fontsize=20)
plt.ylabel('Queue Backlog (MB)', fontsize=20)
plt.grid(True)

plt.show()
