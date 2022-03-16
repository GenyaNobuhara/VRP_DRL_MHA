from time import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from model import AttentionModel

def get_near_doctor(doctors,patient):
	"""Returns extra zeros from path.
    Return 
	   Dynamical model generates duplicated zeros for several graphs when obtaining partial solutions.
	"""


	return doctors


def get_doctor_time_flow(path_list,user_position):
    """
    医師の時間の流れを返す
    全部リストで取得して計算を行う
    doctor_time:医師の時間変化
    path_list_kai:医師の移動経路+診療にプラスする感じ
    """
    doctor_time = [[0] * 1 for i in range(len(path_list))]
    path_list_kai = [[0] * 1 for i in range(len(path_list))]
    for i in range(len(path_list)):
        #計算
        for j in range(len(path_list[i])-1):
            doctor_time[i].append(doctor_time[i][-1] + get_move_time(user_position[path_list[i][j]],user_position[path_list[i][j+1]]))
            path_list_kai[i].append(path_list[i][j+1])
            if(path_list[i][j+1]>0):
                doctor_time[i].append(doctor_time[i][-1] + 1/12)
                path_list_kai[i].append(path_list[i][j+1])
    return doctor_time, path_list_kai

def get_doctor_position(path_list,user_position,sudden_time):
    """
    往診発生時点での医師の位置を返す
    """
    doctor_time, path_list_kai = get_doctor_time_flow(path_list,user_position)
    doctor_position = []
    doctor_position_idx = []
    for i in range(len(doctor_time)):
        for j in range(len(doctor_time[i])):
            if(doctor_time[i][j]<sudden_time):
                if(i == len(doctor_time[i])):
                    doctor_position.append([user_position[0][0],user_position[0][1]])
                    doctor_position_idx.append(0)
                #continue
            else:
                if(path_list_kai[i][j] == path_list_kai[i][j-1]):
                    doctor_position.append([user_position[path_list_kai[i][j]][0],user_position[path_list_kai[i][j]][1]])
                    doctor_position_idx.append(path_list_kai[i][j])
                    break
                else:
                    doctor_x = user_position[path_list_kai[i][j-1]][0]*(doctor_time[i][j] - sudden_time)/(doctor_time[i][j]-doctor_time[i][j-1]) + user_position[path_list_kai[i][j]][0]*(sudden_time-doctor_time[i][j-1])/(doctor_time[i][j]-doctor_time[i][j-1])
                    doctor_y = user_position[path_list_kai[i][j-1]][1]*(doctor_time[i][j] - sudden_time)/(doctor_time[i][j]-doctor_time[i][j-1]) + user_position[path_list_kai[i][j]][1]*(sudden_time-doctor_time[i][j-1])/(doctor_time[i][j]-doctor_time[i][j-1])
                    doctor_position.append([doctor_x,doctor_y])
                    doctor_position_idx.append(path_list_kai[i][j])
                    break
    return doctor_position,doctor_position_idx,doctor_time

def set_route_greedy(pi,doctor_position,doctor_position_idx,sudden_xy,sudden_id):
    """
    往診患者に最も近い人が次の診療で向かうようにルートを変更する
    """
    minimum_distance = 100
    minimum_idx = 0
    which_doctor = 0
    for i in range(len(doctor_position)):
        dis = np.sqrt((doctor_position[i][0]-sudden_xy[0])**2 + (doctor_position[i][1]-sudden_xy[1])**2)
        if(dis < minimum_distance):
            minimum_distance = dis
            minimum_idx = doctor_position_idx[i]
            which_doctor = i+1

    pi_idx = [i for i, x in enumerate(pi) if x == minimum_idx]

    if(len(pi_idx) == 1):
        pi.insert(pi_idx[0]+1,sudden_id)
    else:
        pi.insert(pi_idx[which_doctor]+1,sudden_id)
    return(pi)


def get_move_time(start_position,finish_position):
    """
    スタート地点と終了地点の移動時間を返す
    """
    return np.sqrt((start_position[0]-finish_position[0])**2 + (start_position[1]-finish_position[1])**2)/3


def get_costs(path_list,user_position,readyTime,dueTime,sudden_time):
    """
    経路のコストを返す
    """
    dis_cost = 0
    time_penalty = 0
    FT = 0
    for i in range(len(path_list)):
        T = 0
        for j in range(len(path_list[i])-1):
            dis_cost += np.sqrt((user_position[path_list[i][j]][0]-user_position[path_list[i][j+1]][0])**2 + (user_position[path_list[i][j]][1]-user_position[path_list[i][j+1]][1])**2)
            T += np.sqrt((user_position[path_list[i][j]][0]-user_position[path_list[i][j+1]][0])**2 + (user_position[path_list[i][j]][1]-user_position[path_list[i][j+1]][1])**2)/3
            if(T>dueTime[path_list[i][j+1]] or T< readyTime[path_list[i][j+1]]):
                if(T>dueTime[path_list[i][j+1]]):
                    time_penalty += T - dueTime[path_list[i][j+1]]
                else:
                    time_penalty += readyTime[path_list[i][j+1]] - T
            if(path_list[i][j] == 21):
                FT = T - sudden_time
            T += 1/12

    return dis_cost, time_penalty,FT



def set_route_queue(path_list,user_position,sudden_id,sudden_xy,pi):
    """
    最後の患者が一番近い医師が行くようにする
    """
    minimum_dis = 100
    minimum_idx = 0
    for i in range(len(path_list)):
        dis = np.sqrt((user_position[path_list[i][-2]][0] - sudden_xy[0])**2 + (user_position[path_list[i][-2]][1] - sudden_xy[1])**2)
        if(dis < minimum_dis):
            minimum_dis = dis
            minimum_idx = path_list[i][-2]


    pi_idx = [i for i, x in enumerate(pi) if x == minimum_idx]
    pi.insert(pi_idx[0]+1,sudden_id)
    return pi
    

def set_route_go_back(doctor_time,pi,sudden_id,sudden_time):
    """
    一度診療から戻ってきた人が見る
    """
    backed_doctor = False
    backed_time = 0
    for i in range(len(doctor_time)):
        if(doctor_time[i][-1]< sudden_time):
            print(doctor_time[i][-1])
            print(sudden_time)
            backed_time = doctor_time[i][-1]
            backed_doctor = True

    if(backed_doctor):
        pi.append(sudden_id)
        pi.append(0)



    return pi,backed_doctor,backed_time

