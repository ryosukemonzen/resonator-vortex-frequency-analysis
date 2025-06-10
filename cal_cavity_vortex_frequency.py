# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 10:29:09 2023
@author: aerot
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.interpolate import griddata

# === 設定項目 ===
avm_number = 14
case_num = 2
run_list = [12, 16, 19]

# 渦の周方向上にあると想定される2点（vが支配的）
target_points = [(37.951, -30.740), (37.951, -41.366)]  # [上側, 下側]

for run_num in run_list:

    # 入力ファイルのパス（平均場ファイル）
    data_dir = f"R:\\AVM{avm_number}_{case_num}_DMD\\data\\run{run_num:02}\\Average01\\"
    file_path = os.path.join(data_dir, "A00000.dat")

    # データの読み込み
    df = pd.read_table(
        file_path, header=9, sep=" ",
        names=("x", "y", "z", "u", "v", "w", "flag", "mag")
    )

    # u, v を格納する配列
    u_series = np.zeros(len(target_points))
    v_series = np.zeros(len(target_points))

    # 指定点からu, v抽出
    for i, (x_tgt, y_tgt) in enumerate(target_points):
        row = df[(df["x"] == x_tgt) & (df["y"] == y_tgt)]
        u_series[i] = row.iloc[0]["u"]
        v_series[i] = row.iloc[0]["v"]

    # 速度ベクトルの大きさ（代表速度）
    U1 = np.sqrt(u_series[0] ** 2 + v_series[0] ** 2)
    U2 = np.sqrt(u_series[1] ** 2 + v_series[1] ** 2)
    v1 = v_series[0]
    v2 = v_series[1]

    U_avg = np.mean([U1, U2])
    v_ave = np.mean([v1, v2])

    # 2点間距離
    x1, y1 = target_points[0]
    x2, y2 = target_points[1]
    L = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
    L_v = abs(y2-y1)
    L = L/1000
    L_v = L_v / 1000  # 上下の距離を半分にする

    # 周期の概算 T = L / U
    T_est = L / U_avg 
    T_v = L_v / v_ave  
    F = 1 / T_est
    f  = 1 / T_v
    print(f"[run{run_num:02}] 平均速度からの周期概算: {T_est:.4f}（単位: 座標/速度）")
    print(f"[run{run_num:02}] 周波数概算: {F:.4f} Hz")
    print(f)



    # x, y, mag の抽出
    x = df["x"].values
    y = df["y"].values
    mag = df["mag"].values

    # 一意な x, y 値を抽出（→ グリッドの軸になる）
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())

    # 2D グリッドデータに変形（reshape）
    mag_grid = mag.reshape(len(y_unique), len(x_unique))  # y方向が先！
    mag_grid = np.flipud(mag_grid)  # 上下反転

    # メッシュグリッド作成（描画用）
    X, Y = np.meshgrid(x_unique, y_unique)

    plt.figure(figsize=(10, 10))
    pcm = plt.pcolormesh(X, Y, mag_grid, cmap='jet', shading='auto')
    plt.colorbar(pcm, label='mag')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.tight_layout()
    # plot target points
    target_x = [p[0] for p in target_points]
    target_y = [p[1] for p in target_points]
    plt.scatter(target_x, target_y, color='black', marker='o', s=30, label='target points')
    plt.legend()
    plt.gca().set_aspect('equal')
    #plt.gca().invert_xaxis()
    #plt.gca().invert_yaxis()
    plt.show()

