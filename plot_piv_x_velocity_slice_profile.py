import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.ion()

avm_number = 14
case_num = 2
run_list = [12, 16, 19]

upper_point = 0.0  # 左側のy座標
lower_point = 96.940  # 右側のy座標
x_line_target = (upper_point + lower_point) / 2

valid_flag_value = 3

save_fig = True  # グラフを保存する場合はTrue
save_dir = r"R:\AVM22_Script\figures"  # 保存先ディレクトリ
os.makedirs(save_dir, exist_ok=True)
fig_dpi = 300  # 解像度（dpi）

for run_num in run_list:
    print(f"--- Processing run{run_num:02} ---")

    data_dir = f"R:\\AVM{avm_number}_{case_num}_DMD\\data\\run{run_num:02}\\Average01\\"
    file_path = os.path.join(data_dir, "A00000.dat")

    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        continue

    try:
        df = pd.read_table(
            file_path, header=9, sep=" ",
            names=("x", "y", "z", "u", "v", "w", "flag", "mag")
        )
    except Exception as e:
        print(f"Error reading file for run{run_num:02}: {e}")
        continue

    unique_x_coords = df['x'].unique()
    actual_x_line = unique_x_coords[np.abs(unique_x_coords - x_line_target).argmin()]

    line_data = df[df['x'] == actual_x_line]
    valid_data_on_line = line_data[line_data['flag'].isin([3, 5])].copy()

    valid_data_on_line.sort_values(by='y', inplace=True)

    # y座標の抽出範囲を指定
    y_min = -55.980  # 任意の下限値に変更
    y_max = -25.22  # 任意の上限値に変更

    # x座標でスライスした後、y範囲でさらに抽出
    range_data_on_line = valid_data_on_line[(valid_data_on_line['y'] >= y_min) & (valid_data_on_line['y'] <= y_max)].copy()
    range_data_on_line.sort_values(by='y', inplace=True)

    y_extracted = range_data_on_line['y'].values
    u_extracted = range_data_on_line['u'].values*(-1)

    u_max = np.max(u_extracted)
    u_min = np.min(u_extracted)
    y_at_u_max = y_extracted[np.argmax(u_extracted)]
    y_at_u_min = y_extracted[np.argmin(u_extracted)]

    plt.rcParams["font.family"] = "Times New Roman"
    plt.figure(figsize=(8, 6))
    x_unique = np.sort(df["x"].unique())
    y_unique = np.sort(df["y"].unique())
    X, Y = np.meshgrid(x_unique, y_unique)
    mag_grid = df["mag"].values.reshape(len(y_unique), len(x_unique))
    mag_grid = np.flipud(mag_grid)  # 上下反転
    mag_grid = np.fliplr(mag_grid)  # 左右反転

    pcm = plt.pcolormesh(X, Y, mag_grid, cmap='jet', shading='auto')
    cbar = plt.colorbar(pcm, label='Velocity Magnitude (mag)', pad=0.02, aspect=30)
    cbar.ax.tick_params(labelsize=12)
    plt.plot([actual_x_line, actual_x_line],[y_extracted.min(), y_extracted.max()], 
             'r', linewidth=2, label=f'Extraction Line (y={actual_x_line:.3f}, flag=3/5)')
    plt.xlabel('x, mm', fontsize=14)
    plt.ylabel('y, mm', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend(fontsize=12, loc='best', frameon=True)
    plt.grid(True, linestyle='--', alpha=0.4, which='both')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=4, color='gray')
    plt.tight_layout()
    if save_fig:
        save_path = os.path.join(save_dir, f"mag_map_run{run_num:02}.png")
        plt.savefig(save_path, dpi=fig_dpi, bbox_inches='tight')
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(u_extracted, y_extracted, marker='o', linestyle='-', c="black", markersize=4, label='u profile (flag=3/5)')

    plt.axhline(y=y_at_u_max, color='blue', linestyle='--', linewidth=2, label=f'x at u_max ({y_at_u_max:.2f})')
    plt.axhline(y=y_at_u_min, color='deepskyblue', linestyle='--', linewidth=2, label=f'x at u_min ({y_at_u_min:.2f})')
    plt.xlabel('u, m/s', fontsize=14)
    plt.ylabel('y, mm', fontsize=14)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6, which='both')
    plt.minorticks_on()
    plt.tick_params(axis='both', which='minor', length=4, color='gray')
    plt.legend(fontsize=12, loc='upper right', frameon=True)
    plt.tight_layout()
    if save_fig:
        save_path = os.path.join(save_dir, f"v_profile_run{run_num:02}.png")
        plt.savefig(save_path, dpi=fig_dpi, bbox_inches='tight')
    plt.show()