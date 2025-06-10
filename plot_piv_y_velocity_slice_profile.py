import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
plt.ion()

avm_number = 14
case_num = 2
run_list = [12, 16, 19]

upper_point = -25.22  # 上側のy座標
lower_point = -55.980  # 下側のy座標
y_line_target = (upper_point + lower_point) / 2

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

    unique_y_coords = df['y'].unique()
    actual_y_line = unique_y_coords[np.abs(unique_y_coords - y_line_target).argmin()]

    line_data = df[df['y'] == actual_y_line]
    valid_data_on_line = line_data[line_data['flag'].isin([3, 5])].copy()

    valid_data_on_line.sort_values(by='x', inplace=True)

    x_extracted = valid_data_on_line['x'].values
    v_extracted = valid_data_on_line['v'].values

    v_max = np.max(v_extracted)
    v_min = np.min(v_extracted)
    x_at_v_max = x_extracted[np.argmax(v_extracted)]
    x_at_v_min = x_extracted[np.argmin(v_extracted)]

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
    plt.plot([x_extracted.min(), x_extracted.max()], [actual_y_line, actual_y_line],
             'r', linewidth=2, label=f'Extraction Line (y={actual_y_line:.3f}, flag=3/5)')
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
    plt.plot(x_extracted, v_extracted, marker='o', linestyle='-', c="black", markersize=4, label='v profile (flag=3/5)')

    plt.axvline(x=x_at_v_max, color='blue', linestyle='--', linewidth=2, label=f'x at v_max ({x_at_v_max:.2f})')
    plt.axvline(x=x_at_v_min, color='deepskyblue', linestyle='--', linewidth=2, label=f'x at v_min ({x_at_v_min:.2f})')
    plt.xlabel('x, mm', fontsize=14)
    plt.ylabel('v, m/s', fontsize=14)
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