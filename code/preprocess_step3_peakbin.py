# %%
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import pandas as pd

# prominance : 피크가 주변보다 얼마나 높은지를 나타내는 값
# distance : 피크 간의 최소 거리

def find_spectrum_peaks(x, y, prominence=0.1, distance=2): #distance 10일때 모델 성능 대부분 0.7정도 -> 4로 낮추면 0.7후반정도, prominence=0.1->0.005로 추가적으로 낮추면 피크 차이 없음
    peaks, _ = find_peaks(y, prominence=prominence, distance=distance)
    peak_x = x[peaks]
    peak_y = y[peaks]
    print(f"Found {len(peak_x)} peaks")
    print(peak_x, ':', peak_y)
    return peak_x, peak_y



### P - bin method ###
#%%
import numpy as np

def peak_binning_with_integration(x, y, bin_width=10):
    peak_x, peak_y = find_spectrum_peaks(x, y)
    bins = [[] for _ in range(len(peak_x))]
    bin_integrals = []  # 각 분할의 적분값을 저장할 리스트 초기화

    # 피크 주변 데이터 포인트들의 적분값 계산
    for j in range(len(peak_x)):
        bin_range_indices = [i for i, x_val in enumerate(x) if abs(x_val - peak_x[j]) <= bin_width / 2]
        if bin_range_indices:
            integral = np.trapz([y[i] for i in bin_range_indices], x=[x[i] for i in bin_range_indices])
            bin_integrals.append(integral)
        else:
            bin_integrals.append(0)  # 분할에 데이터 포인트가 없는 경우 적분값을 0으로 설정
    
    return peak_x, bin_integrals

def binning_with_zeros_and_integration(folder_path, output_folder_path, bin_width=10):
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    file_list = os.listdir(folder_path)

    for i, file_name in enumerate(file_list):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            x_values, y_values = np.loadtxt(file_path, unpack=True)
            
            # Peak binning with integration
            peak_x_binning, bin_integrals = peak_binning_with_integration(x_values, y_values, bin_width=bin_width)
            
            # Initialize with zeros
            modified_y_values = np.zeros_like(y_values)
            
            # Assign bin_integrals values to peaks' range and leave the rest as zeros
            for peak_x, integral in zip(peak_x_binning, bin_integrals):
                bin_range_indices = [i for i, x_val in enumerate(x_values) if abs(x_val - peak_x) <= bin_width / 2]
                for idx in bin_range_indices:
                    modified_y_values[idx] = integral / len(bin_range_indices)  # 적분값을 분할 영역의 데이터 포인트 수로 나누어 평균적인 강도 할당
            
            # Save the modified spectrum to the output folder
            new_file_name = f"{file_name[:-4]}_pBN.txt"
            output_file_path = os.path.join(output_folder_path, new_file_name)
            with open(output_file_path, 'w') as output_file:
                for x, y in zip(x_values, modified_y_values):
                    output_file.write(f'{x} {y}\n')


# %%
binning_with_zeros_and_integration('data_BCDS/EAEC_BCDS', 'data_BCDSpBN/EAEC_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/EIEC_BCDS', 'data_BCDSpBN/EIEC_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/EPEC_BCDS', 'data_BCDSpBN/EPEC_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/ETEC_BCDS', 'data_BCDSpBN/ETEC_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/Shigella boydii_BCDS', 'data_BCDSpBN/Shigella boydii_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/Shigella dysenteriae_BCDS', 'data_BCDSpBN/Shigella dysenteriae_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/Shigella flexneri_BCDS', 'data_BCDSpBN/Shigella flexneri_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/Shigella sonnei_BCDS', 'data_BCDSpBN/Shigella sonnei_BCDSpBN')
binning_with_zeros_and_integration('data_BCDS/STEC_BCDS', 'data_BCDSpBN/STEC_BCDSpBN')
# %%
