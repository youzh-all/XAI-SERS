#%%
import os
import numpy as np
import matplotlib.pyplot as plt
#%%
#Despiking 
def despiking_algorithm(x_values, y_values, ma=5, threshold=5):
        #calculates the modified z-scores of a diferentiated spectrum
        def modified_z_score(y):
            ysb = np.diff(y)
            median_y = np.median(ysb)
            median_absolute_deviation_y = np.median(np.abs(ysb - median_y))
            modified_z_scores = 0.6745 * (ysb - median_y) / median_absolute_deviation_y
            modified_z_scores = np.concatenate(([np.nan], modified_z_scores))
            return modified_z_scores

        z_scores = modified_z_score(y_values)
        spikes = abs(np.array(z_scores)) > threshold
        
        #calculates the average values around the point to be replaced.
        despiked_spectrum = y_values.copy()
        for i in np.arange(len(spikes)):
            if spikes[i] != 0:
                start_index = max(0, i - ma)  # Prevent index error
                end_index = min(len(y_values), i + ma + 1)
                w = np.arange(start_index, end_index)
                we = w[spikes[w] == 0]
                if len(we) > 0:
                    #prevent index error&type error
                    despiked_spectrum[i] = np.mean([y_values[idx] for idx in we])

        return despiked_spectrum

def despiking(folder_path, output_folder_path):
    # Create the output folder if it doesn't exist
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    file_list = os.listdir(folder_path)
    plt.figure(figsize=(12, 12))  # Set the overall plot size

    # Subplot for Original Spectra
    plt.subplot(2, 1, 1)  # Two rows, one column, first plot

    for i, file_name in enumerate(file_list):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            x_values, y_values = [], []
            with open(file_path, 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split())
                    x_values.append(x)
                    y_values.append(y)
            # Plot the original spectrum
            plt.plot(x_values[:], y_values[:])

    plt.title('BC Spectra')
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.legend()

    # Subplot for Despiked Spectra
    plt.subplot(2, 1, 2)  # Two rows, one column, second plot
    
    for file_name in file_list:
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            x_values, y_values = [], []
            with open(file_path, 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split())
                    x_values.append(x)
                    y_values.append(y)

            # Apply despiking
            despiked_spectrum = despiking_algorithm(x_values, y_values)
            # Plot the despiked spectrum
            plt.plot(x_values[:], despiked_spectrum[:])

            # save the corrected spectrum to the output folder
            new_file_name = file_name[:-4]#.split('.')[0]
            new_file_name = new_file_name + 'DS.txt'
            output_file_path = os.path.join(output_folder_path, f'{new_file_name}')
            with open(output_file_path, 'w') as output_file:
                for x, despiked_intensity in zip(x_values[:], despiked_spectrum):
                    output_file.write(f'{x} {despiked_intensity}\n')

     # Plot the despiked spectrum
    # plt.plot(x_values[164:600], despiked_spectrum[164:600], color = 'black', label = 'Mixture spectrum with noise')
    # plt.plot(x_values[164:600], despiked_spectrum[164:600], color = 'red', label = 'Despiked Mixture spectrum with noise')
    plt.title('Despiked Spectra')
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.legend()
    plt.show()

     
    
#%%
despiking('data_BC/EAEC_BC', 'data_BCDS/EAEC_BCDS')
despiking('data_BC/EIEC_BC', 'data_BCDS/EIEC_BCDS')
despiking('data_BC/EPEC_BC', 'data_BCDS/EPEC_BCDS')
despiking('data_BC/ETEC_BC', 'data_BCDS/ETEC_BCDS')
despiking('data_BC/Shigella boydii_BC', 'data_BCDS/Shigella boydii_BCDS')
despiking('data_BC/Shigella dysenteriae_BC', 'data_BCDS/Shigella dysenteriae_BCDS')
despiking('data_BC/Shigella flexneri_BC', 'data_BCDS/Shigella flexneri_BCDS')
despiking('data_BC/Shigella sonnei_BC', 'data_BCDS/Shigella sonnei_BCDS')
despiking('data_BC/STEC_BC', 'data_BCDS/STEC_BCDS')
# %%
