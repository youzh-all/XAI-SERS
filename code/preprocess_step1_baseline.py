#%%
#import module
import os 
import matplotlib.pyplot as plt 
from BaselineRemoval import BaselineRemoval

#%%
#baseline_correction definition
def baseline_correction(folder_path, output_folder_path):

    #create the output folder if it doesn't exist 
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    #file명 리스트로 받아내기 
    file_list = os.listdir(folder_path)

    #subplot for original spectra 
    plt.subplot(2, 1, 1) # Two rows, one column, first plot

    for i, file_name in enumerate(file_list):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            x_values, y_values = [],[]
            with open(file_path, 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split())
                    x_values.append(x)
                    y_values.append(y)

            #plot the original spectrum
            plt.plot(x_values[164:495], y_values[164:495])
    
    plt.title('Original Spectra')
    plt.xlabel('Wavenumber')
    plt.ylabel('Intensity')
    plt.legend()


    # Subplot for corrected spectra 
    plt.subplot(2, 1, 2) #Two rows, one column, second plot

    for i, file_name in enumerate(file_list):
        if file_name.endswith('.txt'):
            file_path = os.path.join(folder_path, file_name)
            x_values, y_values = [], []
            with open (file_path, 'r') as file:
                for line in file:
                    x, y = map(float, line.strip().split())
                    x_values.append(x)
                    y_values.append(y)

            # Baseline correction 
            baseline_obj = BaselineRemoval(y_values[164:495])
            corrected_spectrum = baseline_obj.IModPoly(11) #다항식 차수는 11로

            #plot the corrected spectrum
            plt.plot(x_values[164:495], corrected_spectrum)

            # save the corrected spectrum to the output folder
            new_file_name = file_name[:-4]#.split('.')[0]
            new_file_name = new_file_name + '_BC.txt'
            output_file_path = os.path.join(output_folder_path, f'{new_file_name}')
            with open(output_file_path, 'w') as output_file:
                for x, corrected_intensity in zip(x_values[164:495], corrected_spectrum):
                    output_file.write(f'{x} {corrected_intensity}\n')

    plt.title('Corrected Spectra')
    plt.xlabel('Wavenumber')
    plt.ylabel('Corrected Intensity')
    plt.legend()
    plt.tight_layout()
    plt.show()

#%%
# Baseline Correction(BC) for all classes
baseline_correction('data/EAEC', 'data_BC/EAEC_BC')
baseline_correction('data/EIEC', 'data_BC/EIEC_BC')
baseline_correction('data/EPEC', 'data_BC/EPEC_BC')
baseline_correction('data/ETEC', 'data_BC/ETEC_BC')
baseline_correction('data/Shigella boydii', 'data_BC/Shigella boydii_BC')
baseline_correction('data/Shigella dysenteriae', 'data_BC/Shigella dysenteriae_BC')
baseline_correction('data/Shigella flexneri', 'data_BC/Shigella flexneri_BC')
baseline_correction('data/Shigella sonnei', 'data_BC/Shigella sonnei_BC')
baseline_correction('data/STEC', 'data_BC/STEC_BC')
# %%

