import numpy as np
import matplotlib.pyplot as plt

#this program plots 3 measurements
#in the form of a figure with subplots axes 
def CSV_to_matrix(file_name):
    matrix_of_data = np.empty((250,250))
    with open(file_name, "r") as file:
        # Read each line in the file
        row_number = 1
        for line in file:
            # Split the line by semicolon to get individual values
            values = line.strip().split(';')
            values = values[:-1]
            # Process the values as needed
            row_of_pixels = []
            for pixel in values:
                row_of_pixels += [float(pixel[1:-1])]
            #print(len(row_of_pixels))
            #print('Row number = ' + str(row_number))
            #print(np.array(row_of_pixels))
            matrix_of_data[row_number - 1] = row_of_pixels
            row_number += 1
    return matrix_of_data
#convert all measurements from raw to matrix

M33 = CSV_to_matrix("C:/Users/Matthew Ngai/OneDrive/桌面/M33_23.02.2024_Sample_Res250px_Speed80pps.csv")

M35 = CSV_to_matrix("C:/Users/Matthew Ngai/OneDrive/桌面/M35_23.02.2024_Sample_Res250px_Speed110pps.csv")

M37 = CSV_to_matrix("C:/Users/Matthew Ngai/OneDrive/桌面/M37_26.02.2024_Sample_Res250px_Speed80pps.csv")

data=[M33,M35,M37] 

#create the grid for placing different plots, now 1 ROW X 3 COLUMNS
fig, axes = plt.subplots(nrows=1, ncols=3, figsize=(18, 6)) 

#the overall title of the figure, not necessary for report figures
fig.suptitle('Mild steel corrosion series 1',size='xx-large', y=0.9) 
#subtitles of the individual plots
titles=['M33: First mild steel sample \n before corrosion', 'M35: First mild steel sample \n after 5 mins in 0.1M HCl', 
        'M37: First mild steel sample \n after another 10 mins in 1M HCl'] 

#plotting the subplots 
for i in range(3): 

    ax=axes.flat[i] 

    ax.set_xticks(np.linspace(0,250,11)) 

    ax.set_yticks(np.linspace(0,250,11)) 

    ax.grid(color='black', linestyle='-', linewidth=1) 

    im = ax.imshow(data[i], cmap='gnuplot', origin = 'lower') 

    ax.set_title(titles[i]) 

cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8) 
plt.savefig('C:/Users/Matthew Ngai/Downloads/comp_plots_1')
plt.show()
# notice that here we use ax param of figure.colorbar method instead of 

# the cax param as the above example 



#cbar.set_ticks() 

#invert the colorbar so it becomes clear that in constant force mode higher values correspond to lower sample height 

#cbar.ax.invert_yaxis() 