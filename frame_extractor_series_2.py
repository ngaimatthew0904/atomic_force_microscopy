import numpy as np
import matplotlib.pyplot as plt
import math

scan_1 = np.loadtxt("C:/Users/Matthew Ngai/OneDrive/桌面/AFM_corrosion_2/M59_processed.txt")
x_1 = 162
y_1 = 161
scan_1 = scan_1[6:y_1]
extract = np.empty((155,88))
row_num = 0
for row in scan_1:
    row = row.tolist()
    row = row[x_1:250]
    extract[row_num] = row
    row_num += 1
M59 = extract
print(len(extract))

'''
'''

def extract_frame(file_name, x_s, x_l, y_s, y_l):
    scan_2 = np.loadtxt(file_name)
    x_2 = list(range(x_s, x_l))
    y_2 = list(range(y_s, y_l))
    frames_for_trial = np.empty((100,155,88))
    frame_num = 0
    for x in x_2:
        for y in y_2:
            scan_2 = scan_2[y-155:y]
            extract_2 = np.empty((155,88))
            row_num = 0
            for row in scan_2:
                row = row.tolist()
                row = row[x:x+88]
                extract_2[row_num] = np.array(row)
                row_num += 1
            frames_for_trial[frame_num] = extract_2
            frame_num += 1
    print(frame_num)
    #print(frames_for_trial)
    diff = np.empty(100)
    diff_num = 0
    for frame in frames_for_trial:
        delta = frame - extract
        #print(delta)
        delta_val = 0
        for p in delta:
            for q in p:
                delta_val += q**2
        delta_val = np.sqrt(delta_val)
        diff[diff_num] = delta_val
        diff_num += 1
    print(diff_num)
    the_min = diff.min()
    print(diff)
    print(the_min)
    diff = diff.tolist()
    index = diff.index(the_min)
    print(index)

    scan_2 = np.loadtxt(file_name)
    res_x = x_2[0] + math.floor(index/10)
    res_y = y_2[0] + index%10
    scan_2 = scan_2[res_y-155:res_y]
    extract_3 = np.empty((y_1-6,250-x_1))
    row_num = 0
    for row in scan_2:
        row = row.tolist()
        row = row[res_x:res_x+88]
        extract_3[row_num] = row
        row_num += 1
    #plt.imshow(extract_3, cmap='gnuplot')  # 'viridis' is just an example colormap
    #plt.colorbar()  # Add a colorbar to show the mapping of values to colors
    #plt.show()
    print('(' + str(res_x) + ', ' + str(res_y) + ')')
    return extract_3

M62 = extract_frame("C:/Users/Matthew Ngai/OneDrive/桌面/AFM_corrosion_2/M62_processed.txt", 0, 10, 155, 165)
M66 = extract_frame("C:/Users/Matthew Ngai/OneDrive/桌面/AFM_corrosion_2/M66_processed.txt", 52, 62, 239, 249)
M72 = extract_frame("C:/Users/Matthew Ngai/OneDrive/桌面/AFM_corrosion_2/M72_processed.txt", 0, 10, 159, 169)
M73 = extract_frame("C:/Users/Matthew Ngai/OneDrive/桌面/AFM_corrosion_2/M73_processed.txt", 4, 14, 244, 254)


data=[M59,M62,M66,M72,M73] 

#create the grid for placing different plots, now 1 ROW X 3 COLUMNS
fig, axes = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))
fig.subplots_adjust(wspace=0.4)
fig.subplots_adjust(hspace=0.4)
#the overall title of the figure, not necessary for report figures
#fig.suptitle('Mild steel corrosion series 1 (250x250 pixels at speed 80 pps)',size='xx-large', y=0.9) 
#subtitles of the individual plots
titles=['Scan 1', 'Scan 2', 'Scan 3', 'Scan 4', 'Scan 5'] 

#plotting the subplots 
for i in range(5): 

    ax=axes.flat[i] 

    ax.grid(color='black', linestyle='dotted', linewidth=1) 

    ax.set_xticks(np.linspace(0,90,4))

    ax.tick_params(axis='x', labelsize=15)

    if i == 0:
        ax.set_yticks(np.linspace(0,180,7)) 

        ax.tick_params(axis='y', labelsize=15)
    else:
        ax.set_yticks(np.linspace(0,180,7))

        ax.tick_params(axis='y', labelsize=15)

    

    im = ax.imshow(data[i], cmap='gnuplot', origin = 'upper') 

    ax.set_title(titles[i]) 
    
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), shrink=0.1) 

    #ticklabs = cbar.ax.get_yticklabels()
    #cbar.ax.set_yticklabels(ticklabs, fontsize=15)

#plt.savefig("C:/Users/Matthew Ngai/OneDrive/桌面/AFM_corrosion_2/corrosion_series_2.png", dpi = 1000)
#plt.show()

print('At the high voltage point (20, 50), the voltages are ' + str(M59[50][20]) + ',' + str(M62[50][20]) + ','+ str(M66[50][20]) + ','+ str(M72[50][20])+ ','+ str(M73[50][20]))
print('At the high voltage point (20, 10), the voltages are ' + str(M59[10][20]) + ',' + str(M62[10][20]) + ','+ str(M66[10][20]) + ','+ str(M72[10][20])+ ','+ str(M73[10][20]))
print('At the high voltage point (40, 140), the voltages are ' + str(M59[140][40]) + ',' + str(M62[140][40]) + ','+ str(M66[140][40]) + ','+ str(M72[140][40])+ ','+ str(M73[140][40]))
print('At the low voltage point (60, 60), the voltages are ' + str(M59[60][60]) + ',' + str(M62[60][60]) + ','+ str(M66[60][60]) + ','+ str(M72[60][60])+ ','+ str(M73[60][60]))
print('At the low voltage point (50, 110), the voltages are ' + str(M59[110][50]) + ',' + str(M62[110][50]) + ','+ str(M66[110][50]) + ','+ str(M72[110][50])+ ','+ str(M73[110][50]))
print('At the low voltage point (70, 15), the voltages are ' + str(M59[15][70]) + ',' + str(M62[15][70]) + ','+ str(M66[15][70]) + ','+ str(M72[15][70])+ ','+ str(M73[15][70]))

from mpl_toolkits.mplot3d import Axes3D

data = M73

# Create x and y coordinates for the grid
x = np.arange(data.shape[1])
y = np.arange(data.shape[0])
x, y = np.meshgrid(x, y)

# Plot the surface with colormap
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
surf = ax.plot_surface(x, y, data, cmap='gnuplot')
ax.invert_zaxis()
# Add color bar
fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)

# Add labels
ax.set_xlabel('no. of pixel in x')
ax.set_ylabel('no. of pixel in y')
ax.set_zlabel('Z Piezo Voltage')

plt.show()

fig = plt.imshow(M73-M72,cmap='gnuplot')
cbar = plt.colorbar()
cbar.ax.tick_params(labelsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)
plt.ylabel('pixels in y direction', fontsize = 15)
plt.xlabel('pixels in x direction', fontsize = 15)
#plt.savefig('C:/Users/Matthew Ngai/OneDrive/桌面/AFM_corrosion_2/change_4.png')
plt.show()

corroded = 0
for a in M73-M59:
    for b in a:
        if abs(b+7.75) > 0.5:
            corroded += 1
print('Percenrtage of corrosion = ' + str(corroded*100/(155*88)) + '%')