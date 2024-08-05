import matplotlib.pyplot as plt
import numpy as np

data_pair = np.genfromtxt('pair_pull.csv', delimiter=',', names=True)
data_rgb = np.genfromtxt('rgb_pull.csv', delimiter=',', names=True)
data_thermal = np.genfromtxt('thermal_pull.csv', delimiter=',', names=True)

plt.figure()
print(data_pair.dtype.names)
# for col_name in dataArray.dtype.names:
plt.plot(data_pair[data_pair.dtype.names[0]], data_pair[data_pair.dtype.names[1]])

plt.plot(data_pair[data_pair.dtype.names[0]], data_rgb[data_rgb.dtype.names[1]]
         )
plt.plot(data_pair[data_pair.dtype.names[0]], data_thermal[data_thermal.dtype.names[1]])
plt.xlabel('Step')
plt.ylabel('Loss')

# displaying the title
plt.title("Term Partial Loss During Training")

plt.legend(['$L_{cross}$', '$L^{all}_{visible}$', '$L^{all}_{thermal}$'])
plt.show()