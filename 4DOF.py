import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter
from matplotlib.ticker import FixedLocator, FixedFormatter
import openpyxl
import csv
import sys
import os
from pathlib import Path


# global data
responseIdx = 2
m1 = 100
m2 = 100
m3 = 100
m4 = 0.0001
k1 = 100 * (10 ** 3)
k2 = 200 * (10 ** 3)
k3 = 500 * (10 ** 3)
k4 = 0.0001
c1 = 50
c2 = 100
c3 = 250
c4 = 0.0001
f1 = 0
f2 = 0
f3 = 1
f4 = 0
startFreq = 1
endFreq = 100
stepFreq = 0.05


# functions definition
def initialization_value(input_value: str):
    global responseIdx, m1, m2, m3, m4, k1, k2, k3, k4, c1, c2, c3, c4, f1, f2, f3, f4, startFreq, endFreq, stepFreq
    if input_value == 'excel':
        exe_path = Path(sys.argv[0]).parent
        os.chdir(exe_path)
        excel_path = 'value.xlsx'
        workbook = openpyxl.load_workbook(excel_path, data_only=True)
        worksheet = workbook['Sheet1']
        responseIdx = worksheet['G1'].value
        m1 = worksheet['G2'].value
        m2 = worksheet['G3'].value
        m3 = worksheet['G4'].value
        m4 = worksheet['G5'].value
        k1 = worksheet['G6'].value
        k2 = worksheet['G7'].value
        k3 = worksheet['G8'].value
        k4 = worksheet['G9'].value
        c1 = worksheet['G10'].value
        c2 = worksheet['G11'].value
        c3 = worksheet['G12'].value
        c4 = worksheet['G13'].value
        f1 = worksheet['G14'].value
        f2 = worksheet['G15'].value
        f3 = worksheet['G16'].value
        f4 = worksheet['G17'].value
        startFreq = worksheet['G18'].value
        endFreq = worksheet['G19'].value
        stepFreq = worksheet['G20'].value
        workbook.close()
        print('successfully read data from excel')
    elif input_value == 'example':
        print("Use example data")
    else:
        print('Invalid input, you can only choose "excel" or "example"')


# main code
flag = input("Do you want to read the value from excel or just check the example? (enter [excel] or [example])")
initialization_value(flag)

M = np.matrix([[m1, 0, 0, 0], [0, m2, 0, 0], [0, 0, m3, 0], [0, 0, 0, m4]])
C = np.matrix([[c1 + c2, -c2, 0, 0], [-c2, c2 + c3, -c3, 0], [0, -c3, c3 + c4, -c4], [0, 0, -c4, c4]])
K = np.matrix([[k1 + k2, -k2, 0, 0], [-k2, k2 + k3, -k3, 0], [0, -k3, k3 + k4, -k4], [0, 0, -k4, k4]])
F = np.matrix([[f1], [f2], [f3], [f4]])
X = np.matrix([[0], [0], [0], [0]])

# calculate eigenvalue and eigenvector
eigenValue, vector = LA.eig(LA.inv(M) * K)
omega = np.sqrt(eigenValue)
frequency = omega / (2 * np.pi)
idx = eigenValue.argsort()
omega = omega[idx]
frequency = frequency[idx]
vector = vector[:, idx]
print("resonance:\n", frequency)

# calculate relative mass, stiffness and damping of each eigen mode
equivStiff = np.empty(0)
equivMass = np.empty(0)
for i in range(M.shape[0]):
    tempK = vector[:, i].transpose() * K * vector[:, i]
    equivStiff = np.append(equivStiff, LA.norm(tempK))
    tempM = vector[:, i].transpose() * M * vector[:, i]
    equivMass = np.append(equivMass, LA.norm(tempM))

print("equivalent stiff: ", equivStiff)
print("equivalent mass: ", equivMass)

# Calculate response curve of [responseIdx] mass
gain_array = np.empty(0)
phase_array = np.empty(0)
freq_array = np.empty(0)

# calculate compliance
for f in np.arange(startFreq, endFreq, stepFreq):
    temp = -((2 * np.pi * f) ** 2) * M + 1j * (2 * np.pi * f) * C + K
    X = np.dot(LA.inv(temp), F)

    disp_gain = 20 * np.log10(LA.norm(X.item(responseIdx)))
    gain_array = np.append(gain_array, disp_gain)

    disp_phase_elem = np.angle(X.item(responseIdx))
    disp_phase_elem = np.rad2deg(disp_phase_elem).item(0)
    phase_array = np.append(phase_array, disp_phase_elem)

# calculate IPI
for f in np.arange(startFreq, endFreq, stepFreq):
    temp = -((2 * np.pi * f) ** 2) * M + 1j * (2 * np.pi * f) * C + K
    temp_acc = - temp / ((2 * np.pi * f) ** 2)
    X_2dot = np.dot(LA.inv(temp_acc), F)

    acc_gain = 20 * np.log10(LA.norm(X_2dot.item(responseIdx)))
    gain_array = np.append(gain_array, acc_gain)

    acc_phase_elem = np.angle(X_2dot.item(responseIdx))
    acc_phase_elem = np.rad2deg(acc_phase_elem).item(0)
    phase_array = np.append(phase_array, acc_phase_elem)

    # save frequency into array
    freq_array = np.append(freq_array, f)

# reshape all array from 1*2n (n = amount of data) to n*2 to satisfy output format
amount_of_data = int(round((endFreq - startFreq) / stepFreq))
gain_array = np.reshape(gain_array, (2, amount_of_data))
phase_array = np.reshape(phase_array, (2, amount_of_data))
freq_array = np.reshape(freq_array, (1, amount_of_data))

# output value to.csv
with open('output.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['freq(Hz)', 'gain(dB)', 'phase(deg)'])
    label = np.array(["Compliance", "IPI"])
    for i in range(2):
        writer.writerow([])
        writer.writerow([label[i]])
        for f in np.arange(int(round(endFreq-startFreq)/stepFreq)):
            writer.writerow([freq_array[0, round(f)], gain_array[i, round(f)], phase_array[i, round(f)]])

# draw phase figure
fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=1)
ax1.grid()
ax1.set_xscale("log")
ax1.set_yscale("linear")
for i in range(2):
    ax1.plot(freq_array[0, :], phase_array[i, :])
ax1.set_xlim([startFreq, endFreq])
ax1.set_ylim([-360, 360])
ax1.set_position([0.1, 0.7, 0.8, 0.2])
y_ticks = [-360, -180, 0, 180, 360]
ax1.yaxis.set_major_locator(FixedLocator(y_ticks))
ax1.yaxis.set_major_formatter(FixedFormatter([str(y) for y in y_ticks]))
formatter = ax1.get_xaxis().get_major_formatter()
if not isinstance(formatter, ScalarFormatter):
    ax1.xaxis.set_major_formatter(ScalarFormatter())
ax1.ticklabel_format(style="plain", axis="x")

# draw gain figure
ax2.grid()
ax2.set_xscale("log")
ax2.set_yscale("linear")
label = np.array(["compliance", "IPI"])
for i in range(2):
    ax2.plot(freq_array[0, :], gain_array[i, :], label=label[i])
legend = ax2.legend(loc="upper right", shadow=True, fontsize='small')
ax2.set_xlim([startFreq, endFreq])
ax2.set_position([0.1, 0.1, 0.8, 0.5])
formatter = ax2.get_xaxis().get_major_formatter()
if not isinstance(formatter, ScalarFormatter):
    ax2.xaxis.set_major_formatter(ScalarFormatter())
ax2.ticklabel_format(style="plain", axis="x")

plt.show()
