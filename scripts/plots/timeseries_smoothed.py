
import pandas as pd
import matplotlib.pyplot as plt

from lib.utility.statistics import exponential_smoothing as exs
from random import uniform

def timeseries(start, up_step, down_step, no_up_steps, no_down_steps, no_total_steps):

    current = start
    yield current

    mode = 'up'
    up_steps_taken = 0
    down_steps_taken = 0

    for i in range(1, no_total_steps):
        if mode == 'up':
            if up_steps_taken <= no_up_steps:
                up_steps_taken += 1
                current += up_step
                yield current
            else:
                mode = 'down'
                up_steps_taken = 0
                down_steps_taken += 1
                current -= down_step
                yield current
        else:
            if down_steps_taken <= no_down_steps:
                down_steps_taken += 1
                current -= down_step
                yield current
            else:
                mode = 'up'
                down_steps_taken = 0
                up_steps_taken += 1
                current += up_step
                yield current

x_axis = ["Day " + str(i) for i in range(1, 31)]

signal = pd.Series([v for v in timeseries(5, 2, 1, 3, 2, 30)], index=x_axis)
noise = pd.Series([uniform(-5, 5) for i in range(0, 30)], index=x_axis)

received = signal + noise
smoothed_1 = exs(received, 0.75)
smoothed_2 = exs(received, 0.375)
smoothed_3 = exs(received, 0.125)

plt.plot(received, marker='o', label='Received')
plt.plot(smoothed_1, linestyle='--', label='Weak smoothing')
plt.plot(smoothed_2, linestyle='--', label='Strong smoothing')
plt.plot(smoothed_3, linestyle='--', label='Very strong smoothing')

plt.xticks(rotation=45)
plt.xlim(left=0, right=29)
plt.ylim(bottom=0, top=35)
plt.hlines([0, 5, 10, 15, 20, 25, 30], 0, 29, linestyles='dotted', color='grey')
plt.legend(loc='lower right')

plt.show()