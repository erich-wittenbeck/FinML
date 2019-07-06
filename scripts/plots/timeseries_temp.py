
import pandas as pd
import matplotlib.pyplot as plt

from random import choice, uniform

def temperature_walk(start, lower_bound, upper_bound, no_of_steps, max_step_size):

    current = start

    yield current

    for i in range(1, no_of_steps):

        dir = choice([1, -1])

        if (current == upper_bound and dir == 1) or (current == lower_bound and dir == -1):
            yield current
        else:
            delta = upper_bound - current if dir == 1 else current - lower_bound
            step = delta * uniform(0, max_step_size)
            current = current + step*dir
            yield current

x_axis = ["Day " + str(i) for i in range(1, 31)]

city_a = pd.Series([t for t in temperature_walk(15, -5, 20, 30, 0.5)], index=x_axis)
city_b = pd.Series([t for t in temperature_walk(5, -10, 15, 30, 0.5)], index=x_axis)
city_c = pd.Series([t for t in temperature_walk(-5, -15, 10, 30, 0.5)], index=x_axis)
city_d = pd.Series([t for t in temperature_walk(-15, -20, 5, 30, 0.5)], index=x_axis)

plt.plot(city_a, marker='o', label='City A')
plt.plot(city_b, marker='o', label='City B')
plt.plot(city_c, marker='o', label='City C')
plt.plot(city_d, marker='o', label='City D')

plt.xticks(rotation=45)
plt.xlim(right=29, left=0)
# plt.ylim(top=20, bottom=-5)
plt.ylim(top=20, bottom=-20)
plt.ylabel('Temperature in °C')
# plt.hlines([-5, 0, 5, 10, 15], 0, 29, linestyles='dotted', color='grey')
plt.hlines([-15, -10, -5, 0, 5, 10, 15], 0, 29, linestyles='dotted', color='grey')
plt.legend(loc='lower right')

plt.show()