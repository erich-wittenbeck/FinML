
import pandas as pd
import matplotlib.pyplot as plt

from lib.utility.timeseries import slice_timeseries
from lib.utility.visualization import plot_simulation

# TODO: Rework simulator, such that it can potentially handle live data in the future

class Simulator():

    def __init__(self):
        self.__actions = {}
        self.__sim_logs = {}

    # Getters

    def __get_logs(self):
        return self.__sim_logs

    # Define actions on signals

    def on_signal(self, signal, do_what, how_much=None):

        self.__actions[signal] = (do_what, how_much)

        return self

    def run_simulation(self, test_back_data, initial_budget, initial_assets, *models, transaction_rate=0, act_every=1):
        history, X = test_back_data.chart, test_back_data.X

        # Helper function

        def _yield(predictions):
            budget, assets = initial_budget, initial_assets

            for idx, stamp in enumerate(predictions.index):

                if idx % act_every != 0:
                    yield ('hold', budget, assets)
                else:
                    signal = predictions.loc[stamp]
                    close_price = history.loc[stamp]['close']
                    if signal in self.__actions:
                        action, quantity = self.__actions[signal]

                        if action == 'buy':
                            money_spent = budget*quantity
                            fees_paid = money_spent*transaction_rate
                            assets_bought = (money_spent - fees_paid)/close_price

                            budget -= money_spent
                            assets += assets_bought

                            yield ('buy', budget, assets)
                        elif action == 'sell':
                            assets_sold = assets * quantity
                            money_gained = assets_sold * close_price
                            fees_paid = money_gained*transaction_rate

                            budget += (money_gained - fees_paid)
                            assets -= assets_sold

                            yield ('sell', budget, assets)
                        else:
                            yield ('hold', budget, assets)
                    else:
                        yield ('hold', budget, assets)


        for model in models:
            predictions = model.predict(X)
            log_as_list = [tup for tup in _yield(predictions)]
            log = pd.DataFrame(log_as_list, columns=['action', 'budget', 'assets'], index=predictions.index)

            self.__sim_logs[model.name] = _SimLog(history, log, act_every)

        return self



    """ Properties """

    logs = property(__get_logs)

class _SimLog():

    def __init__(self, history, trail, frequency):

        self.__history = history
        self.__trail = trail
        self.__frequency = frequency

        return

    # Getters

    def __get_trail(self):

        return self.__trail

    def plot(self, start=None, end=None):
        history, trail = self.__history, self.__trail

        if start != None:
            history = slice_timeseries(history, start, end)
            trail = slice_timeseries(trail, start, end)

        fig, (ax_history, ax_assets, ax_budget) = plt.subplots(3, sharex=True)
        plot_simulation(ax_history, ax_assets, ax_budget, history, trail, self.__frequency)

        plt.show()

    """ Properties """

    trail = property(__get_trail)