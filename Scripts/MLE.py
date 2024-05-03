import pandas as pd
import numpy as np
import math
import random
from scipy.optimize import minimize
import matplotlib.pyplot as plt
from scipy.stats import t
import scipy.stats as st


class AR1Forecast:
    def __init__(self, data_path):
        self.data = pd.read_csv(data_path, sep=";", decimal='.')
        self.data = self.data.set_index('Date')
        self.best_params = None
        self.forecasts = None
        self.y_true = None

    @staticmethod
    def acf(data, X):
        C = []
        for i in range(1, 20):
            xth = data[X][:len(data[X]) - i]
            xt = data[X][i:len(data[X])]

            mean_xt = sum(xt) / len(xt)
            mean_xth = sum(xth) / len(xth)

            std_dev_xt = (sum((x - mean_xt) ** 2 for x in xt) / len(xt)) ** 0.5
            std_dev_xth = (sum((x - mean_xth) ** 2 for x in xth) / len(xth)) ** 0.5

            covariance = sum((xt[j] - mean_xt) * (xth[j] - mean_xth) for j in range(len(xt))) / len(xt) - 1

            correlation_coefficient = covariance / (std_dev_xt * std_dev_xth)

            C.append(correlation_coefficient)
        return C

    @staticmethod
    def vraisemblance_AR_1(para, x):
        RhoO, Rho1, sigma = para[0], para[1], para[2]
        logv = 0
        for i in range(1, len(x)):
            if sigma <= 0:
                return float('inf')
        for i in range(1, len(x)):
            log_pdf = -0.5 * math.log(2 * math.pi) - math.log(sigma) - ((x[i] - (RhoO + Rho1 * x[i - 1])) ** 2) / (
                    2 * sigma ** 2)
            logv += log_pdf
        return -logv

    @staticmethod
    def get_initial_guess(lower_bound, upper_bound):
        return [random.uniform(lower_bound, upper_bound) for _ in range(3)]

    def fit(self):
        best_result = float('inf')
        best_guess = None
        for i in range(100):
            initial_guess = self.get_initial_guess(0, 10)
            result = minimize(self.vraisemblance_AR_1, initial_guess, args=(self.data["Eurozone"],), method='BFGS',
                              options={'disp': False, 'gtol': 1e-5})
            if result.fun < best_result:
                best_result = result.fun
                best_guess = result.x

        self.best_params = best_guess

    def make_forecast(self, forecast_length):
        x = self.data["Eurozone"].values
        forecasts = np.zeros(forecast_length)
        noise = np.random.normal(0, self.best_params[2], forecast_length)
        for i in range(forecast_length):
            forecasts[i] = self.best_params[0] + self.best_params[1] * x[-1] + noise[i]
            x = np.append(x, forecasts[i])
        self.forecasts = forecasts

    def plot_forecast(self):
        plt.plot(np.arange(len(self.data["Eurozone"])), self.data["Eurozone"], label='Observed')
        plt.plot(np.arange(len(self.data["Eurozone"]), len(self.data["Eurozone"]) + len(self.forecasts)),
                 self.forecasts, label='Forecast')
        plt.legend()
        plt.show()

    def plot_forecast_with_confidence_interval(self, confidence_level=0.95):
        lower, upper = st.norm.interval(confidence_level, loc=self.forecasts, scale=self.best_params[2])
        plt.plot(np.arange(len(self.data["Eurozone"])), self.data["Eurozone"], label='Observed')
        plt.plot(np.arange(len(self.data["Eurozone"]), len(self.data["Eurozone"]) + len(self.forecasts)),
                 self.forecasts, label='Forecast')
        plt.fill_between(np.arange(len(self.data["Eurozone"]), len(self.data["Eurozone"]) + len(self.forecasts)),
                         lower, upper, color='gray', alpha=0.5)
        plt.legend()
        plt.show()

    @staticmethod
    def rmse_calcul(y_hat, y_true):
        rmse = np.sqrt(1 / len(y_hat) * sum((np.array(y_hat) - np.array(y_true)) ** 2))
        rmse = rmse * 100
        print("RMSE is : ", round(rmse, 4), "%")

    def calculate_rmse(self, forecast_length):
        self.y_true = self.data["Eurozone"].values[-forecast_length:]
        self.rmse_calcul(self.forecasts, self.y_true)

    @staticmethod
    def mae_calcul(y_hat, y_true):
        mae = ((1 / len(y_hat)) * sum(np.absolute(y_true - y_hat)))
        mae = mae * 100
        print("MAE is : ", round(mae, 4), "%")

    def calculate_mae(self, forecast_length):
        self.y_true = self.data["Eurozone"].values[-forecast_length:]
        self.mae_calcul(self.forecasts, self.y_true)

    @staticmethod
    def theil_metric(y_true, y_pred):
        # Theil' metrics.
        sigma_a = np.std(y_true)
        sigma_p = np.std(y_pred)
        cov_a_p = np.cov(y_true, y_pred)[0][1]
        R = (cov_a_p / (sigma_a * sigma_p))

        MSE = sum((np.array(y_pred) - np.array(y_true))) ** 2 / len(y_true)
        RMSE = np.sqrt(sum((np.array(y_pred) - np.array(y_true)) ** 2) / len(y_true))

        # UM.
        num = (np.mean(y_true) - np.mean(y_pred)) ** 2
        denum = sum((np.array(y_true) - np.array(y_pred)) ** 2) / len(y_true)
        UM = num / denum

        # US.
        MSE = sum((np.array(y_pred) - np.array(y_true)) ** 2) / len(y_true)
        num = (sigma_p - sigma_a) ** 2
        denum = MSE
        US = num / denum

        # UC.
        MSE = sum((np.array(y_pred) - np.array(y_true)) ** 2) / len(y_true)
        num = (2 * (1 - R) * sigma_p * sigma_a)
        denum = MSE
        UC = num / denum

        # U1.
        RMSE = np.sqrt(sum((np.array(y_pred) - np.array(y_true)) ** 2) / len(y_true))
        num = RMSE
        denum = np.sqrt(sum(np.array(y_true) ** 2) / len(y_true))
        U1 = num / denum

        # U.
        num = np.sqrt(UM ** 2 + US ** 2 + UC ** 2)
        denum = np.sqrt(3)
        U = num / denum

        df_theil_metric = pd.DataFrame(
            {
                "UM": [UM],
                "US": [US],
                "UC": [UC],
                "U1": [U1],
                "U": [U]
            }
        )

        return df_theil_metric, UM, US, UC, U1, U

    def calculate_theil_metric(self, forecast_length):
        self.y_true = self.data["Eurozone"].values[-forecast_length:]
        df_theil_metric, UM, US, UC, U1, U = self.theil_metric(self.y_true, self.forecasts)
        return df_theil_metric, UM, US, UC, U1, U


if __name__ == "__main__":
    path = r"/Users/aryanrazaghi/Desktop/ST/Data.csv"
    ar1_forecast = AR1Forecast(path)
    ar1_forecast.fit()
    ar1_forecast.make_forecast(forecast_length=10)
    ar1_forecast.plot_forecast()
    ar1_forecast.plot_forecast_with_confidence_interval()
    ar1_forecast.calculate_rmse(forecast_length=10)
    ar1_forecast.calculate_mae(forecast_length=10)
    df_theil_metric, UM, US, UC, U1, U = ar1_forecast.calculate_theil_metric(forecast_length=10)
    print(df_theil_metric)
    print("UM:", UM)
    print("US:", US)
    print("UC:", UC)
    print("U1:", U1)
    print("U:", U)