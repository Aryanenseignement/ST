from Scripts.MLE import AR1Forecast


if __name__ == "__main__":

    path = r"C:/Users/Razaghi/Desktop/TS/Data/Data.csv"
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