#!/usr/bin/env python
# coding: utf-8

def inference():
    from os.path import abspath, join, pardir
    import numpy as np # linear algebra
    import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
    import plotly.graph_objects as go

    root_path = abspath(join(__file__, pardir))
    datasets_path = abspath(join(root_path, "datasets"))
    inference_path = abspath(join(datasets_path, "inference"))
    outputs_path = abspath(join(root_path, "outputs"))
    outputs_inference_path = abspath(join(outputs_path, "inference"))
    models_path = abspath(join(root_path, "models"))

    # Data loading
    generation1_file = abspath(join(inference_path, "Plant_1_Generation_Data.csv"))
    generation1 = pd.read_csv(generation1_file)
    weather1_file =  abspath(join(inference_path, "Plant_1_Weather_Sensor_Data.csv"))
    weather1 = pd.read_csv(weather1_file)
    generation1['DATE_TIME'] = pd.to_datetime(generation1['DATE_TIME'], dayfirst=True)
    weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], dayfirst=False)

    inverters = list(generation1['SOURCE_KEY'].unique())
    print(f"total number of inverters {len(inverters)}")

    # EDA
    inv_1 = generation1[generation1['SOURCE_KEY']==inverters[0]]
    mask = ((weather1['DATE_TIME'] >= min(inv_1["DATE_TIME"])) & (weather1['DATE_TIME'] <= max(inv_1["DATE_TIME"])))
    weather_filtered = weather1.loc[mask]

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=inv_1["DATE_TIME"], y=inv_1["AC_POWER"],
                        mode='lines',
                        name='AC Power'))

    fig.add_trace(go.Scatter(x=weather_filtered["DATE_TIME"], y=weather_filtered["IRRADIATION"],
                        mode='lines',
                        name='Irradiation', 
                        yaxis='y2'))

    fig.update_layout(title_text="Irradiation vs AC POWER",
                    yaxis1=dict(title="AC Power in kW",
                                side='left'),
                    yaxis2=dict(title="Irradiation index",
                                side='right',
                                anchor="x",
                                overlaying="y"
                                ))

    figure_file = abspath(join(outputs_inference_path, "AC_power.png"))
    fig.write_image(figure_file)

    # Feature Engineering
    df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
    df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df_timestamp = df[["DATE_TIME"]]
    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X = scaler.fit_transform(df_.loc[:df_.shape[0]])
    X = X.reshape(X.shape[0], 1, X.shape[1])

    from tensorflow.keras.models import load_model
    model_file = abspath(join(models_path, "lstm_model.keras"))
    cached_model = load_model(model_file)

    X_pred = cached_model.predict(X)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=df_.columns)
    X_pred.index = df.index

    scores = X_pred.copy()
    scores['datetime'] = df_timestamp
    scores['real AC'] = df['AC_POWER']
    scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
    scores['Threshold'] = 200
    scores['Anomaly'] = np.where(scores["loss_mae"] > scores["Threshold"], 1, 0)
    scores['Anomaly'].value_counts()

    anomalies = scores[scores['Anomaly'] == 1][['real AC']]
    anomalies = anomalies.rename(columns={'real AC':'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')

    anomaly_csv_file = abspath(join(outputs_inference_path, "anomalies.csv"))
    scores[(scores['Anomaly'] == 1) & (scores['datetime'].notnull())].to_csv(anomaly_csv_file, index=False)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["real AC"],
                        mode='lines',
                        name='AC Power'))

    fig.add_trace(go.Scatter(x=scores["datetime"], y=scores["anomalies"],
                        name='Anomaly', 
                        mode='markers',
                        marker=dict(color="red",
                                    size=11,
                                    line=dict(color="red",
                                            width=2))))

    fig.update_layout(title_text="Anomalies Detected LSTM Autoencoder")

    figure_file = abspath(join(outputs_inference_path, "Anomaly.png"))
    fig.write_image(figure_file)

    return

if __name__ == "__main__":
    inference()