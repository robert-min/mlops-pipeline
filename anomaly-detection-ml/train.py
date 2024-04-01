#!/usr/bin/env python
# coding: utf-8

def train():
    from os.path import abspath, join, pardir
    import numpy as np
    import pandas as pd
    import plotly.graph_objects as go

    root_path = abspath(join(__file__, pardir))
    datasets_path = abspath(join(root_path, "datasets"))
    train_path = abspath(join(datasets_path, "train"))
    outputs_path = abspath(join(root_path, "outputs"))
    outputs_train_path = abspath(join(outputs_path, "train"))
    models_path = abspath(join(root_path, "models"))

    # Data loading
    generatation1_file = abspath(join(train_path, "Plant_1_Generation_Data.csv"))
    generation1 = pd.read_csv(generatation1_file)
    weather1_file = abspath(join(train_path, "Plant_1_Weather_Sensor_Data.csv"))
    weather1 = pd.read_csv(weather1_file)

    generation1['DATE_TIME'] = pd.to_datetime(generation1['DATE_TIME'], dayfirst=True)
    weather1['DATE_TIME'] = pd.to_datetime(weather1['DATE_TIME'], dayfirst=False)
    # print(generation1)

    # EDA
    inverters = list(generation1['SOURCE_KEY'].unique())
    print(f"total number of inverters {len(inverters)}")

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

    figure_file = abspath(join(outputs_train_path, "AC_power.png"))
    fig.write_image(figure_file)
    
    # Feature Engineering
    df = inv_1.merge(weather_filtered, on="DATE_TIME", how='left')
    df = df[['DATE_TIME', 'AC_POWER', 'AMBIENT_TEMPERATURE', 'MODULE_TEMPERATURE', 'IRRADIATION']]
    df = df[["DATE_TIME", "AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]
    df_timestamp = df[["DATE_TIME"]]
    df_ = df[["AC_POWER", "AMBIENT_TEMPERATURE", "MODULE_TEMPERATURE", "IRRADIATION"]]

    train_prp = .6
    train = df_.loc[:df_.shape[0]*train_prp]
    test = df_.loc[df_.shape[0]*train_prp:]

    # Data processing
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(train)
    X_test = scaler.transform(test)
    X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
    X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
    print(f"X_train shape: {X_train.shape}")
    print(f"X_test shape: {X_test.shape}")

    # Train model
    from tensorflow.keras.layers import Input, Dropout, Dense, LSTM, TimeDistributed, RepeatVector
    from tensorflow.keras.models import Model
    from tensorflow.keras import regularizers

    def autoencoder_model(X):
        inputs = Input(shape=(X.shape[1], X.shape[2]))
        L1 = LSTM(16, activation='relu', return_sequences=True, kernel_regularizer=regularizers.l2(0.00))(inputs)
        L2 = LSTM(4, activation='relu', return_sequences=False)(L1)
        L3 = RepeatVector(X.shape[1])(L2)
        L4 = LSTM(4, activation='relu', return_sequences=True)(L3)
        L5 = LSTM(16, activation='relu', return_sequences=True)(L4)
        output = TimeDistributed(Dense(X.shape[2]))(L5)
        model = Model(inputs=inputs, outputs=output)
        return model

    model = autoencoder_model(X_train)
    model.compile(optimizer='adam', loss='mae')
    model.summary()

    epochs = 100
    batch = 10
    history = model.fit(X_train, X_train, epochs=epochs, batch_size=batch, validation_split=.2, verbose=0).history
    model_file = abspath(join(models_path, "lstm_model.keras"))
    model.save(model_file)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=[x for x in range(len(history['loss']))], y=history['loss'],
                        mode='lines',
                        name='loss'))

    fig.add_trace(go.Scatter(x=[x for x in range(len(history['val_loss']))], y=history['val_loss'],
                        mode='lines',
                        name='validation loss'))

    fig.update_layout(title="Autoencoder error loss over epochs",
                    yaxis=dict(title="Loss"),
                    xaxis=dict(title="Epoch"))

    figure_file = abspath(join(outputs_train_path, "Error_Loss.png"))
    fig.write_image(figure_file)

    # Evaluation metrics
    X_pred = model.predict(X_train)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=train.columns)

    scores = pd.DataFrame()
    scores['AC_train'] = train['AC_POWER']
    scores["AC_predicted"] = X_pred["AC_POWER"]
    scores['loss_mae'] = (scores['AC_train']-scores['AC_predicted']).abs()

    fig = go.Figure(data=[go.Histogram(x=scores['loss_mae'])])
    fig.update_layout(title="Error distribution", 
                    xaxis=dict(title="Error delta between predicted and real data [AC Power]"),
                    yaxis=dict(title="Data point counts"))

    figure_file = abspath(join(outputs_train_path, "Error_Distribution.png"))
    fig.write_image(figure_file)

    X_pred = model.predict(X_test)
    X_pred = X_pred.reshape(X_pred.shape[0], X_pred.shape[2])
    X_pred = scaler.inverse_transform(X_pred)
    X_pred = pd.DataFrame(X_pred, columns=train.columns)
    X_pred.index = test.index

    scores = X_pred
    scores['datetime'] = df_timestamp.loc[1893:]
    scores['real AC'] = test['AC_POWER']
    scores["loss_mae"] = (scores['real AC'] - scores['AC_POWER']).abs()
    scores['Threshold'] = 200
    scores['Anomaly'] = np.where(scores["loss_mae"] > scores["Threshold"], 1, 0)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=scores['datetime'], 
                            y=scores['loss_mae'], 
                            name="Loss"))
    fig.add_trace(go.Scatter(x=scores['datetime'], 
                            y=scores['Threshold'],
                            name="Threshold"))

    fig.update_layout(title="Error Timeseries and Threshold", 
                    xaxis=dict(title="DateTime"),
                    yaxis=dict(title="Loss"))
    
    figure_file = abspath(join(outputs_train_path, "Threshold.png"))
    fig.write_image(figure_file)

    scores['Anomaly'].value_counts()
    anomalies = scores[scores['Anomaly'] == 1][['real AC']]
    anomalies = anomalies.rename(columns={'real AC':'anomalies'})
    scores = scores.merge(anomalies, left_index=True, right_index=True, how='left')

    anomaly_csv_file = abspath(join(outputs_train_path, "anomalies.csv"))
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

    figure_file = abspath(join(outputs_train_path, "Anomaly.png"))
    fig.write_image(figure_file)

if __name__ == "__main__":
    train()