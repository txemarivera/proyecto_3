import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from datetime import datetime, timedelta

from modules.ml_func import *



def ml_app():
    
    st.subheader(body = "Trading Model :chart_with_upwards_trend:")

    st.markdown(body = """---""")
    st.markdown(body = """Use the sidebar to try our model.""")
    
    df_trading_input,data_news_sentiment = read_data()

 
     # Modelo y Scalers

    model_trading, model_glove_lstm = load_models()
    model_trading.compile(optimizer = "adam", loss = "mse")     

    X_scaler, y_scaler, scaler = load_scaler() 

   

    df_trading_input['date'] = pd.to_datetime(df_trading_input['date'])

   
    min_date = df_trading_input['date'].min().to_pydatetime()
    max_date = df_trading_input['date'].max().to_pydatetime()-timedelta(days=1)

    # Slider de fecha en la barra lateral
    selected_date = st.sidebar.slider(label      =  "Select a date",
                            min_value   =  min_date,
                            max_value   =  max_date,
                            value       =  min_date,  # Valor inicial
                            step        =  timedelta(days=1))

    # Mostrar la fecha seleccionada
    st.write("Selected Date:", selected_date)

    # User Data Input
    datos_ = df_trading_input[df_trading_input["date"] > pd.Timestamp(selected_date)]
    datos = datos_.drop(['date'], axis=1)
    data_scaled = scaler.transform(datos)
    
    # User Data Input - Display
    col1, col2 = st.columns([1, 1])

#########
    # max_date = max_date.strftime('%Y-%m-%d')
    # df_fin = actualizar_mod(max_date)

    
    # # df_pred["Financial_Sector_Close"] = prediction_scaled
    # col1.markdown(body = "prueba")
    # col1.dataframe(data = df_fin)
############


    # User Data Input - Display
    df_pred = pd.DataFrame(data = df_trading_input, columns = df_trading_input.columns)
    
    col1.markdown(body = "User's Input:")
    col1.dataframe(data = df_pred.iloc[:, :-1])

    col2.markdown(body = "User's Prediction:")
    col2.dataframe(data = df_pred.iloc[:, -1])

    # Predicción

    X,y = X_y_generator(data_scaled)
      
    len_train = int(0.9*len(datos))        
    y_pred = list()

    i = len_train
    validation_target = y[len_train:]
    fig_1 = False
    fig_2 = False

    if st.button(label = "Run Model", key = "submit1"):  

        while len(y_pred) < len(validation_target):
            
            # Predice el siguiente valor de X[i]
            p = model_trading.predict((X[i]).reshape(1, -1, data_scaled.shape[1]))
            i += 1
            y_pred.append(p)

        y_pred = np.array(y_pred)
        validation_predictions = y_scaler.inverse_transform(y_pred.reshape(-1, 1))
        validation_target = y_scaler.inverse_transform(validation_target.reshape(-1, 1))    

        # Grafico     
        df_ = pd.DataFrame({
            'index': np.arange(len(validation_target)),
            'forecast target': validation_target.flatten(),
            'forecast prediction': validation_predictions.flatten()
        })

        # Crear un gráfico de líneas con Plotly Express
        fig_1 = px.line(df_, x='index', y=['forecast target', 'forecast prediction'], 
                    labels={'value': 'USD$', 'index': 'Time'}, 
                    title='Forecast vs Prediction',
                    color_discrete_map={'forecast target': 'red', 'forecast prediction': 'blue'})

    # Mostrar el gráfico
   
    # Plots
    if fig_1:    
        col1.plotly_chart(figure_or_data = fig_1, use_container_width = True)
    if fig_2:    
        col2.plotly_chart(figure_or_data = fig_2, use_container_width = True)
    
    #   Actualización del modelo
    
    if st.button(label = "Actualizar", key = "submit2", type = "primary"):
        st.markdown("""
        ### La actualización conlleva:
        - Descarga de los indicadores económicos y del valor del 'Financial_Sector_Close' 
        - Descarga de las últimas noticias financieras
        - Análisis de sentimiento mediante el modelo 'model_glove_lstm'
        - Incorporación de estos datos al Dataframe inicial
        """)
        st.markdown("""
        Este proceso puede tardar unos minutos.
        
        - Nota 1: Las API-Keys de las distintas fuentes de consulta tienen 
            una duración determinada y podría ser necesaria su actualización manual.
        - Nota 2: Los datos se actualizarán al refrescar la página.                    
        """)
        
        if st.button(label = "Continuar", key = "submit3"):
            if max_date < datetime.today():
                max_date = max_date.strftime('%Y-%m-%d')
                actualizar_mod(max_date)
            else: st.write("A fecha actual los datos ya están actualizados")    
        else: st.button(label = "Volver", key = "submit4")
        


    # DataFrame
    with st.expander(label = "DataFrame", expanded = False):
        st.dataframe(datos_)
        st.markdown(body = download_file(df = datos_), unsafe_allow_html = True)




