import json
import os
from datetime import timedelta

import streamlit as st
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import librosa
import lightgbm as lgb            # Para el modelo LGBM

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from scipy.signal import find_peaks, peak_prominences, periodogram, peak_widths

def graficar_amp_ply(df):
    """Graficar sensores con plotly
    Parámetros:
        df: pd.DataFrame
        Dataframe con 10 columnas, correspondientes a lecturas de sensores
    """
    validos = ~df.isna().all()
    validos = list(validos[validos].index)
    fig = make_subplots(rows = len(validos), cols=1, y_title = "Amplitudes", x_title = "Tiempo [cs]")
    for idx, sensor in enumerate(validos):
        datos = df.loc[:, sensor]
        fig.append_trace(go.Scatter(x = list(range(df.shape[0])), y = datos, 
                                    mode = "lines", name = f"{sensor}"), 
                                     row = idx+1, col = 1)

    fig.update_layout(height = 1000, width = 950, title_text = "Diagramas de amplitudes", title_font_size = 30)
    st.plotly_chart(fig, use_container_width=True)

def estandarizar(df):
    """Estandariza columnas de un dataframe
    Parámetros:
        df: pd.DataFrame
        Dataframe con 10 columnas, correspondientes a lecturas de sensores
    Regresa:
        standarized_df: pd.Dataframe
        Dataframe estandarizado por columnas
    """
    aggs = df.agg([np.nanmean, np.nanstd]).astype("float16")
    standarized_df = (df - aggs.loc["nanmean",:])/ aggs.loc["nanstd",:]
    return standarized_df

def get_chars(df):
    """Obtener característica de los sensores
    Para cada sensor, se obtienen información relacionado a los picos, prominencias y periodogramas
        
    Parámetros:
        df: pd.DataFrame
        Dataframe con 10 columnas, correspondientes a lecturas de sensores
    Regresa:
        cars: list
        Lista con las características
    """
    cars = []
    # zeros_crossings
    cars.extend(librosa.zero_crossings(df.values, axis = 0).sum(axis = 0))
    
    # find_peaks
    cars.extend(df.apply(find_peaks, axis = 0).iloc[0,:].apply(len).values)
            
    # peak_widths_max
    λ0 = lambda x: np.max(peak_widths(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    cars.extend(df.apply(λ0).values)
                
    # peak_widths_mean
    λ01 = lambda x: np.mean(peak_widths(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    cars.extend(df.apply(λ01).values)
                
    # peak_prominences_max
    λ1 = lambda x: np.max(peak_prominences(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    cars.extend(df.apply(λ1).values)
                
    # peak_prominences_mean
    λ11 = lambda x: np.mean(peak_prominences(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    cars.extend(df.apply(λ11).values)
                
    # periodogram_max
    λ2 = lambda x: np.max(periodogram(x[~x.isna()], 100)[1]) if ~x.isna().all() else 0
    cars.extend(np.sqrt(df.apply(λ2).values)) # Es un estimado del RMS
    
    # periodogram_mean
    λ3 = lambda x: np.mean(periodogram(x[~x.isna()], 100)[1]) if ~x.isna().all() else 0
    cars.extend(df.apply(λ3).values)
                
    return cars
    
if __name__ == "__main__":
    with open("texto.txt", mode="rt") as file:
        texto = file.read()

    st.set_page_config(page_title="Pronóstico de erupciones")  # Título en pestaña
    
    # Sidebar
    st.sidebar.title("Acerca de")
    st.sidebar.image("mount_etna.jpeg", caption="Monte Etna", use_column_width=True)
    st.sidebar.markdown(texto)
    st.sidebar.image("logo_iimas_unam.jpeg")
    st.sidebar.markdown("""
    Licenciatura en Ciencia de Datos
    
    **Minería de datos**
    
    Desarrollado por:
    * Óscar Alvarado Morán
    * Dante Bermúdez Marbán""")

    # Principal
    st.title("Pronóstico de erupciones")
    
    # Pedir datos
    st.write("Suba a continuación el archivo con las lecturas de los 10 sensores")
    uploaded_file = st.file_uploader("CSV de sensores")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
        except Exception as e:
            st.error(e)
            df = None
     
        # Ya con los datos cargados
        if df is not None:
            st.write(df.describe()) # estadísticas
            graficar_amp_ply(df) # Visualización
            
            # Resultados
            standarized_df = estandarizar(df)
            features = get_chars(standarized_df) # obtener vector de características
            features = np.array(features).reshape(1,-1)
            model = lgb.Booster(model_file="model.txt") # Cargar modelo
            time_to_eruption = model.predict(features)[0]
            seconds_to_eruption = timedelta(milliseconds=time_to_eruption*10).total_seconds()
            time_to_eruption = timedelta(seconds=round(seconds_to_eruption))
            str_time_to_eruption = str(time_to_eruption).replace("day","día")
            st.write("#### Tiempo aproximado para la próxima erupción (días, HH:MM:SS):")
            st.write(f"## {str_time_to_eruption}")
            st.balloons()
    
