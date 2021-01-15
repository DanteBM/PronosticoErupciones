import os
from datetime import timedelta

import streamlit as st
import numpy as np
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go

import librosa
import lightgbm as lgb  # Light Gradient Boosting Machine model

import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

from scipy.signal import find_peaks, peak_prominences, periodogram, peak_widths

def graph_amp_ply(df):
    """Graph sensor data
    Parameters:
        df: pd.DataFrame
        Dataframe with 10 columns, corresponding to data from sensors
    """
    valids = ~df.isna().all()
    valids = list(valids[valids].index)
    fig = make_subplots(rows = len(valids), cols=1, y_title = "Amplitudes", x_title = "Tiempo [cs]")
    for idx, sensor in enumerate(valids):
        data = df.loc[:, sensor]
        fig.append_trace(go.Scatter(x = list(range(df.shape[0])), y = data, 
                                    mode = "lines", name = f"{sensor}"), 
                                     row = idx+1, col = 1)

    fig.update_layout(height = 1000, width = 950, title_text = "Diagramas de amplitudes", title_font_size = 30)
    st.plotly_chart(fig, use_container_width=True)

def standardize(df):
    """Standardize columns from dataframe
    Parameters:
        df: pd.DataFrame
        Dataframe with 10 columns, corresponding to data from sensors
    Returns:
        standarized_df: pd.Dataframe
        Dataframe with standardized columns
    """
    aggs = df.agg([np.nanmean, np.nanstd]).astype("float16")
    standarized_df = (df - aggs.loc["nanmean",:])/ aggs.loc["nanstd",:]
    return standarized_df

def get_features(df):
    """Get features from sensor data
    For each sensor, peaks, promenences and periodograms features are computed.
        
    Parameters:
        df: pd.DataFrame
        Dataframe with 10 columns, corresponding to data from sensors
    Returns:
        features: list
        List with features
    """
    features = []
    # zeros_crossings
    features.extend(librosa.zero_crossings(df.values, axis = 0).sum(axis = 0))
    
    # find_peaks
    features.extend(df.apply(find_peaks, axis = 0).iloc[0,:].apply(len).values)
            
    # peak_widths_max
    λ0 = lambda x: np.max(peak_widths(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    features.extend(df.apply(λ0).values)
                
    # peak_widths_mean
    λ01 = lambda x: np.mean(peak_widths(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    features.extend(df.apply(λ01).values)
                
    # peak_prominences_max
    λ1 = lambda x: np.max(peak_prominences(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    features.extend(df.apply(λ1).values)
                
    # peak_prominences_mean
    λ11 = lambda x: np.mean(peak_prominences(x, find_peaks(x)[0])[0]) if len(find_peaks(x)[0]) != 0 else 0
    features.extend(df.apply(λ11).values)
                
    # periodogram_max
    λ2 = lambda x: np.max(periodogram(x[~x.isna()], 100)[1]) if ~x.isna().all() else 0
    features.extend(np.sqrt(df.apply(λ2).values)) # Es un estimado del RMS
    
    # periodogram_mean
    λ3 = lambda x: np.mean(periodogram(x[~x.isna()], 100)[1]) if ~x.isna().all() else 0
    features.extend(df.apply(λ3).values)
                
    return features
    
if __name__ == "__main__":
    with open("texto.txt", mode="rt") as file:
        text = file.read()

    st.set_page_config(page_title="Pronóstico de erupciones")  # Tab's title
    
    # Sidebar
    st.sidebar.title("Acerca de")
    st.sidebar.image("mount_etna.jpeg", caption="Monte Etna", use_column_width=True)
    st.sidebar.markdown(text)
    st.sidebar.image("logo_iimas_unam.jpeg")
    st.sidebar.markdown("""
    Licenciatura en Ciencia de Datos
    
    **Minería de datos**
    
    Desarrollado por:
    * Óscar Alvarado Morán
    * Dante Bermúdez Marbán""")

    # Main view
    st.title("Pronóstico de erupciones")
    
    # Requesting data
    st.write("Suba a continuación el archivo con las lecturas de los 10 sensores")
    uploaded_file = st.file_uploader("CSV de sensores")
    if uploaded_file:
        try:
            df = pd.read_csv(uploaded_file)
            if len(df.columns) != 10:
                raise ValueError("El dataframe debe tener solamente 10 columnas (una por sensor)")
            if not all("sensor" in column for column in df.columns):
                raise ValueError("Las columnas deben llevar por nombre 'sensor_#'")
        except Exception as e:
            st.error(e)
            df = None
     
        # Data loaded
        if df is not None:
            st.write("**Estadísticas:**")
            st.write(df.describe().T) # stadistics
            graph_amp_ply(df) # Visualization
            
            # Resultads
            standardized_df = standardize(df)
            features = get_features(standardized_df)
            features = np.array(features).reshape(1,-1)
            model = lgb.Booster(model_file="model.txt") # Load trained model
            time_to_eruption = model.predict(features)[0]
            seconds_to_eruption = timedelta(milliseconds=time_to_eruption*10).total_seconds()
            time_to_eruption = timedelta(seconds=round(seconds_to_eruption))
            str_time_to_eruption = str(time_to_eruption).replace("day","día")
            st.write("#### Tiempo aproximado para la próxima erupción (días, HH:MM:SS):")
            st.write(f"## {str_time_to_eruption}")
            st.balloons()
    
