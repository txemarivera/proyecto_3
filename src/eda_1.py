import io
import os

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from wordcloud import WordCloud

def eda_1_app():

    st.title('Análisis de Sentimientos de las Noticias')
    st.write("""
        Este código implementa una aplicación en Streamlit para realizar análisis de sentimientos en un 
        conjunto de noticias. Carga datos, realiza análisis exploratorio, visualiza distribuciones, filtra por
        sentimientos y procesa texto usando stemmer y lemmatizer. La finalidad es ofrecer una herramienta
        interactiva para analizar y visualizar sentimientos en noticias.
    """)

    df = pd.read_csv("./data/inputs/data_news_sentiment_prepro.csv")
    df.drop_duplicates("Sentence", inplace=True)
    df = df.dropna()

    # Pedir al usuario las categorías a analizar en el EDA
    categoria = st.multiselect(label='Filtra los sentimientos para visualizar los datos:', options=set(df.Sentiment.values))

    # Comprobar que el usuario haya seleccionado algo
    if len(categoria)>0:
        # Filtramgitos el df a petición del usuario
        df_eda = df[df.Sentiment.isin(categoria)]

        # 1. Mostramos las primeras filas del df a estudiar
        st.write(df_eda.head(3))

        # Total de noticias
        st.write(f'Total de noticias para las categorías seleccionadas: {len(df_eda)}.')

        # Realizamos el conteo de palabras
        df_eda['word_count'] = df_eda['Sentence'].apply(lambda x: len(str(x).split()))

        # 2. Ploteamos un histograma de conteos de palabras
        st.subheader('Distribución de la Longitud de Textos')
        st.write("""
            El histograma muestra la distribución de la longitud de los textos en el dataset. La mayoría de los
            textos tienen una longitud entre 30 y 70 caracteres, con un pico notable alrededor de los 40
            caracteres. La frecuencia disminuye gradualmente para textos más largos, siendo raros los textos que 
            superan los 150 caracteres. Esto indica que la mayoría de las entradas son relativamente cortas. Indica Bins a visualizar:
        """)

        bins = st.number_input(label='Bins', min_value=10, max_value=110, step=10)
        fig, ax = plt.subplots()
        plt.hist(df_eda['word_count'], bins=bins, color='#4682B4', alpha=0.7)
        plt.title('Distribución de la Longitud de Textos')
        plt.xlabel('Longitud del Texto')
        plt.ylabel('Frecuencia')
        st.pyplot(fig)

        # 3. Histograma de frecuencia de categorías
        st.subheader('Total de noticias de cada categoría')

        # Contar ocurrencias de cada etiqueta
        counts = df_eda['Sentiment'].value_counts().sort_values(ascending=False)

        # Crear el gráfico de barras con colores gradientes de azul
        colors = ['#0d47a1', '#1976d2', '#64b5f6']  # De más oscuro a más claro

        # Crear el gráfico de barras
        fig, ax = plt.subplots()
        ax = counts.plot(kind='bar', color=colors, figsize=(7, 4))

        # Añadir título y etiquetas
        plt.title('Sentimiento de las noticias')
        plt.xlabel('Etiqueta')
        plt.ylabel('Cantidad')

        # Anotar cada barra con el valor exacto
        for i in ax.patches:
            ax.text(i.get_x() + i.get_width() / 2, i.get_height() + 5, str(int(i.get_height())), ha='center', va='bottom')

        # Mostrar el gráfico
        st.pyplot(fig)

        # 4. Longitudes según categoría
        st.subheader('Total de palabras para cada categoría')
        fig, ax = plt.subplots()
        boxplot = df_eda.boxplot(column='word_count', by='Sentiment', patch_artist=True, ax=ax)
        plt.title('Boxplot de longitudes de texto por sentimiento')
        plt.suptitle('')
        plt.xlabel('Etiqueta')
        plt.ylabel('Longitud del Texto')
        st.pyplot(fig)

        # 5. Estudio de la frecuencia de palabras
        st.subheader('Palabras más/menos frecuentes')
        st.write("""
            Filtra las palabas más frecuentes negativas y positivas.
        """)
        aparicion_min = st.number_input(label='Al menos la palabra debe aparecer...:', min_value=1, max_value=110, step=10)
        top_mas = st.number_input(label='Top_Más', min_value=10, max_value=110, step=10)
        top_menos = st.number_input(label='Top_Menos', min_value=10, max_value=110, step=10)
        fig = plt.figure(figsize=(14,10)) 
        axs = fig.subplots(len(categoria), 2) 
        for i, cat in enumerate(categoria):
            df_counts = pd.read_csv(f'./data/inputs/word_counts_{cat}.csv')
            df_counts = df_counts[df_counts.Count>=aparicion_min]
            df_mas = df_counts.head(top_mas)
            df_menos = df_counts.tail(top_menos)
            axs[i,0].set_title(f'Palabras más frecuentes de {cat}')
            df_mas.plot('Word', 'Count', kind='barh', color='#4682B4', ax=axs[i,0], yticks=[])
            axs[i,1].set_title(f'Palabras menos frecuentes de {cat}')
            df_menos.plot('Word', 'Count', kind='barh', color='#4682B4', ax=axs[i,1], yticks=[])
        st.pyplot(fig)

        # 6. WordCloud
        st.subheader('Nube de palabras')
        st.write("""
            Visualiza una nube de palabras con todo el df completo. ¿Cuántas quieres visualizar?
        """)
        max_words = st.number_input(label='Máximo de palabras', min_value=10, max_value=500, step=10)
        fig1, ax = plt.subplots()
        wordcloud = WordCloud(background_color='white', max_words=max_words)
        wordcloud.generate(' '.join(df.Sentence_prepro.values))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis("off")
        st.pyplot(fig1)