import streamlit as st
import pandas as pd
import pydeck as pdk
import plotly.express as px


data = pd.read_csv('train.csv', index_col=[0])
data = data.drop(['key'], axis=1)
data = data.rename({'long': 'lon'}, axis=1)
data['atm_group'] = data['atm_group'].astype(int)
data = data.dropna()


st.title("Разведочный :red[анализ] данных")
#st.subheader("В исходный датасет мы добавили признаки: ") # будет позже
eda, model = st.tabs(["EDA", "Модель"])

with eda:
    st.sidebar.title("Фильтры")
    st.sidebar.info("Дэшборд для визуализации результатов нашей работы по проекту для МОВС23")
    st.sidebar.info("Репозиторий с исходным "
                    ":red[[кодом](https://github.com/Semyon-Yakovlev/Project_ATM/tree/55db5f201f5b1edd24c13505e8c48e61acd097da)].")
    n_mall = st.sidebar.slider("Количество тц рядом", data['n_mall'].min(), data['n_mall'].max(), (data['n_mall'].min(), data['n_mall'].max()))
    n_bank = st.sidebar.slider("Количество банков рядом", data['n_bank'].min(), data['n_bank'].max(), (data['n_bank'].min(), data['n_bank'].max()))
    n_alcohol = st.sidebar.slider("Количество магазинов алкоголя рядом", data['n_alcohol'].min(), data['n_alcohol'].max(), (data['n_alcohol'].min(), data['n_alcohol'].max()))
    data = data[data['n_mall'].between(n_mall[0], n_mall[1])]
    data = data[data['n_bank'].between(n_bank[0], n_bank[1])]
    data = data[data['n_alcohol'].between(n_alcohol[0], n_alcohol[1])]

    container = st.container(border=True)
    container.subheader('Для начала посмотрим на распределение наших банкоматов по карте России', divider='rainbow')
    container.map(data=data)
    container.write('Видно, что наибольшая концентрация банкоматов наблюдается в Москве и Санкт-Петербурге, но в целом банкоматы достаточно равномерно распределены по густонаселенным городам России')


    container = st.container(border=True)
    container.subheader('Посмотрим на числовые характеристики столбцов')
    container.write(data.describe())

    container = st.container(border=True)
    container.subheader('Далее посмотрим корреляции')
    corr = px.imshow(data.drop(['address','address_rus'], axis=1).corr())
    container.plotly_chart(corr)
    container.write('Приличная корреляция с таргетом есть только у столбца atm_group')
    container.scatter_chart(
        data,
        x='target',
        y='atm_group'
    )
    container.write('На scatter_plot это не выглядит особо красиво, тк atm_group - категориальный')

    table, text, graph = container.columns([0.25, 0.25, 0.5])
    table.subheader(f'Групп всего :blue[{data.atm_group.value_counts().shape[0]}]')
    table.write(data.atm_group.value_counts())
    chart_data = data[['lon', 'lat', 'atm_group']].copy()
    colors = ['#0000ff', '#008000', '#ff0000', '#00bfbf', '#bf00bf', '#bfbf00', '#17becf', '#ff7f0e']
    colors_dict = []
    for i,el in enumerate(chart_data['atm_group'].value_counts().index.tolist()):
        replace_dict = {el: colors[i]}
        colors_dict.append({'color': colors[i], 'atm_group': el})
        chart_data['atm_group'] = chart_data['atm_group'].replace(replace_dict)
    text.write(f'Покрасим точки на карте в соответствии с atm_group')
    text.write(pd.DataFrame(colors_dict).set_index('atm_group'))
    graph.map(chart_data, color='atm_group')
    container.write('atm_group это id банка, к которому относится банкомат, на карте видно, что некий банк с id 5478 как')
    st.subheader(f'Далее мы приступили к feature engineering и обучению моделей, продолжение на листе :red[Модель]')











