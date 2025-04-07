import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
st.header('st.write')

# 样例 1

st.write('Hello, *World!* :apple:')

# 样例 2

df2 = pd.DataFrame(
     np.random.randn(200, 3),
     columns=['a', 'b', 'c'])
c = alt.Chart(df2).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)
# 将 c 的绝对值归一化到合理范围（比如 10 到 100）
size_scaled = (df2['c'] - df2['c'].min()) / (df2['c'].max() - df2['c'].min()) * 90 + 10

fig = px.scatter(
 df2,
 x='a',
 y='b',
 size=size_scaled, # 使用归一化后的值
 color='c', # 用颜色区分正负
 hover_data=['a', 'b', 'c'],
 color_continuous_scale='RdBu', # 红蓝渐变表示正负
)
st.write(fig)
