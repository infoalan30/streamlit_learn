import numpy as np
import altair as alt
import pandas as pd
import streamlit as st
import plotly.express as px
st.header('st.write')

# 样例 1

st.write('Hello, *World!* :apple:')

# 样例 2


if "df2" not in st.session_state:
    st.session_state.df2 = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])


df2 = pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])

@st.cache_data
def create_dataframe():
    return pd.DataFrame(np.random.randn(200, 3), columns=['a', 'b', 'c'])

# 使用缓存函数创建 DataFrame
df3 = create_dataframe()



yyy = st.session_state.df2
yyy = df3


c = alt.Chart(yyy).mark_circle().encode(
     x='a', y='b', size='c', color='c', tooltip=['a', 'b', 'c'])
st.write(c)
# 将 c 的绝对值归一化到合理范围（比如 10 到 100）
size_scaled = (yyy['c'] - yyy['c'].min()) / (yyy['c'].max() - yyy['c'].min()) * 90 + 10

fig = px.scatter(
 yyy,
 x='a',
 y='b',
 size=size_scaled, # 使用归一化后的值
 color='c', # 用颜色区分正负
 hover_data=['a', 'b', 'c'],
 color_continuous_scale='RdBu', # 红蓝渐变表示正负
)
st.write(fig)

from datetime import time, datetime
appointment = st.slider(
     "Schedule your appointment:",
     value=(time(11, 30), time(12, 45)))
st.write("You're scheduled for:", appointment)
