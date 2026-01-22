import streamlit as st

# 这里设置网页的标题和图标
st.set_page_config(page_title="行业择时面板", layout="wide")

# 定义所有的页面路径和对应的中文名称
# 注意：路径要相对于 app.py 的位置
pages = {
    "行业择时面板": [
        st.Page("Coal_Panel.py", title="动力煤面板")
    ]
}

# 启动导航栏
pg = st.navigation(pages)
pg.run()