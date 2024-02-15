import pandas as pd
import streamlit.components.v1 as components
import streamlit as st
from pygwalker.api.streamlit import init_streamlit_comm, get_streamlit_html

st.set_page_config(
    page_title="Dataset Visualization",
    layout="wide"
)

st.title("Data Visualization")


option = st.selectbox(
   "Which dataset would you like to choose to visualize?",
   ("Contraceptive-method-choice", "Fertility", "Glass Identification", "Haberman-s-survival", "Iris", "Parkinsons", "Seeds", "Wine", "Zoo"),
   index=None,
   placeholder="Select dataset...(Glass identification by default)",
)

def setPaths(option):
    d = {
        "Contraceptive-method-choice" : "dataset/contraceptive+method+choice/cmc.data", 
        "Fertility" : "dataset/fertility/fertility_Diagnosis.txt",         
        "Glass Identification" : "dataset/glass+identification/glass.data", 
        "Haberman-s-survival" : "dataset/haberman+s+survival/haberman.data", 
        "Iris" : "dataset/iris/iris.data", 
        "Parkinsons" : "dataset/parkinsons/parkinsons.data", 
        "Seeds" : "dataset/seeds/seeds_dataset.txt", 
        "Wine" : "dataset/wine/wine.data", 
        "Zoo" : "dataset/zoo/zoo.data"
    }
    for key, value in d.items():
        if option == key:
            return value
        
# print(option)
filename = setPaths(option)
# print(filename)

if option:
    # Initialize pygwalker communication
    init_streamlit_comm()

    # When using `use_kernel_calc=True`, you should cache your pygwalker html, if you don't want your memory to explode
    @st.cache_resource
    def get_pyg_html(df: pd.DataFrame) -> str:
        # When you need to publish your application, you need set `debug=False`,prevent other users to write your config file.
        # If you want to use feature of saving chart config, set `debug=True`
        html = get_streamlit_html(df, spec="./gw0.json", use_kernel_calc=True, debug=False)
        return html

    # @st.cache_data - pygwalker should get updated with the latest dataset option user selects from the dropdown 
    def get_df() -> pd.DataFrame:
        # return pd.read_csv("dataset/glass+identification/glass.data", sep=',')
        return pd.read_csv(filename, sep=',')

    df = get_df()

    components.html(get_pyg_html(df), width=1300, height=1000, scrolling=True)
