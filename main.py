# Importing ToolKits
import re
import relations
import prediction

from time import sleep
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.metrics import mean_squared_error

import streamlit as st
from streamlit.components.v1 import html
from streamlit_option_menu import option_menu
import warnings

# pd.set_option('future.no_silent_downcasting', True)
# pd.options.mode.copy_on_write = "warn"


def run():
    st.set_page_config(
        page_title="Salary Prediction",
        page_icon="📊",
        layout="wide"
    )

    warnings.simplefilter(action='ignore', category=FutureWarning)

    # Function To Load Our Dataset
    @st.cache_data
    def load_data(the_file_path):
        df = pd.read_csv(the_file_path)
        return df

    # Function To Load Our Dataset
    @st.cache_data
    def load_linear_regression_model(model_path):
        return pd.read_pickle(model_path)

    df = load_data("Salary Data.csv")

    model = load_linear_regression_model(
        "random_forest_regressor_salary_predictor_v1.pkl")


    # Function To Valid Input Data
    @st.cache_data
    def is_valid_data(d):
        letters = list("qwertyuiopasdfghjklzxcvbnm@!#$%^&*-+~")
        return len(d) >= 2 and not any([i in letters for i in list(d)])

    @st.cache_data
    def validate_test_file(test_file_columns):
        pa = """Age
Years of Experience
Gender
Education Level"""
        col = "\n".join(test_file_columns).lower()
        pattern = re.compile(pa)

        matches = pattern.findall(col)
        return len("\n".join(matches).split("\n")) == 4

    st.markdown(
        """
    <style>
         .main {
            text-align: center; 
         }
         .st-emotion-cache-1ibsh2c {
             padding-left: 3rem;
            padding-right: 3rem;
        }
         .st-emotion-cache-16txtl3 h1 {
         font: bold 29px arial;
         text-align: center;
         margin-bottom: 15px
            
         }
         div[data-testid=stSidebarContent] {
         background-color: #111;
         border-right: 4px solid #222;
         padding: 8px!important
         
         }
         div.block-containers{
            padding-top: 0.5rem
         }

         .st-emotion-cache-z5fcl4{
            padding-top: 1rem;
            padding-bottom: 1rem;
            padding-left: 1.1rem;
            padding-right: 2.2rem;
            overflow-x: hidden;
         }

         .st-emotion-cache-16txtl3{
            padding: 2.7rem 0.6rem
         }

         .plot-container.plotly{
            border: 1px solid #333;
            border-radius: 6px;
         }

         div.st-emotion-cache-1r6slb0 span.st-emotion-cache-10trblm{
            font: bold 24px tahoma
         }
         div [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-left: auto;
            margin-right: auto;
            width: 100%;
        }

        div[data-baseweb=select]>div{
            cursor: pointer;
            background-color: #111;
            border: 2px solid purple
        }

        div[data-baseweb=base-input]{
            background-color: #111;
            border: 4px solid #444;
            border-radius: 5px;
            padding: 5px
        }

        div[data-testid=stFormSubmitButton]> button{
            width: 100%;
            background-color: #111;
            border: 2px solid violet;
            padding: 18px;
            border-radius: 30px;
            opacity: 0.8;
        }
        div[data-testid=stFormSubmitButton]  p{
            font-weight: bold;
            font-size : 20px
        }

        div[data-testid=stFormSubmitButton]> button:hover{
            opacity: 1;
            border: 2px solid violet;
            color: #fff
        }


    </style>
    """,
        unsafe_allow_html=True
    )

    side_bar_options_style = {
        "container": {"padding": "0!important", "background-color": 'transparent'},
        "icon": {"color": "white", "font-size": "18px"},
        "nav-link": {"color": "white", "font-size": "16px", "text-align": "left", "margin": "0px", "margin-bottom": "15px"},
        "nav-link-selected": {"background-color": "#7B06A6", "font-size": "15px"},
    }

    sub_options_style = {
        "container": {"padding": "3!important", "background-color": '#101010', "border": "2px solid #7B06A6"},
        "nav-link": {"color": "white", "padding": "12px", "font-size": "18px", "text-align": "center", "margin": "0px", },
        "nav-link-selected": {"background-color": "#7B06A6"},

    }
    header = st.container()
    content = st.container()

    with st.sidebar:
        st.title("Prediction Apps")
        page = option_menu(
            menu_title=None,
            options=['Home', 'Relations & Correlarions',
                     'Prediction'],
            icons=['diagram-3-fill', 'bar-chart-line-fill',
                   "graph-up-arrow", "cpu"],

            menu_icon="cast",
            default_index=0,
            styles=side_bar_options_style
        )
        st.write("***")

        data_file = st.file_uploader("Upload Your Dataset 📂")
        if data_file is not None:
            if data_file.name.split(".")[-1].lower() != "csv":
                st.error("Please, Upload CSV FILE ONLY")
            else:
                df = pd.read_csv(data_file)

        # Home Page
        if page == "Home":

            with header:
                st.header('Employee Salary Prediction 📈💰')

            with content:
                st.dataframe(df.sample(frac=0.3, random_state=35).reset_index(drop=True),
                                 use_container_width=True)

                st.write("***")

                st.subheader("Data Summary Overview 🧐")

                len_numerical_data = df.select_dtypes(
                    include="number").shape[1]
                len_string_data = df.select_dtypes(include="object").shape[1]

                if len_numerical_data > 0:
                    st.subheader("Numerical Data [123]")

                    data_stats = df.describe().T
                    st.table(data_stats)

                if len_string_data > 0:
                    st.subheader("String Data [𝓗]")

                    data_stats = df.select_dtypes(
                        include="object").describe().T
                    st.table(data_stats)

        # Relations & Correlations
        if page == "Relations & Correlarions":

            with header:
                st.header("Correlations Between Data 📉🚀")

            with content:
                st.plotly_chart(relations.create_heat_map(df),
                                use_container_width=True)

                st.plotly_chart(relations.create_scatter_matrix(
                    df), use_container_width=True)

                st.write("***")
                col1, col2 = st.columns(2)
                with col1:
                    first_feature = st.selectbox(
                        "First Feature", options=(df.select_dtypes(
                            include="number").columns.tolist()), index=0).strip()

                temp_columns = df.select_dtypes(
                    include="number").columns.to_list().copy()

                temp_columns.remove(first_feature)

                with col2:
                    second_feature = st.selectbox(
                        "First Feature", options=(temp_columns), index=0).strip()

                st.plotly_chart(relations.create_relation_scatter(
                    df, first_feature, second_feature), use_container_width=True)

        if page == "Prediction":
            with header:
                st.header("Prediction Model 💰🔥")
                prediction_option = option_menu(menu_title=None, options=["One Value", 'From File'],
                                                icons=[" "]*2, menu_icon="cast", default_index=0,
                                                orientation="horizontal", styles=sub_options_style)

            with content:
                if prediction_option == "One Value":
                    with st.form("Predict_value"):

                        c1, c2 = st.columns(2)
                        with c1:
                            age = st.number_input(
                                'Employee Age', min_value=20, max_value=60, value=24)

                        with c2:
                            exp_year = st.number_input(

                                'Experirnce Years', min_value=0, max_value=30, value=2)

                        education_level = st.selectbox(
                            "Education Level", options=["Bachelor's", "Master's", "PhD"])

                        st.write("")  # Space

                        predict_button = st.form_submit_button(
                            label='Predict', use_container_width=True)

                        st.write("***")  # Space

                        if predict_button:
                            education = [0, 0]  # Bachelor's

                            if education_level == "Master's":
                                education = [1, 0]

                            elif education_level == "PhD":
                                education = [0, 1]

                            with st.spinner(text='Predict The Value..'):
                                new_data = [age, exp_year]
                                new_data.extend(education)

                                predicted_value = f"{model.predict([new_data])[0]:,.0f}"
                                sleep(1.2)

                                predicted_col, score_col = st.columns(2)

                                with predicted_col:
                                    st.image("imgs/money.png",
                                             caption="", width=70)

                                    st.subheader("Expected Salary")
                                    st.subheader(f"${predicted_value}")

                                with score_col:
                                    st.image("imgs/accuracy.png",
                                             caption="", width=70)
                                    st.subheader("Model Accuracy")
                                    st.subheader(f"{np.round(91.85, 2)}%")

                if prediction_option == "From File":
                    st.warning("Please upload your file with the following columns' names in the same order\n\
                            ['Age', 'Years of Experience', 'Education Level']", icon="⚠️")

                    test_file = st.file_uploader("Upload Your Test File 📂")

                    if test_file is not None:
                        extention = test_file.name.split(".")[-1]
                        if extention.lower() != "csv":
                            st.error("Please, Upload CSV FILE ONLY")
                        else:
                            X_test = pd.read_csv(test_file)
                            X_test.dropna(inplace=True)

                            if validate_test_file(X_test.columns.to_list()):
                                X_test_encodded = pd.get_dummies(
                                    X_test, columns=["Education Level"], drop_first=True) * 1
                                st.info(X_test_encodded.columns)

                            else:
                                X_test = X_test[[
                                    'Age', 'Years of Experience', 'Education Level']]

                                X_test_encodded = pd.get_dummies(
                                    X_test, columns=["Education Level"], drop_first=True) * 1

                                st.info(X_test_encodded.columns)

                            all_predicted_values = model.predict(
                                X_test_encodded)

                            final_complete_file = pd.concat([X_test, pd.DataFrame(all_predicted_values,
                                                                                  columns=["Predicted Salary"])], axis=1)

                            st.write("")

                            st.dataframe(final_complete_file,
                                         use_container_width=True,)

                    with st.form("comaprison_form"):

                        if st.form_submit_button("Compare Predicted"):
                            st.info(
                                "Be Sure Your Actual Values File HAS ONLY One Column (Actual Salary)", icon="ℹ️")

                            actual_file = st.file_uploader(
                                "Upload Your Actual Salary File 📂")

                            if actual_file is not None and test_file is not None:
                                if actual_file.name.split(".")[-1].lower() != "csv":
                                    st.error("Please, Upload CSV FILE ONLY")
                                else:

                                    y_test = pd.read_csv(
                                        actual_file).iloc[:, -1]
                                    y_test.dropna(inplace=True)

                                    col1, col2 = st.columns(2)

                                    with col1:
                                        test_score = np.round(
                                            model.score(X_test_encodded, y_test) * 100, 2)
                                        prediction.creat_matrix_score_cards("imgs/accuracy.png",
                                                                            "Prediction Accuracy",
                                                                            test_score,
                                                                            True
                                                                            )

                                    with col2:
                                        mse = mean_squared_error(
                                            y_test, all_predicted_values)
                                        prediction.creat_matrix_score_cards("imgs/question.png",
                                                                            "Error Ratio", f'{np.sqrt(mse):,.2f}', False)

                                    predicted_df = prediction.create_comparison_df(
                                        y_test, all_predicted_values)

                                    st.dataframe(
                                        predicted_df, use_container_width=True, height=300)

                                    st.plotly_chart(prediction.create_residules_scatter(predicted_df),
                                                    use_container_width=True)

                            else:
                                st.warning(
                                    "Please, Check That You Upload The Test File & Actual Value", icon="⚠️")


run()

