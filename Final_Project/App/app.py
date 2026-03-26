import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from functools import reduce

sns.set(style="whitegrid")

st.set_page_config(
    page_title="Life Expectancy – EDA & Prediction",
    layout="wide",
)

# ------------------------
# 1. Load model & features
# ------------------------
@st.cache_resource
def load_model_and_features():
    model = joblib.load("rf_model_all_features.pkl")
    feature_cols = joblib.load("feature_cols_all_features.pkl")
    return model, feature_cols

model, feature_cols = load_model_and_features()


# ------------------------
# 2. Load & prepare data (same logic as notebook)
# ------------------------
@st.cache_data
def load_raw_data():
    obesity_raw = pd.read_csv("obesity_adults.csv")
    health_raw = pd.read_csv("health_nutrition.csv")
    pop_raw = pd.read_csv("country_population.csv")
    return obesity_raw, health_raw, pop_raw


def prepare_obesity(df, start_year=1985, end_year=2015):
    df = df.copy()
    df = df.rename(columns={"Unnamed: 0": "Country Name"})
    df = df.iloc[3:].reset_index(drop=True)

    year_cols = [c for c in df.columns if c.isdigit()]
    year_cols = [c for c in year_cols if start_year <= int(c) <= end_year]

    cols = ["Country Name"] + sorted(year_cols)
    df = df[cols]

    for c in year_cols:
        df[c] = (
            df[c]
            .astype(str)
            .str.extract(r"([\d\.]+)", expand=False)
            .astype(float)
        )

    long_df = df.melt(
        id_vars="Country Name",
        var_name="Year",
        value_name="Obesity_Adult_BMI30plus",
    )
    long_df["Year"] = long_df["Year"].astype(int)
    return long_df


def extract_indicator(df, indicator_name, value_name,
                      start_year=1985, end_year=2015):
    sub = df[df["Indicator Name"] == indicator_name].copy()
    year_cols = [str(y) for y in range(start_year, end_year + 1)]
    cols = ["Country Name", "Country Code"] + year_cols
    sub = sub[cols]

    long = sub.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name=value_name,
    )
    long["Year"] = long["Year"].astype(int)
    return long


def prepare_population(df, start_year=1985, end_year=2015):
    sub = df[df["Indicator Name"] == "Population, total"].copy()
    year_cols = [str(y) for y in range(start_year, end_year + 1)]
    cols = ["Country Name", "Country Code"] + year_cols
    sub = sub[cols]

    long = sub.melt(
        id_vars=["Country Name", "Country Code"],
        value_vars=year_cols,
        var_name="Year",
        value_name="Population_Total",
    )
    long["Year"] = long["Year"].astype(int)
    return long


@st.cache_data
def build_dataset():
    obesity_raw, health_raw, pop_raw = load_raw_data()

    indicator_names = {
        "Life expectancy at birth, total (years)": "LifeExpectancy",
        "Mortality rate, under-5 (per 1,000)": "Under5_Mortality",
        "Fertility rate, total (births per woman)": "FertilityRate",
        "Health expenditure, total (% of GDP)": "HealthExp_pct_GDP",
        "Health expenditure per capita (current US$)": "HealthExp_perCapita_USD",
        "GNI per capita, Atlas method (current US$)": "GNI_perCapita_Atlas_USD",
    }

    health_frames = []
    for ind_name, value_name in indicator_names.items():
        health_frames.append(
            extract_indicator(health_raw, ind_name, value_name,
                              start_year=1985, end_year=2015)
        )

    health = reduce(
        lambda left, right: pd.merge(
            left, right,
            on=["Country Name", "Country Code", "Year"],
            how="outer",
        ),
        health_frames,
    )

    population = prepare_population(pop_raw)
    obesity = prepare_obesity(obesity_raw)

    data = (
        health
        .merge(population, on=["Country Name", "Country Code", "Year"], how="left")
        .merge(obesity, on=["Country Name", "Year"], how="left")
    )

    # Keep rows with target
    model_df = data[~data["LifeExpectancy"].isna()].copy()

    # Same log transforms as in the notebook
    for col in ["HealthExp_perCapita_USD",
                "GNI_perCapita_Atlas_USD",
                "Population_Total"]:
        model_df[f"log_{col}"] = np.log1p(model_df[col])

    return data, model_df


data, model_df = build_dataset()
target_col = "LifeExpectancy"


# ------------------------
# 3. Helper for prediction
# ------------------------
def predict_life_expectancy(input_dict):
    """
    input_dict: dict with keys = feature_cols, values = floats
    """
    X = pd.DataFrame([input_dict])[feature_cols]
    return model.predict(X)[0]


# ------------------------
# 4. UI – Intro + EDA + What-if prediction
# ------------------------
st.title("Life Expectancy – EDA & Prediction (Full Model)")

tab_intro, tab1, tab2 = st.tabs(["🏠 Introduction", "📊 EDA", "🔮 What-if prediction"])

# ---- Tab 0: Introduction ----
with tab_intro:
    st.subheader("Welcome to the Life Expectancy Explorer 👋")

    st.markdown(
        """
        This app explores how **health, demographics and economic factors** are related  
        to **life expectancy** across countries and years.

        The underlying data comes from:
        - World Bank health, nutrition and population statistics  
        - A dataset on adult obesity prevalence  
        - Country population data  

        We trained a **Random Forest regression model** that predicts
        _life expectancy at birth (years)_ using:
        - **Under-5 mortality** (deaths per 1,000 live births)
        - **Fertility rate** (births per woman)
        - **Health expenditure** (% of GDP and per capita in US$)
        - **Income level** (GNI per capita, Atlas method)
        - **Population size**
        - **Adult obesity prevalence**
        - Log-transformed versions of some monetary / population variables

        ---
        ### 🔍 What you can do in this app

        **📊 EDA tab**

        - See how global life expectancy has evolved over time  
        - Visualize the relationship between **under-5 mortality** and life expectancy  
        - Compare **top and bottom countries** by life expectancy for a given year  

        **🔮 What-if prediction tab**

        - Use sliders to set a hypothetical country's:
          - under-5 mortality, fertility, health spending, income, population and obesity
        - Get the model’s **predicted life expectancy**
        - Inspect the exact inputs used for the prediction  

        """
    )

# ---- Tab 1: EDA ----
with tab1:
    st.subheader("Global patterns")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Global average life expectancy over time**")
        global_trend = (
            model_df.groupby("Year")[target_col]
            .mean()
            .reset_index()
        )
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(global_trend["Year"], global_trend[target_col], marker="o")
        ax.set_xlabel("Year")
        ax.set_ylabel("Life expectancy (years)")
        ax.set_title("Global average life expectancy (1985–2015)")
        st.pyplot(fig)

    with col2:
        st.markdown("**Life expectancy vs under-5 mortality**")
        fig, ax = plt.subplots(figsize=(6, 4))
        sns.scatterplot(
            data=model_df,
            x="Under5_Mortality",
            y=target_col,
            alpha=0.3,
            ax=ax,
        )
        ax.set_xlabel("Under-5 mortality (per 1,000)")
        ax.set_ylabel("Life expectancy (years)")
        ax.set_title("Life expectancy vs under-5 mortality")
        st.pyplot(fig)

    st.markdown("---")

    st.subheader("Top and bottom countries by life expectancy")

    year_focus = st.slider(
        "Select year", int(model_df["Year"].min()), int(model_df["Year"].max()), 2014
    )
    snapshot = model_df[model_df["Year"] == year_focus].dropna(subset=[target_col])

    n_countries = st.slider("Number of countries", 5, 20, 10)

    top_countries = snapshot.nlargest(n_countries, target_col)
    bottom_countries = snapshot.nsmallest(n_countries, target_col)

    c1, c2 = st.columns(2)

    with c1:
        st.write(f"Top {n_countries} countries in {year_focus}")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(top_countries["Country Name"], top_countries[target_col])
        ax.invert_yaxis()
        ax.set_xlabel("Life expectancy (years)")
        st.pyplot(fig)

    with c2:
        st.write(f"Bottom {n_countries} countries in {year_focus}")
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.barh(bottom_countries["Country Name"], bottom_countries[target_col])
        ax.invert_yaxis()
        ax.set_xlabel("Life expectancy (years)")
        st.pyplot(fig)

# ---- Tab 2: What-if prediction ----
with tab2:
    st.subheader("Simulate a hypothetical count")

    df = model_df.copy()

    def slider_for(col, label=None, step=None):
        if col not in df.columns:
            return None
        if label is None:
            label = col
        series = df[col].dropna()
        min_val = float(series.quantile(0.01))
        max_val = float(series.quantile(0.99))
        default = float(series.median())
        if step is None:
            step = (max_val - min_val) / 100.0
        return st.slider(label, min_val, max_val, default, step=step)

    inputs = {}

    st.markdown("**Health, demographic & economic inputs**")

    if "Under5_Mortality" in feature_cols:
        inputs["Under5_Mortality"] = slider_for(
            "Under5_Mortality", "Under-5 mortality (per 1,000)", step=1.0
        )

    if "FertilityRate" in feature_cols:
        inputs["FertilityRate"] = slider_for(
            "FertilityRate", "Fertility rate (births per woman)", step=0.1
        )

    if "HealthExp_pct_GDP" in feature_cols:
        inputs["HealthExp_pct_GDP"] = slider_for(
            "HealthExp_pct_GDP", "Health expenditure (% of GDP)", step=0.1
        )

    if "HealthExp_perCapita_USD" in feature_cols:
        inputs["HealthExp_perCapita_USD"] = slider_for(
            "HealthExp_perCapita_USD", "Health expenditure per capita (US$)", step=10.0
        )

    if "GNI_perCapita_Atlas_USD" in feature_cols:
        inputs["GNI_perCapita_Atlas_USD"] = slider_for(
            "GNI_perCapita_Atlas_USD", "GNI per capita (Atlas, US$)", step=50.0
        )

    if "Population_Total" in feature_cols:
        inputs["Population_Total"] = slider_for(
            "Population_Total", "Population (total)", step=1_000_000.0
        )

    if "Obesity_Adult_BMI30plus" in feature_cols:
        inputs["Obesity_Adult_BMI30plus"] = slider_for(
            "Obesity_Adult_BMI30plus", "Adult obesity prevalence (%)", step=0.5
        )

    # Derived log features
    if "log_HealthExp_perCapita_USD" in feature_cols:
        inputs["log_HealthExp_perCapita_USD"] = np.log1p(inputs["HealthExp_perCapita_USD"])
    if "log_GNI_perCapita_Atlas_USD" in feature_cols:
        inputs["log_GNI_perCapita_Atlas_USD"] = np.log1p(inputs["GNI_perCapita_Atlas_USD"])
    if "log_Population_Total" in feature_cols:
        inputs["log_Population_Total"] = np.log1p(inputs["Population_Total"])

    st.markdown("---")

    if st.button("Predict life expectancy"):
        pred = predict_life_expectancy(inputs)
        st.success(f"Predicted life expectancy: **{pred:.1f} years**")

        st.markdown("**Inputs used:**")
        st.json(inputs)