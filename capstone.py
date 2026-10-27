from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent
FILES = {
    "Confirmed cases": "time_series_covid19_confirmed_global.csv",
    "Deaths": "time_series_covid19_deaths_global.csv",
}

st.set_page_config(page_title="COVID-19 Country Explorer", layout="wide")


@st.cache_data

def load_data(filename):
    path = DATA_DIR / filename
    data = pd.read_csv(path)
    country_col = "Country/Region" if "Country/Region" in data.columns else "Country_Region"
    date_cols = [col for col in data.columns if pd.notna(pd.to_datetime(col, format="%m/%d/%y", errors="coerce"))]

    data[date_cols] = data[date_cols].apply(pd.to_numeric, errors="coerce")
    totals = data.groupby(country_col)[date_cols].sum(min_count=1)
    totals.columns = pd.to_datetime(date_cols, format="%m/%d/%y")

    result = totals.rename_axis("country").reset_index().melt(
        id_vars="country", var_name="date", value_name="cumulative"
    )
    # melt may produce an object column even with Timestamp column labels.
    result["date"] = pd.to_datetime(result["date"], errors="raise")
    result = result.dropna(subset=["cumulative"]).sort_values(["country", "date"])
    result["daily"] = result.groupby("country")["cumulative"].diff()
    first_day = result.groupby("country").cumcount() == 0
    result.loc[first_day, "daily"] = result.loc[first_day, "cumulative"]
    result["seven_day_average"] = result.groupby("country")["daily"].transform(
        lambda values: values.rolling(7, min_periods=1).mean()
    )
    return result


st.title("COVID-19 Country Explorer")
st.caption("Johns Hopkins University CSSE · January 2020–March 2023")

try:
    datasets = {name: load_data(filename) for name, filename in FILES.items()}
except (OSError, ValueError, KeyError) as error:
    st.error("Could not read the COVID-19 data files. Make sure both CSV files are in the same folder as capstone.py.")
    st.code(str(error))
    st.stop()

with st.sidebar:
    st.header("Options")
    measure = st.selectbox("Data", list(FILES))
    mode = st.radio("Display", ["Daily", "Cumulative"])
    countries = st.multiselect(
        "Countries",
        sorted(datasets[measure]["country"].unique()),
        default=[c for c in ["US", "India", "Brazil"] if c in datasets[measure]["country"].values],
    )
    moving_average = st.checkbox("7-day average", value=False, disabled=mode == "Cumulative")

if not countries:
    st.info("Select at least one country to view the data.")
    st.stop()

frame = datasets[measure]
start_date = frame["date"].min().date()
end_date = frame["date"].max().date()
start, end = st.slider("Date range", start_date, end_date, (start_date, end_date))

selected = frame[
    frame["country"].isin(countries)
    & frame["date"].between(pd.Timestamp(start), pd.Timestamp(end))
].copy()

column = "cumulative" if mode == "Cumulative" else "seven_day_average" if moving_average else "daily"
chart_title = f"{mode} {measure.lower()}"
if moving_average and mode == "Daily":
    chart_title += " (7-day average)"

fig = px.line(
    selected,
    x="date",
    y=column,
    color="country",
    title=chart_title,
    labels={"date": "Date", column: "Reported count", "country": "Country"},
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("Data")
st.dataframe(
    selected[["date", "country", "cumulative", "daily", "seven_day_average"]],
    use_container_width=True,
    hide_index=True,
)
st.download_button(
    "Download CSV",
    data=selected.to_csv(index=False),
    file_name="covid_country_data.csv",
    mime="text/csv",
)

st.caption(
    "Daily counts are calculated from differences in cumulative totals. "
    "Negative values are retained because they may represent reporting corrections. "
    "Country totals combine provinces and territories. The JHU CSSE series is archived and is not current."
)
