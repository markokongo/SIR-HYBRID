
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.integrate import solve_ivp
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(
    page_title="Hybrid Malaria Prediction",
    layout="wide"
)

st.title("Hybrid Mechanistic–AI Malaria Prediction Dashboard")
st.write(
    "A mosquito-human transmission model combined "
    "with Random Forest machine learning."
)

# ---------------------------------
# 1. Upload malaria surveillance data
# ---------------------------------

uploaded_file = st.file_uploader(
    "Upload malaria CSV dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info(
        "Upload a CSV containing day, cases, rainfall, "
        "temperature and humidity."
    )
    st.stop()

df = pd.read_csv(uploaded_file)

required = [
    "day", "cases", "rainfall",
    "temperature", "humidity"
]

if not all(c in df.columns for c in required):
    st.error("Your CSV is missing one or more required columns.")
    st.stop()

df = df[required].dropna().sort_values("day").reset_index(drop=True)

if len(df) < 20:
    st.error("Please upload at least 20 observations.")
    st.stop()

if (df["cases"] < 0).any():
    st.error("Malaria cases cannot be negative.")
    st.stop()

# ---------------------------------
# 2. Model parameters
# ---------------------------------

st.sidebar.header("Mechanistic model parameters")

Nh = st.sidebar.number_input(
    "Human population", 1000, 10000000, 100000
)

Nv = st.sidebar.number_input(
    "Mosquito population", 1000, 10000000, 200000
)

beta_h = st.sidebar.slider(
    "Mosquito-to-human transmission rate",
    0.001, 1.0, 0.20
)

beta_v = st.sidebar.slider(
    "Human-to-mosquito transmission rate",
    0.001, 1.0, 0.15
)

gamma_h = st.sidebar.slider(
    "Human recovery rate",
    0.01, 1.0, 0.10
)

mu_v = st.sidebar.slider(
    "Mosquito mortality rate",
    0.001, 1.0, 0.10
)

forecast_days = st.sidebar.slider(
    "Forecast period (days)",
    7, 60, 14
)

# ---------------------------------
# 3. Mechanistic SIR-mosquito model
# ---------------------------------

def malaria_model(t, y):

    Sh, Ih, Rh, Sv, Iv = y

    lambda_h = beta_h * Iv / Nh
    lambda_v = beta_v * Ih / Nh

    dSh = -lambda_h * Sh
    dIh = lambda_h * Sh - gamma_h * Ih
    dRh = gamma_h * Ih

    dSv = mu_v * Nv - lambda_v * Sv - mu_v * Sv
    dIv = lambda_v * Sv - mu_v * Iv

    return [dSh, dIh, dRh, dSv, dIv]


# Initial conditions
Ih0 = max(float(df["cases"].iloc[0]), 1.0)
Rh0 = 0.0
Sh0 = Nh - Ih0 - Rh0

if Sh0 <= 0:
    st.error("Human population must exceed initial infections.")
    st.stop()

Iv0 = 0.01 * Nv
Sv0 = Nv - Iv0

initial_conditions = [
    Sh0, Ih0, Rh0, Sv0, Iv0
]

# ---------------------------------
# 4. Solve mechanistic model
# ---------------------------------

days = len(df)

solution = solve_ivp(
    malaria_model,
    [0, days - 1],
    initial_conditions,
    t_eval=np.arange(days),
    method="RK45"
)

if not solution.success:
    st.error("Mechanistic model solution failed.")
    st.stop()

Sh, Ih, Rh, Sv, Iv = solution.y

# Daily new human infections
mechanistic_cases = []

for i in range(days):
    incidence = beta_h * Sh[i] * Iv[i] / Nh
    mechanistic_cases.append(incidence)

mechanistic_cases = np.array(mechanistic_cases)

# ---------------------------------
# 5. Hybrid AI model
# ---------------------------------

df["mechanistic_cases"] = mechanistic_cases

features = [
    "day",
    "rainfall",
    "temperature",
    "humidity",
    "mechanistic_cases"
]

X = df[features]

residual = (
    df["cases"].values - df["mechanistic_cases"].values
)

# Chronological 80:20 training and testing
split = int(0.8 * len(df))

if split < 10 or split >= len(df):
    st.error("Insufficient observations for training and testing.")
    st.stop()

X_train = X.iloc[:split]
X_test = X.iloc[split:]

y_train = residual[:split]
y_test = residual[split:]

model = RandomForestRegressor(
    n_estimators=200,
    max_depth=5,
    min_samples_leaf=2,
    random_state=42
)

model.fit(X_train, y_train)

# Predict the residual correction
train_correction = model.predict(X_train)
test_correction = model.predict(X_test)

# Hybrid predictions
hybrid_train = np.maximum(
    0, df["mechanistic_cases"].iloc[:split].values
    + train_correction
)

hybrid_test = np.maximum(
    0, df["mechanistic_cases"].iloc[split:].values
    + test_correction
)

observed_test = df["cases"].iloc[split:].values
mechanistic_test = df["mechanistic_cases"].iloc[split:].values

# ---------------------------------
# 6. Model performance
# ---------------------------------

mae = mean_absolute_error(observed_test, hybrid_test)

rmse = np.sqrt(
    mean_squared_error(observed_test, hybrid_test)
)

r2 = r2_score(observed_test, hybrid_test)

st.subheader("Model performance on the test data")

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")
c3.metric("R-squared", f"{r2:.3f}")

# Compare with mechanistic model
mechanistic_mae = mean_absolute_error(
    observed_test, mechanistic_test
)

st.write(
    f"Mechanistic model MAE: {mechanistic_mae:.2f}"
)

st.write(
    f"Hybrid model MAE: {mae:.2f}"
)

# ---------------------------------
# 7. Historical predictions
# ---------------------------------

df["hybrid_prediction"] = np.concatenate(
    [hybrid_train, hybrid_test]
)

st.subheader("Observed versus predicted malaria cases")

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["day"], df["cases"],
    label="Observed cases"
)

ax.plot(
    df["day"], df["mechanistic_cases"],
    label="Mechanistic prediction"
)

ax.plot(
    df["day"], df["hybrid_prediction"],
    label="Hybrid prediction"
)

ax.axvline(
    df["day"].iloc[split],
    linestyle="--",
    label="Train/test boundary"
)

ax.set_xlabel("Day")
ax.set_ylabel("Malaria cases")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

# ---------------------------------
# 8. Future forecast
# ---------------------------------

st.subheader("Future malaria forecasts")

# Extend mechanistic model
future_t = np.arange(
    days,
    days + forecast_days
)

full_t = np.arange(
    0,
    days + forecast_days
)

future_solution = solve_ivp(
    malaria_model,
    [0, days + forecast_days - 1],
    initial_conditions,
    t_eval=full_t
)

if not future_solution.success:
    st.error("Future mechanistic forecast failed.")
    st.stop()

future_Sh = future_solution.y[0, days:]
future_Iv = future_solution.y[4, days:]

future_mechanistic = (
    beta_h * future_Sh * future_Iv / Nh
)

# Assume future environmental conditions remain
# at their most recent 7-day average.
weather = df.tail(7)

future_data = pd.DataFrame({
    "day": future_t,
    "rainfall": weather["rainfall"].mean(),
    "temperature": weather["temperature"].mean(),
    "humidity": weather["humidity"].mean(),
    "mechanistic_cases": future_mechanistic
})

future_correction = model.predict(
    future_data[features]
)

future_hybrid = np.maximum(
    0,
    future_mechanistic + future_correction
)

forecast = pd.DataFrame({
    "day": future_t,
    "mechanistic_forecast": future_mechanistic,
    "hybrid_forecast": future_hybrid
})

fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(
    df["day"], df["cases"],
    label="Historical observed cases"
)

ax.plot(
    future_t, future_mechanistic,
    linestyle="--",
    label="Mechanistic forecast"
)

ax.plot(
    future_t, future_hybrid,
    linestyle="--",
    label="Hybrid AI forecast"
)

ax.set_xlabel("Day")
ax.set_ylabel("Malaria cases")
ax.set_title("Future Malaria Risk Prediction")
ax.legend()
ax.grid(alpha=0.3)

st.pyplot(fig)
plt.close(fig)

st.dataframe(forecast, use_container_width=True)

st.download_button(
    "Download forecast CSV",
    forecast.to_csv(index=False),
    file_name="hybrid_malaria_forecast.csv",
    mime="text/csv"
)

st.success("Hybrid malaria prediction completed.")
