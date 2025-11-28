# === IMPORTS ===
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import matplotlib

matplotlib.rcParams['font.family'] = 'Times New Roman'
plt.rcParams.update({'font.size': 12})

from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# === LOAD & CLEAN DATA ===
df = pd.read_csv(r"Book1000.csv")
df['Total_pore_volume'] = pd.to_numeric(df['Total_pore_volume'], errors='coerce')
df = df[df['Permeability'].notna()]

# === STATISTICS & DISTRIBUTION ===
print("📊 Dataset Info:")
print(df.info())
print(df.describe(include='all'))

numeric_cols = df.select_dtypes(include=[np.number]).columns
df[numeric_cols].hist(bins=30, figsize=(16, 12), edgecolor='black')
plt.suptitle("Feature Distributions", fontsize=12, fontweight='bold')
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

for col in numeric_cols:
    plt.figure(figsize=(6, 4))
    sns.kdeplot(df[col].dropna(), shade=True)
    plt.title(f"Distribution of {col}", fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()

# === CORRELATION HEATMAP BEFORE DROPPING ===
imputer = KNNImputer(n_neighbors=5)
X_corr_full = df.drop(columns=['Permeability', 'Molar_mass _sweep_gas', 'Sweep_gas_flow', 'Feed_flow_rate'])
X_corr_full = pd.DataFrame(imputer.fit_transform(X_corr_full), columns=X_corr_full.columns)
X_corr_full = pd.DataFrame(StandardScaler().fit_transform(X_corr_full), columns=X_corr_full.columns)
Xy_corr = pd.concat([X_corr_full, df['Permeability'].reset_index(drop=True).rename("Permeability")], axis=1)

plt.figure(figsize=(12, 10))
sns.heatmap(Xy_corr.corr(), annot=True, fmt=".2f", cmap="coolwarm", square=True)
plt.title("Pearson Correlation Heatmap", fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# === PREPROCESSING ===
drop_features = ['Total_pore_volume', 'Molar_mass _sweep_gas', 'Sweep_gas_flow', 'Feed_flow_rate','Kinetic_diameter ']
X = df.drop(columns=['Permeability'] + drop_features)
y = df['Permeability']

X_imputed = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
X_scaled = pd.DataFrame(StandardScaler().fit_transform(X_imputed), columns=X.columns)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# === DEFINE MODELS ===
models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Lasso Regression": Lasso(alpha=0.1),
    "ElasticNet": ElasticNet(alpha=0.1, l1_ratio=0.5),
    "Random Forest": RandomForestRegressor(random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(random_state=42),
    "AdaBoost": AdaBoostRegressor(random_state=42),
    "Extra Trees": ExtraTreesRegressor(random_state=42),
    "XGBoost": XGBRegressor(random_state=42, verbosity=0),
    "SVR": SVR()
}

results = []
predictions = {}
fitted_models = {}
X_all = X_scaled
y_all = y.reset_index(drop=True)

# === MODEL FITTING ===
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_pred_all = model.predict(X_all)

    predictions[name] = {"test": y_pred_test, "train": y_pred_train, "all": y_pred_all}
    fitted_models[name] = model

    results.append({
        "Model": name,
        "R2_Train": r2_score(y_train, y_pred_train),
        "R2_Test": r2_score(y_test, y_pred_test),
        "R2_All": r2_score(y_all, y_pred_all),
        "MAE": mean_absolute_error(y_test, y_pred_test),
        "MSE": mean_squared_error(y_test, y_pred_test),
        "RMSE": np.sqrt(mean_squared_error(y_test, y_pred_test))
    })

results_df = pd.DataFrame(results).sort_values(by="R2_Test", ascending=False)
print("\n📋 Model Evaluation:")
print(results_df)

# === PREDICTED VS ACTUAL (TRAIN/TEST PLOT) ===
top4_models = results_df.head(4)['Model'].tolist()
all_predicted_dfs = []

for name in top4_models:
    model = fitted_models[name]
    y_test_pred = predictions[name]["test"]
    y_train_pred = predictions[name]["train"]

    plt.figure(figsize=(6, 6))
    plt.scatter(y_train, y_train_pred, alpha=0.3, label="Train", color='blue')
    plt.scatter(y_test, y_test_pred, alpha=0.3, label="Test", color='red')
    plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
    plt.xlabel("Actual Permeability (Barrer)")
    plt.ylabel("Predicted Permeability (Barrer)")
    plt.title(f"Predicted vs Actual - {name}", fontsize=12, fontweight='bold')
    plt.text(0.05, 0.95, f"$R^2$ = {r2_score(y_test, y_test_pred):.3f}", transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    plt.legend()
    plt.tight_layout()
    plt.show()

    # Save predicted data
    df_train = pd.DataFrame({"Actual": y_train.values, "Predicted": y_train_pred, "Dataset": "Train"})
    df_test = pd.DataFrame({"Actual": y_test.values, "Predicted": y_test_pred, "Dataset": "Test"})
    df_combined = pd.concat([df_train, df_test], axis=0).reset_index(drop=True)
    df_combined["Model"] = name
    all_predicted_dfs.append(df_combined)

# === SAVE TO EXCEL ===
with pd.ExcelWriter(r"C:\Users\ZENTOU\Downloads\Predicted_vs_Actual_Top4.xlsx") as writer:
    for df, name in zip(all_predicted_dfs, top4_models):
        df.to_excel(writer, sheet_name=name.replace(" ", "_"), index=False)

# === FEATURE IMPORTANCES (TREE MODELS) ===
tree_models = ["Random Forest", "Gradient Boosting", "AdaBoost", "Extra Trees", "XGBoost"]
feature_names = np.array(X.columns)

for name in tree_models:
    model = fitted_models[name]
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    plt.figure(figsize=(10, 5))
    plt.title(f"Feature Importances - {name}", fontsize=12, fontweight='bold')
    bars = plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), feature_names[indices], rotation=45, ha='right')
    for bar, imp in zip(bars, importances[indices]):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{imp:.2f}",
                 ha='center', va='bottom', fontsize=9)
    plt.tight_layout()
    plt.show()

# === SHAP ANALYSIS ===
for name in tree_models:
    print(f"\n🧠 SHAP Summary Plot - {name}")
    model = fitted_models[name]
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=X.columns)
    except Exception as e:
        print(f"SHAP failed for {name}: {e}")

# === COMBINED METRICS BAR PLOT FOR TOP 4 MODELS ===
top4 = results_df.head(4).copy()
top4.set_index("Model", inplace=True)

metrics = ['R2_Test', 'MAE', 'RMSE']
normalized = top4[metrics].copy()
normalized['MAE'] = normalized['MAE'] / normalized['MAE'].max()
normalized['RMSE'] = normalized['RMSE'] / normalized['RMSE'].max()

x = np.arange(len(top4.index))
width = 0.25

plt.figure(figsize=(10, 6))
bars1 = plt.bar(x - width, normalized['R2_Test'], width, label='R² Test')
bars2 = plt.bar(x, normalized['MAE'], width, label='MAE (normalized)')
bars3 = plt.bar(x + width, normalized['RMSE'], width, label='RMSE (normalized)')

for bars, metric in zip([bars1, bars2, bars3], ['R2_Test', 'MAE', 'RMSE']):
    for i, bar in enumerate(bars):
        val = top4.iloc[i][metric]
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                 f"{val:.3f}", ha='center', va='bottom', fontsize=9)

plt.xticks(x, top4.index, rotation=45, ha='right')
plt.title("Top 4 Models - R², MAE, RMSE Comparison", fontsize=12, fontweight='bold')
plt.ylabel("Score (Normalized for MAE & RMSE)", fontsize=12)
plt.legend(loc="upper left")
plt.ylim(0, max(max(normalized['R2_Test']), max(normalized['MAE']), max(normalized['RMSE'])) + 0.3)
plt.tight_layout()
plt.show()

