import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy.stats import gaussian_kde
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

sns.set_style("whitegrid")
plt.rcParams.update({
    "figure.facecolor": "#F7F9FC",
    "axes.facecolor": "#FFFFFF",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "font.family": "DejaVu Sans",
})

ACCENT = "#2C7BB6"
HIGHLIGHT = "#D7191C"
BG = "#F7F9FC"


# loading and cleaning

df = pd.read_csv(r"C:\Users\arpit\Downloads\Electric_Vehicle_Population_Data.csv")

df.drop_duplicates(subset="DOL Vehicle ID", inplace=True)

df["Electric Range"] = pd.to_numeric(df["Electric Range"], errors="coerce").fillna(0)
df.dropna(subset=["County", "City"], inplace=True)
df["Model Year"] = pd.to_numeric(df["Model Year"], errors="coerce")
df["Electric Range"] = pd.to_numeric(df["Electric Range"], errors="coerce")
df.dropna(subset=["Model Year", "Electric Range"], inplace=True)

print("Shape after cleaning:", df.shape)
print("\nNull values remaining:")
print(df.isnull().sum()[df.isnull().sum() > 0])
print("\n", df[["Model Year", "Electric Range"]].describe().round(2))


# categorical analysis

top10_makes = df["Make"].value_counts().head(10)

bev_phev = df["Electric Vehicle Type"].value_counts()
bev_phev.index = bev_phev.index.str.replace("Battery Electric Vehicle (BEV)", "BEV", regex=False)
bev_phev.index = bev_phev.index.str.replace("Plug-in Hybrid Electric Vehicle (PHEV)", "PHEV", regex=False)

fig = plt.figure(figsize=(18, 7))
fig.patch.set_facecolor(BG)
fig.suptitle("🚗  Categorical Analysis Dashboard", fontsize=17, fontweight="bold", y=1.02)
gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.4)

ax1 = fig.add_subplot(gs[0, 0])
colours_bar = sns.color_palette("Blues_d", n_colors=10)[::-1]
bars = ax1.barh(top10_makes.index[::-1], top10_makes.values[::-1], color=colours_bar, edgecolor="white")
for bar, val in zip(bars, top10_makes.values[::-1]):
    ax1.text(bar.get_width() + 200, bar.get_y() + bar.get_height() / 2,
             f"{val:,}", va="center", fontsize=9, color="#333333")
ax1.set_title("Top 10 EV Manufacturers", fontweight="bold")
ax1.set_xlabel("Number of Vehicles")
ax1.set_xlim(0, top10_makes.max() * 1.18)

ax2 = fig.add_subplot(gs[0, 1])
wedges, texts, autotexts = ax2.pie(
    bev_phev.values, labels=bev_phev.index,
    autopct="%1.1f%%", startangle=140,
    colors=["#2C7BB6", "#FDAE61"],
    wedgeprops=dict(edgecolor="white", linewidth=2),
    textprops=dict(fontsize=11)
)
for at in autotexts:
    at.set_fontweight("bold")
ax2.set_title("BEV vs PHEV Split", fontweight="bold")

ax3 = fig.add_subplot(gs[0, 2])
ev_type_counts = df["Electric Vehicle Type"].value_counts()
sns.barplot(x=["BEV", "PHEV"], y=ev_type_counts.values,
            palette=["#2C7BB6", "#FDAE61"], ax=ax3, edgecolor="white")
for i, v in enumerate(ev_type_counts.values):
    ax3.text(i, v + 500, f"{v:,}", ha="center", fontsize=10, fontweight="bold")
ax3.set_title("EV Type Count", fontweight="bold")
ax3.set_xlabel("Electric Vehicle Type")
ax3.set_ylabel("Count")
ax3.set_ylim(0, ev_type_counts.max() * 1.12)

plt.tight_layout(pad=2)
plt.savefig("plot_categorical.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()


# EV growth trend

growth = df.groupby("Model Year").size().reset_index(name="Count")
growth = growth[growth["Model Year"] >= 2010]

fig, ax = plt.subplots(figsize=(13, 6))
fig.patch.set_facecolor(BG)

ax.fill_between(growth["Model Year"], growth["Count"], alpha=0.18, color=ACCENT)
ax.plot(growth["Model Year"], growth["Count"],
        color=ACCENT, linewidth=2.8, marker="o", markersize=7,
        markerfacecolor="white", markeredgecolor=ACCENT,
        markeredgewidth=2.2, label="Registered EVs")

peak = growth.loc[growth["Count"].idxmax()]
ax.annotate(
    f"  Peak: {int(peak['Model Year'])}\n  {int(peak['Count']):,} EVs",
    xy=(peak["Model Year"], peak["Count"]),
    xytext=(peak["Model Year"] - 3, peak["Count"] * 0.88),
    arrowprops=dict(arrowstyle="->", color=HIGHLIGHT, lw=1.8),
    color=HIGHLIGHT, fontsize=10, fontweight="bold"
)

ax.set_title("📅  EV Registration Growth Over the Years (2010 onwards)",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Model Year")
ax.set_ylabel("Number of Registered EVs")
ax.set_xticks(growth["Model Year"])
ax.tick_params(axis="x", rotation=45)
ax.legend(fontsize=10)
plt.tight_layout(pad=2)
plt.savefig("plot_trend.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()


# correlation heatmap

numeric_cols = ["Model Year", "Electric Range", "Postal Code",
                "Legislative District", "DOL Vehicle ID"]
corr_matrix = df[numeric_cols].corr()

fig, ax = plt.subplots(figsize=(9, 6))
fig.patch.set_facecolor(BG)

sns.heatmap(
    corr_matrix, annot=True, fmt=".2f",
    cmap="coolwarm", center=0,
    linewidths=0.6, linecolor="#E0E0E0",
    annot_kws={"size": 11, "weight": "bold"},
    square=True, ax=ax,
    cbar_kws={"shrink": 0.8, "label": "Correlation"}
)
ax.set_title("📊  Correlation Heatmap of Numeric Features",
             fontsize=15, fontweight="bold", pad=15)
ax.tick_params(axis="x", rotation=30)
ax.tick_params(axis="y", rotation=0)
plt.tight_layout(pad=2)
plt.savefig("plot_correlation.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()


# distributions with KDE

dist_cols = [
    ("Electric Range",       "Electric Range (miles)", "#2C7BB6"),
    ("Model Year",           "Model Year",             "#1A9641"),
    ("Postal Code",          "Postal Code",            "#7B2D8B"),
    ("Legislative District", "Legislative District",   "#D7191C"),
    ("DOL Vehicle ID",       "DOL Vehicle ID",         "#FF7F00"),
    ("2020 Census Tract",    "2020 Census Tract",      "#4DAC26"),
]

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.patch.set_facecolor(BG)
fig.suptitle("📈  Feature Distributions with KDE", fontsize=17, fontweight="bold", y=1.01)

for ax, (col, label, colour) in zip(axes.flat, dist_cols):
    data = df[col].dropna()
    ax.set_facecolor("#FFFFFF")
    ax.hist(data, bins=40, color=colour, alpha=0.55,
            edgecolor="white", linewidth=0.4, density=True, label="Histogram")
    kde = gaussian_kde(data, bw_method="scott")
    xs = np.linspace(data.min(), data.max(), 300)
    ax.plot(xs, kde(xs), color=colour, linewidth=2.5, label="KDE")
    ax.set_title(label, fontweight="bold")
    ax.set_xlabel(label)
    ax.set_ylabel("Density")
    ax.legend(fontsize=8, framealpha=0.6)

plt.tight_layout(pad=2.5)
plt.savefig("plot_distributions.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()


# outlier detection

box_cols = ["Electric Range", "Model Year", "Legislative District"]
box_colours = ["#2C7BB6", "#1A9641", "#D7191C"]

fig, axes = plt.subplots(1, 3, figsize=(15, 6))
fig.patch.set_facecolor(BG)
fig.suptitle("📦  Outlier Detection via Boxplots", fontsize=16, fontweight="bold")

for ax, col, clr in zip(axes, box_cols, box_colours):
    sns.boxplot(y=df[col].dropna(), ax=ax, color=clr,
                flierprops=dict(marker="o", color=clr, markersize=3, alpha=0.4),
                width=0.45, linewidth=1.4)
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    n_out = ((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum()
    ax.set_title(f"{col}\nOutliers: {n_out:,}", fontweight="bold")
    ax.set_ylabel(col)

plt.tight_layout(pad=2.5)
plt.savefig("plot_outliers.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()


# linear regression

reg_df = df[(df["Electric Range"] > 0) &
            (df["Electric Vehicle Type"].str.contains("BEV"))].copy()

X = reg_df[["Model Year"]].values
y = reg_df["Electric Range"].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("R2 Score :", round(r2, 4))
print("MSE      :", round(mse, 4))
print("Slope    :", round(model.coef_[0], 4))
print("Intercept:", round(model.intercept_, 4))

sample = reg_df.sample(min(3000, len(reg_df)), random_state=1)

line_x = np.linspace(X.min(), X.max(), 200).reshape(-1, 1)
line_y = model.predict(line_x)

fig, ax = plt.subplots(figsize=(11, 6))
fig.patch.set_facecolor(BG)

ax.scatter(sample["Model Year"], sample["Electric Range"],
           alpha=0.35, s=18, color=ACCENT, label="Actual data (sample)")
ax.plot(line_x, line_y, color=HIGHLIGHT, linewidth=2.8, label="Regression line")

ax.set_title("🔵  Model Year vs Electric Range  –  Linear Regression",
             fontsize=15, fontweight="bold")
ax.set_xlabel("Model Year")
ax.set_ylabel("Electric Range (miles)")
ax.legend(fontsize=10)

plt.tight_layout(pad=2)
plt.savefig("plot_regression.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()


# master summary dashboard

fig = plt.figure(figsize=(22, 20))
fig.patch.set_facecolor(BG)
fig.suptitle("⚡  Electric Vehicle Population – EDA Summary Dashboard",
             fontsize=22, fontweight="bold", y=1.005)

gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.55, wspace=0.40)

ax_bar = fig.add_subplot(gs[0, :2])
colours_bar = sns.color_palette("Blues_d", n_colors=10)[::-1]
ax_bar.barh(top10_makes.index[::-1], top10_makes.values[::-1],
            color=colours_bar, edgecolor="white")
for i, (val, lbl) in enumerate(zip(top10_makes.values[::-1], top10_makes.index[::-1])):
    ax_bar.text(val + 300, i, f"{val:,}", va="center", fontsize=9)
ax_bar.set_title("Top 10 EV Manufacturers", fontweight="bold")
ax_bar.set_xlabel("Number of Vehicles")
ax_bar.set_xlim(0, top10_makes.max() * 1.18)

ax_pie = fig.add_subplot(gs[0, 2])
ax_pie.pie(bev_phev.values, labels=bev_phev.index, autopct="%1.1f%%",
           startangle=140, colors=["#2C7BB6", "#FDAE61"],
           wedgeprops=dict(edgecolor="white", linewidth=2),
           textprops=dict(fontsize=10))
ax_pie.set_title("BEV vs PHEV", fontweight="bold")

ax_trend = fig.add_subplot(gs[1, :])
ax_trend.fill_between(growth["Model Year"], growth["Count"], alpha=0.15, color=ACCENT)
ax_trend.plot(growth["Model Year"], growth["Count"],
              color=ACCENT, linewidth=2.5, marker="o", markersize=6,
              markerfacecolor="white", markeredgecolor=ACCENT, markeredgewidth=2)
ax_trend.set_title("EV Registration Growth (2010 →)", fontweight="bold")
ax_trend.set_xlabel("Model Year")
ax_trend.set_ylabel("Registered EVs")
ax_trend.set_xticks(growth["Model Year"])
ax_trend.tick_params(axis="x", rotation=45)

ax_scat = fig.add_subplot(gs[2, :2])
ax_scat.scatter(sample["Model Year"], sample["Electric Range"],
                alpha=0.3, s=15, color=ACCENT, label="Actual (sample)")
ax_scat.plot(line_x, line_y, color=HIGHLIGHT, linewidth=2.5, label="Regression line")
ax_scat.set_title("Model Year vs Electric Range", fontweight="bold")
ax_scat.set_xlabel("Model Year")
ax_scat.set_ylabel("Electric Range (miles)")
ax_scat.legend(fontsize=9)

ax_hist = fig.add_subplot(gs[2, 2])
rng_data = df["Electric Range"]
ax_hist.hist(rng_data, bins=40, color="#2C7BB6", alpha=0.6,
             edgecolor="white", density=True)
kde2 = gaussian_kde(rng_data, bw_method="scott")
xs2 = np.linspace(rng_data.min(), rng_data.max(), 300)
ax_hist.plot(xs2, kde2(xs2), color="#D7191C", linewidth=2.2)
ax_hist.set_title("Electric Range Distribution", fontweight="bold")
ax_hist.set_xlabel("Electric Range (miles)")
ax_hist.set_ylabel("Density")

plt.savefig("plot_master_dashboard.png", dpi=150, bbox_inches="tight", facecolor=BG)
plt.show()
