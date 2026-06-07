"""
Визуализация датасета California Sold Properties 2026
Генерирует все 9 рисунков для курсовой работы

Зависимости:
    py -m pip install pandas numpy matplotlib seaborn plotly
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import plotly.express as px

# ─────────────────────────────────────────────────────────────
# ЗАГРУЗКА И ПРЕДОБРАБОТКА
# ─────────────────────────────────────────────────────────────

df_raw = pd.read_csv("california_sold_properties_2026_final.csv")
df = df_raw.copy()

# Удаление дубликатов
df = df.drop_duplicates()

# Удаление строк без целевой переменной
df = df.dropna(subset=["listPrice"])

# Исправление ошибки: garage = 999
df.loc[df["garage"] > 20, "garage"] = np.nan

# Стандартизация категории type
df["type"] = df["type"].replace({
    "condo": "condos",
    "condo_townhome_rowhome_coop": "townhomes"
})

# Заполнение пропусков медианой по типу объекта
for col in ["sqft", "beds", "baths", "baths_full", "year_built", "stories", "garage"]:
    df[col] = df.groupby("type")[col].transform(
        lambda x: x.fillna(x.median())
    )

print(f"Датасет загружен: {len(df)} записей, {df.shape[1]} признаков")


def print_analysis_report():
    """Печатает в консоль основные результаты первичного анализа датасета."""
    print()
    print("=" * 70)
    print("ПЕРВИЧНЫЙ АНАЛИЗ: California Sold Properties 2026")
    print("=" * 70)

    print("\nШАГ 0: Структура датасета")
    print(f"Исходный размер                 : {df_raw.shape[0]} строк, {df_raw.shape[1]} признаков")
    print(f"После удаления дублей и NaN target: {df.shape[0]} строк, {df.shape[1]} признаков")
    print(f"Дубликатов в исходных данных    : {df_raw.duplicated().sum()}")
    print(f"Строк без целевого listPrice    : {df_raw['listPrice'].isna().sum()}")
    print("Типы признаков:")
    print(df_raw.dtypes.astype(str).to_string())

    print("\nШАГ 1: Целевой признак listPrice")
    target = df_raw["listPrice"].dropna()
    print(f"Количество значений             : {len(target)}")
    print(f"Минимум                         : ${target.min():,.0f}")
    print(f"Медиана                         : ${target.median():,.0f}")
    print(f"Среднее                         : ${target.mean():,.0f}")
    print(f"Максимум                        : ${target.max():,.0f}")
    print("Интерпретация: распределение цены правосторонне асимметрично, дорогие объекты повышают среднее.")

    print("\nШАГ 2: Пропущенные значения")
    missing = (
        pd.DataFrame({
            "Кол-во пропусков": df_raw.isna().sum(),
            "Доля, %": df_raw.isna().mean() * 100
        })
        .query("`Кол-во пропусков` > 0")
        .sort_values("Доля, %", ascending=False)
    )
    if missing.empty:
        print("Пропуски не обнаружены.")
    else:
        print(missing.to_string(formatters={"Доля, %": lambda x: f"{x:.1f}"}))
        print("Вывод: основной проблемный признак - sub_type; для числовых признаков возможна медианная импутация.")

    print("\nШАГ 3: Выбросы по методу IQR")
    outlier_cols = ["listPrice", "sqft", "beds", "baths", "year_built", "garage", "stories"]
    rows = []
    for col in outlier_cols:
        series = df_raw[col].dropna()
        if series.empty:
            continue
        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        count = ((series < lower) | (series > upper)).sum()
        rows.append({
            "Признак": col,
            "Нижняя граница": lower,
            "Верхняя граница": upper,
            "Выбросов": int(count),
            "Доля, %": count / len(series) * 100
        })
    outliers = pd.DataFrame(rows)
    print(outliers.to_string(
        index=False,
        formatters={
            "Нижняя граница": lambda x: f"{x:,.2f}",
            "Верхняя граница": lambda x: f"{x:,.2f}",
            "Доля, %": lambda x: f"{x:.1f}"
        }
    ))
    bad_garage = (df_raw["garage"] > 20).sum()
    print(f"Предметная проверка: garage > 20 найдено {bad_garage} строк; значение 999 трактуется как ошибка.")

    print("\nШАГ 4: Корреляционный анализ")
    corr_cols = ["listPrice", "sqft", "beds", "baths", "year_built", "garage", "stories"]
    corr = df[corr_cols].corr()["listPrice"].drop("listPrice").sort_values(key=lambda s: s.abs(), ascending=False)
    print("Корреляция признаков с listPrice:")
    print(corr.to_string(float_format=lambda x: f"{x:.2f}"))
    print("Вывод: наиболее заметная линейная связь с ценой наблюдается у площади объекта sqft.")

    print("\nШАГ 5: Категориальные признаки")
    print("Распределение type:")
    type_table = pd.DataFrame({
        "Кол-во": df["type"].value_counts(),
        "Доля, %": df["type"].value_counts(normalize=True) * 100
    })
    print(type_table.to_string(formatters={"Доля, %": lambda x: f"{x:.1f}"}))
    print("\nРаспределение sub_type в исходных данных:")
    sub_table = pd.DataFrame({
        "Кол-во": df_raw["sub_type"].fillna("(нет данных)").value_counts(),
        "Доля, %": df_raw["sub_type"].fillna("(нет данных)").value_counts(normalize=True) * 100
    })
    print(sub_table.to_string(formatters={"Доля, %": lambda x: f"{x:.1f}"}))

    print("\nШАГ 6: Рекомендации по предобработке")
    print("- удалить дубликаты и строки без listPrice;")
    print("- заменить ошибочные значения garage, например 999, на NaN;")
    print("- заполнить умеренные пропуски числовых признаков медианой по type;")
    print("- для sub_type рассмотреть исключение признака или укрупнение категории из-за 82,8% пропусков;")
    print("- логарифмировать listPrice для ослабления правосторонней асимметрии;")
    print("- применить One-Hot Encoding к type и масштабирование числовых признаков.")
    print("=" * 70)


print_analysis_report()

# ─────────────────────────────────────────────────────────────
# НАСТРОЙКИ СТИЛЯ (Matplotlib / Seaborn)
# ─────────────────────────────────────────────────────────────

matplotlib.rcParams.update({
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    12,
    "axes.labelsize":    11,
    "figure.dpi":        150,
    "savefig.dpi":       200,
    "savefig.bbox":      "tight",
    "savefig.facecolor": "white",
})

sns.set_theme(style="whitegrid", palette="Blues_d")

# ─────────────────────────────────────────────────────────────
# РИС. 1 — Гистограммы числовых признаков (Matplotlib)
# ─────────────────────────────────────────────────────────────

fig1_cols   = ["listPrice", "sqft", "beds", "baths", "year_built"]
fig1_labels = ["Цена, $", "Площадь, кв.фут", "Спальни", "Ванные", "Год постройки"]

fig, axes = plt.subplots(1, 5, figsize=(22, 4))

for ax, col, label in zip(axes, fig1_cols, fig1_labels):
    data = df[col].dropna()
    ax.hist(data, bins=40, color="#2171B5", edgecolor="white", linewidth=0.4)
    ax.set_title(label)
    ax.set_ylabel("Частота")
    ax.yaxis.set_major_formatter(
        mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
    )
    if col == "listPrice":
        ax.xaxis.set_major_formatter(
            mticker.FuncFormatter(lambda x, _: f"{int(x/1e3)}К")
        )
    ax.tick_params(axis="x", labelsize=9)

plt.suptitle(
    "Рисунок 1. Гистограммы распределения числовых признаков",
    y=1.02, fontsize=12
)
plt.tight_layout()
plt.savefig("рис1_гистограммы.png")
plt.show()
print("✓ Рис. 1 сохранён → рис1_гистограммы.png")

# ─────────────────────────────────────────────────────────────
# РИС. 2 — Scatter: площадь vs цена (Seaborn)
# ─────────────────────────────────────────────────────────────

sample2 = (
    df[df["listPrice"] < 5_000_000]
    .dropna(subset=["sqft", "listPrice"])
    .sample(n=min(3000, len(df)), random_state=42)
)

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(
    data=sample2, x="sqft", y="listPrice",
    alpha=0.35, s=18, color="#2171B5", ax=ax
)
sns.regplot(
    data=sample2, x="sqft", y="listPrice",
    scatter=False, color="#C0392B",
    line_kws={"linewidth": 2.2}, ax=ax
)
ax.set_xlabel("Площадь объекта, кв. фут")
ax.set_ylabel("Цена, $")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${int(x/1e3)}К")
)
r = sample2["sqft"].corr(sample2["listPrice"])
ax.set_title(f"r = {r:.2f}  |  n = {len(sample2):,}", fontsize=10, color="grey")

plt.suptitle(
    "Рисунок 2. Зависимость цены от площади объекта (seaborn)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("рис2_scatter_sqft_price.png")
plt.show()
print("✓ Рис. 2 сохранён → рис2_scatter_sqft_price.png")

# ─────────────────────────────────────────────────────────────
# РИС. 3 — Boxplot: цена по кол-ву спален (Seaborn)
# ─────────────────────────────────────────────────────────────

box3 = df[
    df["beds"].between(1, 7) &
    (df["listPrice"] < 5_000_000)
].copy()
box3["beds"] = box3["beds"].astype(int).astype(str) + " сп."

fig, ax = plt.subplots(figsize=(11, 6))
sns.boxplot(
    data=box3, x="beds", y="listPrice",
    order=[f"{i} сп." for i in range(1, 8)],
    palette="Blues",
    flierprops={"markersize": 2, "alpha": 0.3},
    ax=ax
)
ax.set_xlabel("Количество спален")
ax.set_ylabel("Цена, $")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${int(x/1e3)}К")
)

for i in range(1, 8):
    med = df[df["beds"] == i]["listPrice"].median()
    if not np.isnan(med):
        ax.text(i - 1, med + 25_000, f"${med/1e3:.0f}К",
                ha="center", va="bottom", fontsize=9, color="#C0392B")

plt.suptitle(
    "Рисунок 3. Распределение цены по количеству спален (seaborn)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("рис3_boxplot_beds_price.png")
plt.show()
print("✓ Рис. 3 сохранён → рис3_boxplot_beds_price.png")

# ─────────────────────────────────────────────────────────────
# РИС. 4 — Scatter: год постройки vs цена (Seaborn)
# ─────────────────────────────────────────────────────────────

sample4 = df[
    (df["listPrice"] < 5_000_000) &
    (df["year_built"] >= 1900) &
    (df["year_built"] <= 2026)
].dropna(subset=["year_built"]).sample(n=min(3000, len(df)), random_state=7)

fig, ax = plt.subplots(figsize=(9, 6))
sns.scatterplot(
    data=sample4, x="year_built", y="listPrice",
    alpha=0.3, s=16, color="#2171B5", ax=ax
)

dec_med = (
    sample4
    .assign(decade=((sample4["year_built"] // 10) * 10))
    .groupby("decade")["listPrice"].median()
    .reset_index()
)
ax.plot(dec_med["decade"], dec_med["listPrice"],
        color="#C0392B", linewidth=2.5, label="Медиана по десятилетиям")
ax.legend(fontsize=10)
ax.set_xlabel("Год постройки")
ax.set_ylabel("Цена, $")
ax.yaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"${int(x/1e3)}К")
)
r4 = sample4["year_built"].corr(sample4["listPrice"])
ax.set_title(f"r = {r4:.2f}  |  n = {len(sample4):,}", fontsize=10, color="grey")

plt.suptitle(
    "Рисунок 4. Зависимость цены от года постройки (seaborn)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("рис4_scatter_year_price.png")
plt.show()
print("✓ Рис. 4 сохранён → рис4_scatter_year_price.png")

# ─────────────────────────────────────────────────────────────
# РИС. 5 — Интерактивная гистограмма цен (Plotly → HTML)
# ─────────────────────────────────────────────────────────────

fig5 = px.histogram(
    df[df["listPrice"] < 5_000_000],
    x="listPrice",
    nbins=80,
    color_discrete_sequence=["#2171B5"],
    labels={"listPrice": "Цена, $", "count": "Количество объектов"},
    title="Рисунок 5. Интерактивная гистограмма распределения цен (Plotly)",
)
fig5.update_layout(
    bargap=0.05,
    xaxis_tickprefix="$",
    xaxis_tickformat=",.0f",
    yaxis_title="Количество объектов",
    font=dict(size=13),
    width=1000, height=520,
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig5.add_vline(
    x=df["listPrice"].median(),
    line_dash="dash", line_color="#C0392B",
    annotation_text=f"Медиана: ${df['listPrice'].median()/1e3:.0f}К",
    annotation_position="top right",
    annotation_font_size=12,
)
fig5.write_html("рис5_histogram_price.html")
fig5.show()
print("✓ Рис. 5 сохранён → рис5_histogram_price.html")
print("  Откройте HTML в браузере и нажмите 📷 в правом верхнем углу графика")

# ─────────────────────────────────────────────────────────────
# РИС. 6 — Boxplot по типу недвижимости (Plotly → HTML)
# ─────────────────────────────────────────────────────────────

type_order = (
    df[df["listPrice"] < 5_000_000]
    .groupby("type")["listPrice"]
    .median()
    .sort_values(ascending=False)
    .index.tolist()
)

fig6 = px.box(
    df[(df["listPrice"] < 5_000_000) & df["type"].isin(type_order)],
    x="type", y="listPrice",
    category_orders={"type": type_order},
    color="type",
    color_discrete_sequence=[
        "#08306B", "#08519C", "#2171B5",
        "#4292C6", "#6BAED6", "#9ECAE1", "#C6DBEF"
    ],
    labels={"type": "Тип недвижимости", "listPrice": "Цена, $"},
    title="Рисунок 6. Интерактивный boxplot цен по типу недвижимости (Plotly)",
)
fig6.update_layout(
    yaxis_tickprefix="$",
    yaxis_tickformat=",.0f",
    showlegend=False,
    font=dict(size=13),
    width=1000, height=540,
    plot_bgcolor="white",
    paper_bgcolor="white",
)
fig6.write_html("рис6_boxplot_type_price.html")
fig6.show()
print("✓ Рис. 6 сохранён → рис6_boxplot_type_price.html")
print("  Откройте HTML в браузере и нажмите 📷 в правом верхнем углу графика")

# ─────────────────────────────────────────────────────────────
# РИС. 7 — Доля пропущенных значений по признакам (Matplotlib)
# ─────────────────────────────────────────────────────────────

missing_stats = (
    pd.DataFrame({
        "missing_count": df_raw.isnull().sum(),
        "missing_pct": df_raw.isnull().mean() * 100
    })
    .query("missing_count > 0")
    .sort_values("missing_pct", ascending=True)
)

fig, ax = plt.subplots(figsize=(11, 6))
colors = [
    "#DE2D26" if pct >= 60 else "#FD8D3C" if pct >= 20 else "#6BAED6"
    for pct in missing_stats["missing_pct"]
]
bars = ax.barh(
    missing_stats.index,
    missing_stats["missing_pct"],
    color=colors,
    edgecolor="#333333",
    linewidth=0.4
)

ax.set_xlabel("Доля пропусков, %")
ax.set_ylabel("")
ax.set_xlim(0, 100)
ax.grid(axis="x", color="#E0E0E0", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right"]].set_visible(False)

for bar, (_, row) in zip(bars, missing_stats.iterrows()):
    label = f"{int(row['missing_count'])} строк ({row['missing_pct']:.1f}%)"
    ax.text(
        row["missing_pct"] + 1.2,
        bar.get_y() + bar.get_height() / 2,
        label,
        va="center",
        fontsize=9,
        color="#222222"
    )

plt.suptitle(
    "Рисунок 7. Доля пропущенных значений по признакам датасета",
    fontsize=12
)
plt.tight_layout()
plt.savefig("рис7_missing_values_bar.png")
plt.show()
print("✓ Рис. 7 сохранён → рис7_missing_values_bar.png")

# ─────────────────────────────────────────────────────────────
# РИС. 8 — Тепловая карта корреляций (Seaborn)
# ─────────────────────────────────────────────────────────────

corr_cols  = ["listPrice", "sqft", "beds", "baths", "year_built", "garage", "stories"]
corr_names = ["Цена", "Площадь", "Спальни", "Ванные", "Год постр.", "Гараж", "Этажи"]

corr_matrix = df[corr_cols].corr()
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(9, 7))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True, fmt=".2f", annot_kws={"size": 11},
    cmap="Blues",
    vmin=-1, vmax=1,
    linewidths=0.5, linecolor="white",
    square=True,
    xticklabels=corr_names,
    yticklabels=corr_names,
    ax=ax,
)
plt.suptitle(
    "Рисунок 8. Тепловая карта корреляционной матрицы числовых признаков",
    fontsize=12
)
plt.tight_layout()
plt.savefig("рис8_heatmap_corr.png")
plt.show()
print("✓ Рис. 8 сохранён → рис8_heatmap_corr.png")

# ─────────────────────────────────────────────────────────────
# РИС. 9 — Категориальные признаки (Matplotlib)
# ─────────────────────────────────────────────────────────────

type_counts = df["type"].value_counts()

sub_raw = df_raw["sub_type"].value_counts(dropna=False)
sub_raw.index = sub_raw.index.fillna("(нет данных)")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Горизонтальный bar chart — type
n = len(type_counts)
colors1 = plt.cm.Blues(np.linspace(0.35, 0.85, n))
bars = ax1.barh(type_counts.index[::-1], type_counts.values[::-1], color=colors1)
ax1.set_xlabel("Количество объектов")
ax1.set_title("Признак  type")
ax1.xaxis.set_major_formatter(
    mticker.FuncFormatter(lambda x, _: f"{int(x):,}".replace(",", " "))
)
for bar, val in zip(bars, type_counts.values[::-1]):
    ax1.text(bar.get_width() + 20, bar.get_y() + bar.get_height() / 2,
             f"{val:,}".replace(",", " "), va="center", fontsize=9)
ax1.set_xlim(0, type_counts.max() * 1.18)

# Круговая диаграмма — sub_type
colors2 = ["#2171B5", "#6BAED6", "#DEEBF7"][:len(sub_raw)]
wedges, texts, autotexts = ax2.pie(
    sub_raw.values,
    labels=sub_raw.index,
    autopct="%1.1f%%",
    colors=colors2,
    startangle=90,
    textprops={"fontsize": 11},
)
for at in autotexts:
    at.set_fontsize(10)
ax2.set_title("Признак  sub_type")

plt.suptitle(
    "Рисунок 9. Распределение категориальных признаков (type и sub_type)",
    fontsize=12
)
plt.tight_layout()
plt.savefig("рис9_categorical.png")
plt.show()
print("✓ Рис. 9 сохранён → рис9_categorical.png")

# ─────────────────────────────────────────────────────────────
# ИТОГОВЫЙ СПИСОК ФАЙЛОВ
# ─────────────────────────────────────────────────────────────

print()
print("=" * 62)
print("Готово! Файлы сохранены в папке со скриптом:")
print()
files = [
    ("PNG",  "рис1_гистограммы.png"),
    ("PNG",  "рис2_scatter_sqft_price.png"),
    ("PNG",  "рис3_boxplot_beds_price.png"),
    ("PNG",  "рис4_scatter_year_price.png"),
    ("HTML", "рис5_histogram_price.html   ← открыть в браузере, нажать 📷"),
    ("HTML", "рис6_boxplot_type_price.html  ← открыть в браузере, нажать 📷"),
    ("PNG",  "рис7_missing_values_bar.png"),
    ("PNG",  "рис8_heatmap_corr.png"),
    ("PNG",  "рис9_categorical.png"),
]
for fmt, name in files:
    print(f"  [{fmt}]  {name}")
print()
print("PNG → вставляйте в Word напрямую.")
print("HTML → откройте в Chrome/Edge, нажмите 📷 → скачается PNG.")
print("=" * 62)