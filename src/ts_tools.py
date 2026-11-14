"""Функции для анализа и предобработки временного ряда продаж.

Все, что используется в ноутбуке, лежит здесь, а в самом ноутбуке остаются
только вызовы и интерпретация результатов.
"""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import linregress
from statsmodels.tsa.stattools import adfuller, kpss, acf
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


# ----------------------------------------------------------------- данные
def load_retail(path):
    """Читает Online Retail из xlsx и приводит типы."""
    df = pd.read_excel(path)
    df["InvoiceDate"] = pd.to_datetime(df["InvoiceDate"])
    df["InvoiceNo"] = df["InvoiceNo"].astype(str)
    df["StockCode"] = df["StockCode"].astype(str)
    return df


def select_product(df, stock_code):
    """Оставляет только реальные продажи товара: без возвратов (счета на C)
    и без строк с неположительным количеством."""
    mask = (
        (df["StockCode"] == stock_code)
        & ~df["InvoiceNo"].str.startswith("C")
        & (df["Quantity"] > 0)
    )
    return df.loc[mask].copy()


def aggregate(product_df, freq, value="Quantity"):
    """Агрегирует транзакции в ряд с нужной частотой.

    Ряд возвращается на полной календарной сетке. Периоды без единой
    транзакции остаются NaN, а не нулями: решать, что это (нулевой спрос
    или дыра в данных), нужно отдельно, см. weekday_gaps / missing_report.
    """
    s = product_df.set_index("InvoiceDate")[value].resample(freq).sum(min_count=1)
    return s.rename(value)


def drop_partial_edges(s, data_start, data_end, period_freq):
    """Выкидывает периоды на краях, которые данные покрывают не целиком
    (например, первая/последняя неделя или последний месяц).
    period_freq: 'D', 'W-SUN' или 'M'."""
    per = s.index.to_period(period_freq)
    keep = (per.start_time >= pd.Timestamp(data_start).normalize()) & (
        per.end_time < pd.Timestamp(data_end).normalize()
    )
    return s[keep]


def fill_zero_if_open(s, open_days, period_freq):
    """Для недельного/месячного ряда: пустой период, в котором магазин работал,
    это настоящий ноль. Если же в периоде не было ни одного рабочего дня,
    значение остаётся NaN."""
    out = s.copy()
    per_open = pd.Series(1, index=open_days).groupby(
        open_days.to_period(period_freq)
    ).sum()
    for ts in out.index[out.isna()]:
        if per_open.get(ts.to_period(period_freq), 0) > 0:
            out.loc[ts] = 0
    return out


def trim_to_sales_span(s):
    """Обрезает ряд по первой и последней реальной продаже товара."""
    return s.loc[s.first_valid_index():s.last_valid_index()]


def build_series(df, stock_code):
    """Весь путь от транзакций до очищенных рядов трёх частот.

    Возвращает словарь {'D': ..., 'W': ..., 'MS': ...}. Дневной ряд - на
    торговом календаре (нули в рабочие дни, дни без торговли выкинуты),
    недельный и месячный - нули только там, где магазин работал.
    """
    start, end = df["InvoiceDate"].min(), df["InvoiceDate"].max()
    open_days = shop_open_days(df)
    prod = select_product(df, stock_code)
    out = {}
    for freq, pfreq in [("D", "D"), ("W", "W-SUN"), ("MS", "M")]:
        s = trim_to_sales_span(aggregate(prod, freq))
        s = drop_partial_edges(s, start, end, pfreq)
        if freq == "D":
            out[freq] = to_trading_calendar(s, open_days)
        else:
            s = fill_zero_if_open(s, open_days, pfreq)
            out[freq] = fill_missing(s, "interpolate") if s.isna().any() else s
    return out


# ----------------------------------------------------- качество данных
def missing_report(s):
    """Сводка по пропускам: сколько, какая доля и где они лежат
    (список непрерывных отрезков с датами начала и конца)."""
    is_na = s.isna().to_numpy()
    runs, start = [], None
    for i, flag in enumerate(is_na):
        if flag and start is None:
            start = i
        if not flag and start is not None:
            runs.append((s.index[start], s.index[i - 1], i - start))
            start = None
    if start is not None:
        runs.append((s.index[start], s.index[-1], len(s) - start))
    return {
        "n_total": len(s),
        "n_missing": int(is_na.sum()),
        "share_missing": float(is_na.mean()),
        "runs": pd.DataFrame(runs, columns=["start", "end", "length"]),
    }


def shop_open_days(df):
    """Дни, в которые в магазине была хотя бы одна транзакция по любому товару.
    Остальные календарные дни считаем днями, когда торговли не было."""
    return pd.DatetimeIndex(df["InvoiceDate"].dt.normalize().unique()).sort_values()


def split_empty_days(s, open_days):
    """Делит пустые дни товара на две группы:
    - 'zero': магазин работал, а товар не купили, это настоящий нулевой спрос;
    - 'closed': в магазине не было торговли вообще, это не спрос, а отсутствие
      наблюдения.
    Возвращает (zero_days, closed_days)."""
    empty = s.index[s.isna()]
    is_open = empty.isin(open_days)
    return empty[is_open], empty[~is_open]


def to_trading_calendar(s, open_days):
    """Ряд на торговом календаре: нулевой спрос в рабочие дни проставляем нулями,
    дни без торговли выкидываем из ряда."""
    s = s.reindex(s.index.intersection(open_days.union(s.dropna().index)))
    return s.fillna(0)


def weekday_gaps(s):
    """Для дневного ряда: на какие дни недели приходятся пустые дни."""
    na = s[s.isna()]
    counts = na.index.dayofweek.value_counts().reindex(range(7), fill_value=0)
    counts.index = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
    return counts


def iqr_outliers(s, k=1.5):
    """Выбросы по правилу Тьюки. Возвращает булеву маску и границы."""
    x = s.dropna()
    q1, q3 = x.quantile([0.25, 0.75])
    iqr = q3 - q1
    lo, hi = q1 - k * iqr, q3 + k * iqr
    return (s < lo) | (s > hi), (lo, hi)


def mad_outliers(s, thr=3.5):
    """Модифицированный z-score на медиане и MAD. Для скошенных рядов
    продаж устойчивее, чем среднее и std."""
    x = s.dropna()
    med = x.median()
    mad = np.median(np.abs(x - med))
    z = 0.6745 * (s - med) / mad
    return z.abs() > thr, z


def stl_residual_outliers(s, period, thr=3.0):
    """Выбросы по остатку STL: точки, у которых остаток больше thr робастных
    сигм. В отличие от IQR учитывает тренд и сезонность."""
    r = decompose(s, period, method="stl", robust=True).resid
    sigma = 1.4826 * np.median(np.abs(r - np.median(r)))
    return r.abs() > thr * sigma, r


def inspect_day(product_df, date, top=5):
    """Крупнейшие транзакции товара в конкретный день: помогает понять,
    выброс - это один оптовый заказ или реальный всплеск спроса."""
    d = product_df[product_df["InvoiceDate"].dt.normalize() == pd.Timestamp(date)]
    cols = [c for c in ["InvoiceNo", "Quantity", "UnitPrice", "CustomerID", "Country"]
            if c in d.columns]
    return d.sort_values("Quantity", ascending=False)[cols].head(top)


def fill_missing(s, how="zero"):
    """Заполнение пропусков: 'zero' | 'interpolate' | 'ffill'."""
    if how == "zero":
        return s.fillna(0)
    if how == "interpolate":
        return s.interpolate("time")
    if how == "ffill":
        return s.ffill()
    raise ValueError(f"неизвестный способ: {how}")


def winsorize(s, lo=None, hi=None):
    """Обрезка значений по границам (точка сохраняется, но смягчается)."""
    return s.clip(lower=lo, upper=hi)


# -------------------------------------------------------- описательное
def describe_series(s):
    """Описательные статистики плюс характеристики, полезные для продаж."""
    x = s.dropna()
    d = x.describe().to_dict()
    d["median"] = x.median()
    d["skew"] = x.skew()
    d["kurtosis"] = x.kurt()
    d["cv"] = x.std() / x.mean() if x.mean() else np.nan
    d["share_zero"] = float((x == 0).mean())
    d["total"] = x.sum()
    # ADI и CV^2 по ненулевым значениям - классификация прерывистого спроса
    nz = x[x > 0]
    d["adi"] = len(x) / len(nz) if len(nz) else np.nan
    d["cv2_nonzero"] = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
    return pd.Series(d)


def rolling_stats(s, window):
    """Скользящие среднее, std и коэффициент вариации."""
    r = s.rolling(window, min_periods=max(2, window // 2))
    out = pd.DataFrame({"mean": r.mean(), "std": r.std()})
    out["cv"] = out["std"] / out["mean"]
    return out


def trend_slope(s):
    """Наклон линейного тренда (по номеру периода), p-value и R^2."""
    x = s.dropna()
    res = linregress(np.arange(len(x)), x.to_numpy(dtype=float))
    return {"slope": res.slope, "pvalue": res.pvalue, "r2": res.rvalue ** 2}


# ------------------------------------------------------------ декомпозиция
def decompose(s, period, method="stl", model="additive", robust=True):
    """STL или классическая декомпозиция (seasonal_decompose).

    Считаем по значениям, без привязки к индексу: у ряда торговых дней индекс
    нерегулярный (нет суббот), и statsmodels не должен пытаться угадывать
    частоту. Даты возвращаем обратно в результат.
    """
    y = s.to_numpy(dtype=float)
    if method == "stl":
        r = STL(y, period=period, robust=robust).fit()
    else:
        r = seasonal_decompose(y, model=model, period=period)
    mk = lambda a: pd.Series(np.asarray(a, dtype=float), index=s.index)
    return SimpleNamespace(
        observed=mk(r.observed), trend=mk(r.trend),
        seasonal=mk(r.seasonal), resid=mk(r.resid),
    )


def weekday_kruskal(s):
    """Критерий Краскела-Уоллиса: отличаются ли продажи по дням недели.
    H0 - распределения во все дни одинаковы."""
    from scipy.stats import kruskal
    groups = [g.to_numpy() for _, g in s.groupby(s.index.dayofweek) if len(g) > 1]
    stat, p = kruskal(*groups)
    return stat, p


def seasonal_strength(res):
    """Сила сезонности и тренда по Хайндману:
    F = max(0, 1 - Var(остаток) / Var(остаток + компонента)).
    Близко к 0 - компоненты нет, близко к 1 - она доминирует."""
    resid = res.resid.dropna()
    seas = res.seasonal.loc[resid.index]
    trend = res.trend.loc[resid.index]
    fs = max(0.0, 1 - np.var(resid) / np.var(resid + seas))
    ft = max(0.0, 1 - np.var(resid) / np.var(resid + trend))
    return {"seasonal_strength": fs, "trend_strength": ft}


def acf_at_lags(s, lags):
    """ACF в выбранных лагах и доверительная граница 1.96/sqrt(N)."""
    a = acf(s.dropna(), nlags=max(lags), fft=True)
    bound = 1.96 / np.sqrt(s.notna().sum())
    return pd.Series({l: a[l] for l in lags}, name="acf"), bound


def spectrum_peaks(s, top=5):
    """Топ пиков амплитудного спектра; период в отсчётах ряда."""
    y = s.dropna().to_numpy(dtype=float)
    y = y - y.mean()
    n = len(y)
    amp = np.abs(np.fft.rfft(y)) / n * 2
    f = np.fft.rfftfreq(n, d=1)
    amp, f = amp[1:], f[1:]
    idx = np.argsort(amp)[::-1][:top]
    return pd.DataFrame({"period": 1 / f[idx], "amplitude": amp[idx]})


# ----------------------------------------------------------- стационарность
def stationarity_tests(s, regression="c", name=""):
    """ADF (H0: есть единичный корень, ряд нестационарен) и
    KPSS (H0: ряд стационарен). Возвращает таблицу и вердикт словами."""
    x = s.dropna()
    adf_stat, adf_p, adf_lag, *_ = adfuller(x, regression=regression, autolag="AIC")
    kpss_stat, kpss_p, kpss_lag, _ = kpss(x, regression=regression, nlags="auto")
    adf_ok = adf_p < 0.05
    kpss_ok = kpss_p >= 0.05
    if adf_ok and kpss_ok:
        verdict = "стационарен (оба теста согласны)"
    elif not adf_ok and not kpss_ok:
        verdict = "нестационарен (оба теста согласны)"
    elif adf_ok and not kpss_ok:
        verdict = "тесты расходятся: ADF за стационарность, KPSS против"
    else:
        verdict = "тесты расходятся: ADF против стационарности, KPSS за"
    table = pd.DataFrame(
        {
            "stat": [adf_stat, kpss_stat],
            "p-value": [adf_p, kpss_p],
            "lags": [adf_lag, kpss_lag],
            "H0": ["единичный корень", "стационарность"],
        },
        index=[f"ADF {name}".strip(), f"KPSS {name}".strip()],
    )
    return table, verdict


def make_stationary(s, log=False, d=0, seasonal_lag=None):
    """Преобразование ряда: log1p (опционально), обычные разности порядка d,
    затем сезонная разность с лагом seasonal_lag."""
    out = s.copy()
    if log:
        out = np.log1p(out)
    for _ in range(d):
        out = out.diff()
    if seasonal_lag:
        out = out.diff(seasonal_lag)
    return out.dropna()


def choose_transformation(s, candidates=None, alpha=0.05):
    """Перебирает преобразования от простых к сложным и возвращает первое,
    после которого ADF и KPSS согласно говорят 'стационарен'.
    Если такого нет - возвращает None. Заодно отдаёт таблицу всех попыток."""
    if candidates is None:
        candidates = [
            dict(log=False, d=0),
            dict(log=True, d=0),
            dict(log=False, d=1),
            dict(log=True, d=1),
        ]
    rows, chosen = [], None
    for kw in candidates:
        z = make_stationary(s, **kw)
        tab, verdict = stationarity_tests(z)
        rows.append({
            **kw,
            "ADF p": tab["p-value"].iloc[0],
            "KPSS p": tab["p-value"].iloc[1],
            "вердикт": verdict,
        })
        ok = tab["p-value"].iloc[0] < alpha and tab["p-value"].iloc[1] >= alpha
        if ok and chosen is None:
            chosen = kw
    return chosen, pd.DataFrame(rows)


# ---------------------------------------------------------------- графики
def plot_series(s, title, ylabel, label="продажи", marks=None, ax=None,
                figsize=(13, 4)):
    """График ряда с названием, подписями осей и легендой.
    marks - список кортежей (серия, подпись, цвет) для подсветки точек."""
    if ax is None:
        _, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, lw=1, label=label)
    for pts, lab, color in marks or []:
        pts = pts.dropna()
        ax.scatter(pts.index, pts.values, color=color, zorder=3, s=25, label=lab)
    ax.set_title(title)
    ax.set_xlabel("Дата")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3)
    ax.legend()
    return ax


def plot_decomposition(res, title, unit):
    """Четыре панели: ряд, тренд, сезонность, остаток."""
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    parts = [
        ("Ряд", res.observed),
        ("Тренд", res.trend),
        ("Сезонность", res.seasonal),
        ("Остаток", res.resid),
    ]
    for ax, (name, data) in zip(axes, parts):
        ax.plot(data.index, data.values, lw=1)
        ax.set_ylabel(f"{name}, {unit}", fontsize=9)
        ax.grid(alpha=0.3)
    axes[0].set_title(title)
    axes[-1].set_xlabel("Дата")
    fig.tight_layout()
    return fig


def plot_acf_pacf(s, lags, suffix=""):
    """ACF и PACF рядом, с подписанными осями."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    plot_acf(s.dropna(), lags=lags, ax=axes[0])
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], method="ywm")
    axes[0].set_title(f"ACF {suffix}".strip())
    axes[1].set_title(f"PACF {suffix}".strip())
    for ax in axes:
        ax.set_xlabel("Лаг")
        ax.set_ylabel("Корреляция")
    fig.tight_layout()
    return fig
