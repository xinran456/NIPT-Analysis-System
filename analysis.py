"""
NIPT时点选择与胎儿异常判定 —— 数据分析与交互式图表构建层
说明：本模块只做数据加载、统计计算和 Plotly 图构建，不依赖 Streamlit，
     便于单元测试与复用。界面层 app/app.py 负责交互控件与页面组织。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy import stats
from sklearn.ensemble import (
    GradientBoostingClassifier,
    RandomForestClassifier,
    RandomForestRegressor,
)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, cross_val_score
from sklearn.svm import SVC
import statsmodels.api as sm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"

# ---------- 常量与命名规范 ----------
# 数据文件中的 BMI 组写法 -> 规范化写法（与两个优化结果 CSV 的索引一致）
BMI_MAP = {"<28": "[20,28)", "28-32": "[28,32)", "32-36": "[32,36)", "36-40": "[36,40)", "40+": "40以上"}
BMI_ORDER = ["[20,28)", "[28,32)", "[32,36)", "[36,40)", "40以上"]
STAGE_ORDER = ["早期(≤12周)", "中期(13-27周)", "晚期(≥28周)"]
GROUP_COLORS = {
    "[20,28)": "#2E86AB",
    "[28,32)": "#F6AE2D",
    "[32,36)": "#F26419",
    "[36,40)": "#931F1D",
    "40以上": "#5B2A86",
}
IVF_LABEL = {0: "自然受孕", 1: "IVF", 2: "其他"}
DEFAULT_THRESHOLD = 4.0  # Y染色体浓度达标线（%）


def _bmi_group(v):
    """按分组定义由BMI值得到组名：<28, 28-32, 32-36, 36-40, 40+"""
    if pd.isna(v):
        return np.nan
    if v < 28:
        return "<28"
    if v < 32:
        return "28-32"
    if v < 36:
        return "32-36"
    if v < 40:
        return "36-40"
    return "40+"

# 问题3：男胎 Y 浓度预测特征
M_FEATURES = [
    "孕周", "孕妇BMI", "年龄", "身高", "体重", "GC含量", "原始读段数",
    "在参考基因组上比对的比例", "X浓度_pct", "13号染色体的Z值",
    "18号染色体的Z值", "21号染色体的Z值", "X染色体的Z值",
]
# 问题4：女胎异常判定特征
F_FEATURES = [
    "X染色体的Z值", "13号染色体的Z值", "18号染色体的Z值", "21号染色体的Z值",
    "X浓度_pct", "GC含量", "原始读段数", "在参考基因组上比对的比例",
    "孕妇BMI", "年龄",
]
# 问题1：相关性分析的变量
CORR_VARS = [
    "Y浓度_pct", "孕周", "孕妇BMI", "年龄", "身高", "体重", "GC含量",
    "原始读段数", "在参考基因组上比对的比例", "X浓度_pct",
]

PLOTLY_CONFIG = {
    "scrollZoom": True,
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
    "toImageButtonOptions": {"format": "png", "filename": "nipt_chart", "scale": 2},
}


# ================= 数据加载 =================
def load_data() -> dict:
    """加载全部数据文件，返回字典。"""
    male = pd.read_csv(DATA_DIR / "male_clean.csv", encoding="utf-8-sig")
    female = pd.read_csv(DATA_DIR / "female_clean.csv", encoding="utf-8-sig")
    pred = pd.read_csv(DATA_DIR / "model_predictions.csv", encoding="utf-8-sig")
    opt2 = pd.read_csv(DATA_DIR / "optimal_ntip_times.csv", index_col=0)
    opt3 = pd.read_csv(DATA_DIR / "problem3_optimal_times.csv", index_col=0)
    raw_male = pd.read_excel(DATA_DIR / "附件.xlsx", sheet_name="男胎检测数据")
    raw_female = pd.read_excel(DATA_DIR / "附件.xlsx", sheet_name="女胎检测数据")
    for df in (male, female):
        # 修复个别记录的BMI组缺失（按其BMI值按分组定义补填，不改变其他值）
        miss_bmi = df["BMI组"].isna()
        if miss_bmi.any():
            df.loc[miss_bmi, "BMI组"] = df.loc[miss_bmi, "孕妇BMI"].apply(_bmi_group)
        df["BMI组规范"] = df["BMI组"].map(BMI_MAP).fillna(df["BMI组"])
        # 怀孕次数为"1/2/≥3"语义列（≥3为文本），统一转字符串避免混型
        if "怀孕次数" in df.columns:
            df["怀孕次数"] = df["怀孕次数"].astype(str)
    for df in (raw_male, raw_female):
        if "怀孕次数" in df.columns and df["怀孕次数"].dtype == object:
            df["怀孕次数"] = df["怀孕次数"].astype(str)
    # 规范化两个优化结果表的索引顺序
    opt2.index = [str(i) for i in opt2.index]
    opt3.index = [str(i) for i in opt3.index]
    return {"male": male, "female": female, "pred": pred, "opt2": opt2, "opt3": opt3,
            "raw_male": raw_male, "raw_female": raw_female}


def filter_data(df: pd.DataFrame, bmi_groups: list, stages: list,
                ivf_values: list, abnormal_values: list) -> pd.DataFrame:
    """按侧边栏筛选条件过滤数据。空列表表示不过滤。"""
    out = df.copy()
    if bmi_groups:
        out = out[out["BMI组规范"].isin(bmi_groups)]
    if stages:
        out = out[out["孕周阶段"].isin(stages)]
    if ivf_values is not None and len(ivf_values) > 0:
        out = out[out["IVF编码"].isin(ivf_values)]
    if abnormal_values is not None and len(abnormal_values) > 0:
        out = out[out["是否异常"].isin(abnormal_values)]
    return out


def pass_rate_curves(male: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
                     error: float = 0.0) -> pd.DataFrame:
    """
    按 BMI 组计算 Y 浓度达标率（>= threshold）随孕周的变化曲线。
    - 孕周按 0.5 周分箱（箱中心 10.5 ~ 24.5），组内达标比例 + 3 箱滑动平滑
    - error>0 时模拟测量误差：在 Y 浓度上叠加 ±error（百分点）后重算达标率
    返回 DataFrame：行=孕周箱中心，列=BMI组
    """
    bins = np.arange(10.25, 24.75, 0.5)
    centers = bins[:-1] + 0.25
    curves = {}
    for g in BMI_ORDER:
        sub = male[male["BMI组规范"] == g]
        if len(sub) < 5:
            curves[g] = pd.Series(np.nan, index=centers)
            continue
        cut = pd.cut(sub["孕周"], bins=bins, labels=centers, include_lowest=True)
        if error == 0:
            y = sub["Y浓度_pct"]
        else:
            y = sub["Y浓度_pct"]
        rate = sub.assign(bin=cut, ok=(y + error >= threshold)).groupby("bin", observed=True)["ok"].mean()
        rate_lo = sub.assign(bin=cut, ok=(y - error >= threshold)).groupby("bin", observed=True)["ok"].mean()
        # 上偏曲线在加 error 一侧，下偏曲线在减 error 一侧
        if error > 0:
            up = sub.assign(bin=cut, ok=(y + error >= threshold)).groupby("bin", observed=True)["ok"].mean()
            lo = sub.assign(bin=cut, ok=(y - error >= threshold)).groupby("bin", observed=True)["ok"].mean()
            curves[g] = pd.DataFrame({"center": centers, "base": rate.reindex(centers),
                                      "up": up.reindex(centers), "lo": lo.reindex(centers)})
        else:
            curves[g] = pd.DataFrame({"center": centers, "base": rate.reindex(centers),
                                      "up": np.nan, "lo": np.nan})
    # 平滑（窗口3）
    if error > 0:
        out = {}
        for g, d in curves.items():
            d2 = d.copy()
            d2["base"] = d2["base"].rolling(3, center=True, min_periods=1).mean()
            d2["up"] = d2["up"].rolling(3, center=True, min_periods=1).mean()
            d2["lo"] = d2["lo"].rolling(3, center=True, min_periods=1).mean()
            out[g] = d2
        return out
    out = pd.DataFrame(index=centers, columns=BMI_ORDER)
    for g, d in curves.items():
        out[g] = d["base"].rolling(3, center=True, min_periods=1).mean()
    return out


# ================= 图01：相关系数热力图 =================
def fig_corr_heatmap(male: pd.DataFrame, method: str = "Pearson") -> go.Figure:
    df = male[CORR_VARS].dropna()
    corr = df.corr(method="pearson" if method == "Pearson" else "spearman")
    fig = go.Figure(go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        zmin=-1, zmax=1, colorscale="RdBu_r",
        text=np.round(corr.values, 2), texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"{method}相关系数矩阵（男胎，n={len(df)}）",
        height=620, margin=dict(l=20, r=20, t=60, b=20),
        xaxis_title="", yaxis_title="",
    )
    return fig


# ================= 图02：散点回归分析 =================
def _trend_xy(x: np.ndarray, y: np.ndarray, deg: int = 2, grid=None):
    grid = np.linspace(x.min(), x.max(), 120) if grid is None else grid
    coef = np.polyfit(x, y, deg)
    p = np.poly1d(coef)
    r2 = 1 - np.sum((y - p(x)) ** 2) / np.sum((y - y.mean()) ** 2)
    return grid, p(grid), r2


def fig_scatter_regression(male: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> go.Figure:
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=("Y染色体浓度 vs 孕周（按BMI组着色）",
                                        "Y染色体浓度 vs 孕妇BMI（按孕周阶段着色）"))
    for g in BMI_ORDER:
        sub = male[male["BMI组规范"] == g]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(x=sub["孕周"], y=sub["Y浓度_pct"], mode="markers",
                                 name=g, marker=dict(size=5, opacity=0.55, color=GROUP_COLORS[g]),
                                 hovertemplate="孕周 %{x:.1f}周<br>Y浓度 %{y:.2f}%<extra>" + g + "</extra>"),
                      row=1, col=1)
    # 总体二次趋势线
    gx, gy, r2 = _trend_xy(male["孕周"].values, male["Y浓度_pct"].values, 2)
    fig.add_trace(go.Scatter(x=gx, y=gy, mode="lines", name=f"总体趋势(R²={r2:.3f})",
                             line=dict(color="#111111", width=3, dash="dash")), row=1, col=1)
    fig.add_hline(y=threshold, line_dash="dot", line_color="#555", row=1, col=1,
                  annotation_text=f"达标线 {threshold}%", annotation_position="top left")

    for s in STAGE_ORDER:
        sub = male[male["孕周阶段"] == s]
        if len(sub) == 0:
            continue
        fig.add_trace(go.Scatter(x=sub["孕妇BMI"], y=sub["Y浓度_pct"], mode="markers",
                                 name=s, marker=dict(size=5, opacity=0.55),
                                 hovertemplate="BMI %{x:.1f}<br>Y浓度 %{y:.2f}%<extra>" + s + "</extra>"),
                      row=1, col=2)
    bx, by, br2 = _trend_xy(male["孕妇BMI"].values, male["Y浓度_pct"].values, 2)
    fig.add_trace(go.Scatter(x=bx, y=by, mode="lines", name=f"总体趋势(R²={br2:.3f})",
                             line=dict(color="#111111", width=3, dash="dash")), row=1, col=2)
    fig.update_layout(height=520, margin=dict(l=20, r=20, t=50, b=20), legend=dict(font=dict(size=11)))
    fig.update_xaxes(title_text="孕周（周）", row=1, col=1)
    fig.update_yaxes(title_text="Y染色体浓度（%）", row=1, col=1)
    fig.update_xaxes(title_text="孕妇BMI", row=1, col=2)
    fig.update_yaxes(title_text="Y染色体浓度（%）", row=1, col=2)
    return fig


# ================= 图03：回归诊断图 =================
def fit_ols(male: pd.DataFrame):
    X = sm.add_constant(male[["孕周", "孕妇BMI"]])
    model = sm.OLS(male["Y浓度_pct"], X).fit()
    return model


def fig_regression_diagnostics(model) -> go.Figure:
    fitted, resid = model.fittedvalues, model.resid
    n, k = model.nobs, model.df_model + 1
    std_resid = resid / resid.std(ddof=k) if resid.std(ddof=k) > 0 else resid
    fig = make_subplots(rows=2, cols=2, subplot_titles=("残差 vs 拟合值", "Q-Q 图",
                                                        "残差直方图", "尺度-位置图"))
    fig.add_trace(go.Scatter(x=fitted, y=resid, mode="markers", name="残差",
                             marker=dict(size=6, opacity=0.6),
                             hovertemplate="拟合值 %{x:.2f}<br>残差 %{y:.3f}<extra></extra>"), 1, 1)
    fig.add_hline(y=0, line_dash="dot", line_color="#999", row=1, col=1)
    (osm, osr), (slope, intercept, r) = stats.probplot(resid, dist="norm")
    fig.add_trace(go.Scatter(x=osm, y=osr, mode="markers", name="样本分位",
                             marker=dict(size=6, opacity=0.6)), 1, 2)
    fig.add_trace(go.Scatter(x=osm, y=intercept + slope * osm, mode="lines", name="理论线",
                             line=dict(color="crimson", width=2)), 1, 2)
    hist, edges = np.histogram(resid, bins=30, density=True)
    centers_ = (edges[:-1] + edges[1:]) / 2
    xs = np.linspace(resid.min(), resid.max(), 200)
    fig.add_trace(go.Bar(x=centers_, y=hist, name="残差密度", marker=dict(color="#9AC4F8")), 2, 1)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, resid.mean(), resid.std()),
                             name="正态密度", line=dict(color="crimson", width=2)), 2, 1)
    fig.add_trace(go.Scatter(x=fitted, y=np.sqrt(np.abs(std_resid)), mode="markers", name="√|标准化残差|",
                             marker=dict(size=6, opacity=0.6)), 2, 2)
    fig.update_layout(height=640, margin=dict(l=20, r=20, t=50, b=20),
                      legend=dict(font=dict(size=10)))
    fig.update_xaxes(title_text="拟合值", row=1, col=1)
    fig.update_yaxes(title_text="残差", row=1, col=1)
    fig.update_xaxes(title_text="理论分位数", row=1, col=2)
    fig.update_yaxes(title_text="样本分位数", row=1, col=2)
    fig.update_xaxes(title_text="残差", row=2, col=1)
    fig.update_yaxes(title_text="密度", row=2, col=1)
    fig.update_xaxes(title_text="拟合值", row=2, col=2)
    fig.update_yaxes(title_text="√|标准化残差|", row=2, col=2)
    return fig


# ================= 图04：非线性模型比较 =================
def nonlinear_compare(male: pd.DataFrame):
    w = male["孕周"].values
    y = male["Y浓度_pct"].values
    models, results = {}, {}
    lin = np.polyfit(w, y, 1); models["线性"] = lambda x: np.polyval(lin, x)
    quad = np.polyfit(w, y, 2); models["二次"] = lambda x: np.polyval(quad, x)
    logp = np.polyfit(np.log(w), y, 1); models["对数"] = lambda x: np.polyval(logp, np.log(x))
    for name, f in models.items():
        rss = np.sum((y - f(w)) ** 2)
        r2 = 1 - rss / np.sum((y - y.mean()) ** 2)
        aic = len(y) * np.log(rss / len(y)) + 2 * 2  # k=2个参数
        results[name] = {"R²": r2, "AIC": aic, "RSS": rss}
    return models, pd.DataFrame(results).T


def fig_nonlinear_compare(male: pd.DataFrame) -> tuple:
    models, metrics = nonlinear_compare(male)
    bins = np.arange(11, 25, 1)
    centers = bins[:-1] + 0.5
    cut = pd.cut(male["孕周"], bins=bins, labels=centers, include_lowest=True)
    means = male.assign(bin=cut).groupby("bin", observed=True)["Y浓度_pct"].mean()
    grid = np.linspace(11, 24, 150)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=means.index.astype(float), y=means.values, mode="markers+lines",
                             name="孕周分箱均值", marker=dict(size=8, color="#333"),
                             line=dict(width=1, color="#999"),
                             hovertemplate="孕周 %{x:.1f}周<br>Y浓度均值 %{y:.2f}%<extra></extra>"))
    colors = {"线性": "#2E86AB", "二次": "#F26419", "对数": "#5B2A86"}
    for name, f in models.items():
        fig.add_trace(go.Scatter(x=grid, y=f(grid), mode="lines", name=f"{name}拟合",
                                 line=dict(color=colors[name], width=3)))
    fig.update_layout(title=f"线性 / 二次 / 对数模型拟合对比（R²: 线性{metrics.loc['线性','R²']:.4f}，"
                            f"二次{metrics.loc['二次','R²']:.4f}，对数{metrics.loc['对数','R²']:.4f}）",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="孕周（周）", yaxis_title="Y染色体浓度（%）")
    return fig, metrics


# ================= 图05：BMI分组Y浓度曲线 =================
def fig_bmi_curves(male: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
                   show_scatter: bool = True) -> go.Figure:
    grid = np.linspace(11, 24, 120)
    fig = go.Figure()
    for g in BMI_ORDER:
        sub = male[male["BMI组规范"] == g]
        if len(sub) < 8:
            continue
        if show_scatter:
            fig.add_trace(go.Scatter(x=sub["孕周"], y=sub["Y浓度_pct"], mode="markers",
                                     name=f"{g} (n={len(sub)})",
                                     marker=dict(size=4, opacity=0.35, color=GROUP_COLORS[g]),
                                     hovertemplate="孕周 %{x:.1f}周<br>Y浓度 %{y:.2f}%<extra></extra>",
                                     showlegend=False))
        gy, ggrid = None, None
        try:
            coef = np.polyfit(sub["孕周"], sub["Y浓度_pct"], 2)
            gy = np.polyval(coef, grid)
        except np.linalg.LinAlgError:
            continue
        fig.add_trace(go.Scatter(x=grid, y=gy, mode="lines", name=f"{g}",
                                 line=dict(color=GROUP_COLORS[g], width=3.5),
                                 hovertemplate="孕周 %{x:.1f}周<br>拟合Y浓度 %{y:.2f}%<extra>" + g + "</extra>"))
    fig.add_hline(y=threshold, line_dash="dot", line_color="#555",
                  annotation_text=f"达标线 {threshold}%", annotation_position="top left")
    fig.update_layout(title="各BMI组Y染色体浓度随孕周变化的拟合曲线（二次多项式）",
                      height=500, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="孕周（周）", yaxis_title="Y染色体浓度（%）")
    return fig


# ================= 图06：达标率曲线 =================
def fig_pass_rate(male: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD) -> tuple:
    curves = pass_rate_curves(male, threshold)
    fig = go.Figure()
    for g in BMI_ORDER:
        s = curves[g].dropna()
        if len(s) == 0:
            continue
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode="lines+markers", name=g,
                                 line=dict(color=GROUP_COLORS[g], width=3),
                                 marker=dict(size=6),
                                 hovertemplate="孕周 %{x:.1f}周<br>达标率 %{y:.1%}<extra>" + g + "</extra>"))
    fig.add_hline(y=0.8, line_dash="dot", line_color="#999",
                  annotation_text="参考线 80%", annotation_position="top left")
    fig.update_layout(title=f"各BMI组Y浓度达标率（≥{threshold}%）随孕周变化（0.5周分箱+平滑）",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="孕周（周）", yaxis_title="达标率",
                      yaxis=dict(tickformat=".0%", range=[0, 1.05]))
    return fig, curves


# ================= 图07：风险权衡图 =================
def _time_coeff(week):
    return np.where(week < 13, 1.0, np.where(week < 28, 5.0, 20.0))


def risk_tradeoff(male: pd.DataFrame, threshold: float, fail_pen: float, time_pen: float):
    curves = pass_rate_curves(male, threshold)
    grid = np.array([float(i) for i in curves.index])
    rows = []
    for g in BMI_ORDER:
        p = curves[g].values
        if np.isnan(p).all():
            continue
        risk = fail_pen * (1 - p) + time_pen * _time_coeff(grid) * np.maximum(0, grid - 10) ** 2
        i = np.nanargmin(risk)
        rows.append({"BMI组": g, "最优孕周": round(float(grid[i]), 1),
                     "风险值": round(float(risk[i]), 4), "达标率": round(float(p[i]), 4)})
    return pd.DataFrame(rows)


def fig_risk_tradeoff(male: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
                      fail_pen: float = 0.5, time_pen: float = 0.1) -> tuple:
    curves = pass_rate_curves(male, threshold)
    grid = np.array([float(i) for i in curves.index])
    best = risk_tradeoff(male, threshold, fail_pen, time_pen)
    fig = go.Figure()
    for g in BMI_ORDER:
        p = curves[g].values
        if np.isnan(p).all():
            continue
        risk = fail_pen * (1 - p) + time_pen * _time_coeff(grid) * np.maximum(0, grid - 10) ** 2
        fig.add_trace(go.Scatter(x=grid, y=risk, mode="lines", name=g,
                                 line=dict(color=GROUP_COLORS[g], width=3),
                                 hovertemplate="孕周 %{x:.1f}周<br>风险 %{y:.3f}<extra>" + g + "</extra>"))
        b = best[best["BMI组"] == g]
        if len(b):
            fig.add_trace(go.Scatter(x=[b["最优孕周"].iloc[0]], y=[b["风险值"].iloc[0]],
                                     mode="markers", marker=dict(symbol="star", size=14, color=GROUP_COLORS[g],
                                                                 line=dict(color="#111", width=1)),
                                     name=f"{g}·最优", hovertemplate="最优孕周 %{x:.1f}周<br>风险 %{y:.3f}<extra></extra>",
                                     showlegend=False))
    fig.update_layout(title=f"风险-时点权衡（失败惩罚={fail_pen}，时间惩罚={time_pen}，达标阈值={threshold}%）",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="孕周（周）", yaxis_title="总风险", legend=dict(font=dict(size=11)))
    return fig, best


# ================= 图08：检测误差影响 =================
def fig_error_impact(male: pd.DataFrame, threshold: float = DEFAULT_THRESHOLD,
                     error: float = 1.0) -> go.Figure:
    curves = pass_rate_curves(male, threshold, error=error)
    fig = go.Figure()
    for g in BMI_ORDER:
        d = curves[g]
        if d is None or d["base"].isna().all():
            continue
        fig.add_trace(go.Scatter(x=d["center"], y=d["base"], mode="lines", name=f"{g} 基准",
                                 line=dict(color=GROUP_COLORS[g], width=3)))
        fig.add_trace(go.Scatter(x=d["center"], y=d["up"], mode="lines", name=f"{g} +{error}%",
                                 line=dict(color=GROUP_COLORS[g], width=2, dash="dash")))
        fig.add_trace(go.Scatter(x=d["center"], y=d["lo"], mode="lines", name=f"{g} -{error}%",
                                 line=dict(color=GROUP_COLORS[g], width=2, dash="dot")))
    fig.add_hline(y=0.8, line_dash="dot", line_color="#999")
    fig.update_layout(title=f"±{error}%测量误差对各BMI组达标率的影响（达标线 {threshold}%）",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="孕周（周）", yaxis_title="达标率",
                      yaxis=dict(tickformat=".0%", range=[0, 1.05]), legend=dict(font=dict(size=10)))
    return fig


# ================= 图09：随机森林特征重要性（问题3） =================
def rf_importance(male: pd.DataFrame):
    df = male[["Y浓度_pct"] + M_FEATURES].dropna()
    X, y = df[M_FEATURES], df["Y浓度_pct"]
    rf = RandomForestRegressor(n_estimators=300, random_state=42, n_jobs=-1)
    cv_r2 = cross_val_score(rf, X, y, cv=5, scoring="r2")
    rf.fit(X, y)
    imp = pd.DataFrame({"特征": M_FEATURES, "重要性": rf.feature_importances_}).sort_values("重要性")
    return imp, cv_r2


def fig_rf_importance(male: pd.DataFrame) -> tuple:
    imp, cv_r2 = rf_importance(male)
    fig = go.Figure(go.Bar(x=imp["重要性"], y=imp["特征"], orientation="h",
                           marker=dict(color=imp["重要性"], colorscale="Blues",
                                       line=dict(color="#333", width=0.5)),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title=f"随机森林预测Y浓度·特征重要性（5折CV R²={cv_r2.mean():.3f}±{cv_r2.std():.3f}）",
                      height=520, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="特征重要性", yaxis_title="")
    return fig, imp, cv_r2


# ================= 图10：多因素达标率对比 =================
def multifactor_compare(male: pd.DataFrame, opt3: pd.DataFrame, threshold: float,
                        window: str) -> pd.DataFrame:
    rows = []
    for g in BMI_ORDER:
        if g not in opt3.index:
            continue
        best = float(opt3.loc[g, "best_week"])
        sub = male[male["BMI组规范"] == g]
        if len(sub) == 0:
            rows.append({"BMI组": g, "实测达标率": np.nan, "模型预测达标率": float(opt3.loc[g, "pass_rate"]), "样本数": 0})
            continue
        if window == "≥最优时点":
            mask = sub["孕周"] >= best
        else:
            mask = (sub["孕周"] >= best - 0.5) & (sub["孕周"] < best + 0.5)
        sub2 = sub[mask]
        rows.append({"BMI组": g, "实测达标率": round(float((sub2["Y浓度_pct"] >= threshold).mean()), 4) if len(sub2) else np.nan,
                     "模型预测达标率": round(float(opt3.loc[g, "pass_rate"]), 4), "样本数": len(sub2)})
    return pd.DataFrame(rows)


def fig_multifactor_compare(male: pd.DataFrame, opt3: pd.DataFrame,
                            threshold: float = DEFAULT_THRESHOLD, window: str = "≥最优时点") -> tuple:
    df = multifactor_compare(male, opt3, threshold, window)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df["BMI组"], y=df["实测达标率"], name="实测达标率",
                         marker_color="#2E86AB",
                         text=[f"{v:.1%}" if not np.isnan(v) else "—" for v in df["实测达标率"]],
                         textposition="outside",
                         hovertemplate="%{x}: 实测 %{y:.1%}<extra></extra>"))
    fig.add_trace(go.Bar(x=df["BMI组"], y=df["模型预测达标率"], name="模型预测达标率(文件)",
                         marker_color="#F26419",
                         text=[f"{v:.1%}" for v in df["模型预测达标率"]], textposition="outside",
                         hovertemplate="%{x}: 预测 %{y:.1%}<extra></extra>"))
    fig.update_layout(title=f"各BMI组实测 vs 模型预测达标率（阈值{threshold}%，口径：{window}）",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="BMI组", yaxis_title="达标率", yaxis=dict(tickformat=".0%", range=[0, 1.1]),
                      barmode="group")
    return fig, df


# ================= 图11：多因素优化结果 =================
def fig_multifactor_optimal(opt3: pd.DataFrame) -> go.Figure:
    df = opt3.reindex([g for g in BMI_ORDER if g in opt3.index])
    fig = go.Figure(go.Bar(x=df.index, y=df["best_week"], name="最优检测时点",
                           marker=dict(color=df["pass_rate"], colorscale="YlOrRd", showscale=True,
                                       colorbar=dict(title="达标率")),
                           text=[f"{v:.1f}周" for v in df["best_week"]], textposition="outside",
                           hovertemplate="%{x}: 最优时点 %{y:.1f}周<br>达标率 %{customdata[0]:.1%}<br>成本 %{customdata[1]:.2f}<extra></extra>",
                           customdata=np.stack([df["pass_rate"], df["cost"]], axis=1)))
    fig.update_layout(title="多因素综合最优NIPT检测时点（各BMI组）",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="BMI组", yaxis_title="最优检测时点（孕周）")
    return fig


# ================= 图12：分类模型对比（问题4） =================
def build_models():
    return {
        "逻辑回归": LogisticRegression(max_iter=3000, class_weight="balanced"),
        "随机森林": RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1),
        "梯度提升": GradientBoostingClassifier(n_estimators=300, random_state=42),
        "SVM(RBF)": SVC(probability=True, class_weight="balanced", random_state=42),
    }


def cv_classification_metrics(female: pd.DataFrame, models=None) -> tuple:
    models = models or build_models()
    df = female[["是否异常"] + F_FEATURES].dropna()
    X, y = df[F_FEATURES], df["是否异常"]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    rows = []
    for name, model in models.items():
        proba = cross_val_predict(model, X, y, cv=skf, method="predict_proba")[:, 1]
        pred = (proba >= 0.5).astype(int)
        rows.append({
            "模型": name,
            "准确率": accuracy_score(y, pred),
            "精确率": precision_score(y, pred, zero_division=0),
            "召回率": recall_score(y, pred, zero_division=0),
            "F1": f1_score(y, pred, zero_division=0),
            "AUC": roc_auc_score(y, proba),
        })
    metrics_df = pd.DataFrame(rows).set_index("模型")
    # 供图15使用的最佳随机森林（全量训练）
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1).fit(X, y)
    return metrics_df, X, y, rf


def fig_model_comparison(metrics_df: pd.DataFrame, show_metrics: list) -> go.Figure:
    df = metrics_df.reset_index()
    fig = go.Figure()
    for m in show_metrics:
        fig.add_trace(go.Bar(x=df["模型"], y=df[m], name=m,
                             text=[f"{v:.3f}" for v in df[m]], textposition="outside",
                             hovertemplate="%{x}: %{y:.3f}<extra></extra>"))
    fig.update_layout(title="5折交叉验证·分类模型性能对比",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="", yaxis_title="得分", yaxis=dict(range=[0, 1.15]), barmode="group")
    return fig


# ================= 图13：ROC / PR 曲线 =================
def fig_roc_pr(pred: pd.DataFrame) -> go.Figure:
    y_true = pred["真实标签"]
    y_prob = pred["预测概率"]
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    pr_auc = auc(rec, prec)
    base = y_true.mean()
    fig = make_subplots(rows=1, cols=2, subplot_titles=(f"ROC曲线 (AUC={roc_auc:.3f})",
                                                        f"精确率-召回率曲线 (AUC={pr_auc:.3f})"))
    fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name="ROC",
                             line=dict(color="#2E86AB", width=3),
                             hovertemplate="FPR %{x:.2f}<br>TPR %{y:.2f}<extra></extra>"), 1, 1)
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="随机基线",
                             line=dict(color="#999", dash="dash")), 1, 1)
    fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name="PR",
                             line=dict(color="#F26419", width=3),
                             hovertemplate="召回率 %{x:.2f}<br>精确率 %{y:.2f}<extra></extra>"), 1, 2)
    fig.add_hline(y=base, line_dash="dash", line_color="#999", row=1, col=2,
                  annotation_text=f"基线 {base:.1%}", annotation_position="top left")
    fig.update_layout(height=460, margin=dict(l=20, r=20, t=50, b=20))
    fig.update_xaxes(title_text="假阳性率", row=1, col=1)
    fig.update_yaxes(title_text="真阳性率", row=1, col=1)
    fig.update_xaxes(title_text="召回率", row=1, col=2)
    fig.update_yaxes(title_text="精确率", row=1, col=2)
    return fig


# ================= 图14：混淆矩阵 =================
def confusion_metrics(pred: pd.DataFrame, threshold: float = 0.5) -> tuple:
    y_true = pred["真实标签"]
    y_pred = (pred["预测概率"] >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return cm, {
        "准确率": accuracy_score(y_true, y_pred),
        "精确率": precision_score(y_true, y_pred, zero_division=0),
        "召回率": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
    }


def fig_confusion(pred: pd.DataFrame, threshold: float = 0.5) -> tuple:
    cm, m = confusion_metrics(pred, threshold)
    labels = [["真阴性", "假阳性"], ["假阴性", "真阳性"]]
    fig = go.Figure(go.Heatmap(
        z=cm, x=["预测正常", "预测异常"], y=["实际正常", "实际异常"],
        colorscale="Blues", text=cm, texttemplate="%{text}",
        hovertemplate="%{y} × %{x}: %{z}<extra></extra>"))
    fig.update_layout(title=f"混淆矩阵（判定阈值 = {threshold:.2f}）",
                      height=440, margin=dict(l=20, r=20, t=60, b=20))
    return fig, m


# ================= 图15：分类特征重要性 =================
def fig_clf_importance(rf: RandomForestClassifier) -> go.Figure:
    imp = pd.DataFrame({"特征": F_FEATURES, "重要性": rf.feature_importances_}).sort_values("重要性")
    fig = go.Figure(go.Bar(x=imp["重要性"], y=imp["特征"], orientation="h",
                           marker=dict(color=imp["重要性"], colorscale="Reds",
                                       line=dict(color="#333", width=0.5)),
                           hovertemplate="%{y}: %{x:.3f}<extra></extra>"))
    fig.update_layout(title="女胎异常判定·随机森林特征重要性",
                      height=480, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="特征重要性", yaxis_title="")
    return fig


# ================= 概览页辅助图 =================
def fig_week_hist(male: pd.DataFrame, female: pd.DataFrame, bins: int = 30) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=male["孕周"], name="男胎", nbinsx=bins, opacity=0.7,
                               marker_color="#2E86AB", hovertemplate="孕周 %{x:.1f}<br>记录数 %{y}<extra></extra>"))
    fig.add_trace(go.Histogram(x=female["孕周"], name="女胎", nbinsx=bins, opacity=0.7,
                               marker_color="#F26419", hovertemplate="孕周 %{x:.1f}<br>记录数 %{y}<extra></extra>"))
    fig.update_layout(barmode="overlay", title="孕周分布（男胎 vs 女胎）",
                      height=380, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="孕周（周）", yaxis_title="记录数")
    return fig


def fig_bmi_hist(male: pd.DataFrame, female: pd.DataFrame, bins: int = 30) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Histogram(x=male["孕妇BMI"], name="男胎", nbinsx=bins, opacity=0.7,
                               marker_color="#2E86AB"))
    fig.add_trace(go.Histogram(x=female["孕妇BMI"], name="女胎", nbinsx=bins, opacity=0.7,
                               marker_color="#F26419"))
    fig.update_layout(barmode="overlay", title="孕妇BMI分布",
                      height=380, margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="孕妇BMI", yaxis_title="记录数")
    return fig


def fig_abnormal_pie(male: pd.DataFrame, female: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=1, cols=2, specs=[[{"type": "domain"}, {"type": "domain"}]],
                        subplot_titles=("男胎异常构成", "女胎异常构成"))
    for i, (df, name) in enumerate([(male, "男胎"), (female, "女胎")], start=1):
        vc = df["是否异常"].map({0: "正常", 1: "异常"}).value_counts()
        fig.add_trace(go.Pie(labels=vc.index, values=vc.values, hole=0.45,
                             marker=dict(colors=["#8FD694", "#E85D5D"]),
                             hovertemplate="%{label}: %{value}条 (%{percent})<extra></extra>"), 1, i)
    fig.update_layout(title="胎儿健康标签构成", height=380, margin=dict(l=20, r=20, t=60, b=20))
    return fig


def fig_times_dist(male: pd.DataFrame, female: pd.DataFrame) -> go.Figure:
    def dist(df):
        return df.groupby("孕妇代码").size().value_counts().sort_index()
    dm, df_ = dist(male), dist(female)
    fig = go.Figure()
    fig.add_trace(go.Bar(x=dm.index.astype(str), y=dm.values, name="男胎",
                         marker_color="#2E86AB"))
    fig.add_trace(go.Bar(x=df_.index.astype(str), y=df_.values, name="女胎",
                         marker_color="#F26419"))
    fig.update_layout(title="每位孕妇检测次数分布", height=380,
                      margin=dict(l=20, r=20, t=60, b=20),
                      xaxis_title="检测次数", yaxis_title="孕妇数", barmode="group")
    return fig

