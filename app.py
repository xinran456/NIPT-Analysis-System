"""
NIPT时点选择与胎儿异常判定 —— 交互式可视化系统（Streamlit）
运行：streamlit run app/app.py

升级说明（相对静态图片版）：
1. 不再依赖 charts/ 目录下的静态 PNG，全部图表由 src/analysis.py 从 CSV 实时计算生成；
2. 图表全部为 Plotly 交互式：支持缩放、框选、悬浮读数、图例开关、导出 PNG；
3. 全局侧边栏筛选器（BMI组 / 孕周阶段 / IVF / 是否异常）+ 达标阈值滑块；
4. 关键分析（风险函数、判定阈值、误差幅度、模型对比指标）均可交互调节。
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import analysis  # noqa: E402

st.set_page_config(
    page_title="NIPT时点选择与胎儿异常判定",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ---------------- 数据与通用组件 ----------------
@st.cache_data(show_spinner=False)
def load_data() -> dict:
    return analysis.load_data()


def filtered(df: pd.DataFrame) -> pd.DataFrame:
    return analysis.filter_data(df, st.session_state["bmi"], st.session_state["stage"],
                                st.session_state["ivf"], st.session_state["abn"])


def render(fig, height=None):
    """渲染交互式 Plotly 图。"""
    if height:
        fig.update_layout(height=height)
    st.plotly_chart(fig, width="stretch", config=analysis.PLOTLY_CONFIG)


def fmt_pct(x):
    return "—" if pd.isna(x) else f"{x:.1%}"


def init_state():
    for key, val in {
        "bmi": list(analysis.BMI_ORDER),
        "stage": list(analysis.STAGE_ORDER),
        "ivf": [],
        "abn": [],
        "threshold": 4.0,
    }.items():
        st.session_state.setdefault(key, val)


# ---------------- 页面1：数据概览 ----------------
def page_overview(data):
    male, female = filtered(data["male"]), filtered(data["female"])
    st.title("🩺 NIPT时点选择与胎儿异常判定分析系统")
    st.caption("交互式可视化版：所有图表由数据实时计算，可缩放、悬浮读数和筛选")
    st.markdown("---")

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("男胎记录数", f"{len(male)}")
    c2.metric("女胎记录数", f"{len(female)}")
    c3.metric("男胎孕妇数", f"{male['孕妇代码'].nunique()}")
    c4.metric("女胎孕妇数", f"{female['孕妇代码'].nunique()}")
    c5.metric("男胎异常",f"{male['是否异常'].sum()}条" if len(male) else "-",delta=f"{male['是否异常'].mean():.1%}" if len(male) else None,)
    c6.metric("女胎异常",f"{female['是否异常'].sum()}条" if len(female) else "-",delta=f"{female['是否异常'].mean():.1%}" if len(female) else None,)

    st.markdown("### 数据基本信息（当前筛选范围）")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("男胎数据")
        if len(male):
            st.write(f"- 孕周范围：{male['孕周'].min():.1f} ~ {male['孕周'].max():.1f} 周")
            st.write(f"- BMI范围：{male['孕妇BMI'].min():.1f} ~ {male['孕妇BMI'].max():.1f}（均值 {male['孕妇BMI'].mean():.1f}）")
            st.write(f"- Y染色体浓度：{male['Y浓度_pct'].min():.2f}% ~ {male['Y浓度_pct'].max():.2f}%")
            st.write(f"- Y浓度达标（≥{st.session_state['threshold']:.0f}%）：{(male['Y浓度_pct'] >= st.session_state['threshold']).sum()} 条")
    with c2:
        st.subheader("女胎数据")
        if len(female):
            st.write(f"- 孕周范围：{female['孕周'].min():.1f} ~ {female['孕周'].max():.1f} 周")
            st.write(f"- BMI范围：{female['孕妇BMI'].min():.1f} ~ {female['孕妇BMI'].max():.1f}（均值 {female['孕妇BMI'].mean():.1f}）")
            st.write(f"- X染色体浓度：{female['X浓度_pct'].min():.2f}% ~ {female['X浓度_pct'].max():.2f}%")
            st.write(f"- 18/21号染色体Z值绝对值>3：{((female['18号染色体的Z值'].abs() > 3) | (female['21号染色体的Z值'].abs() > 3)).sum()} 条")

    st.markdown("### 交互式分布图")
    tab1, tab2, tab3, tab4 = st.tabs(["孕周分布", "BMI分布", "异常构成", "检测次数分布"])
    with tab1:
        render(analysis.fig_week_hist(male, female))
    with tab2:
        render(analysis.fig_bmi_hist(male, female))
    with tab3:
        render(analysis.fig_abnormal_pie(male, female))
    with tab4:
        render(analysis.fig_times_dist(male, female))

    st.markdown("### 数据预览")
    tabA, tabB, tabC, tabD = st.tabs(["男胎（清洗后）", "女胎（清洗后）", "原始Excel·男胎", "原始Excel·女胎"])
    with tabA:
        st.dataframe(male.head(1084), width="stretch", height=320)
    with tabB:
        st.dataframe(female.head(607), width="stretch", height=320)
    with tabC:
        st.dataframe(data["raw_male"].head(1084), width="stretch", height=320)
    with tabD:
        st.dataframe(data["raw_female"].head(607), width="stretch", height=320)


# ---------------- 页面2：数据清洗 ----------------
def page_cleaning(data):
    st.title("🧹 数据清洗与预处理")
    st.markdown("---")
    st.header("清洗流程")
    st.markdown("""
    1. **数据读取**：从Excel两个工作表分别读取男胎(1082条)和女胎(605条)数据
    2. **孕周解析**：将"11w+6"格式解析为小数周数 = 周 + 天/7
    3. **日期处理**：将YYYYMMDD整数解析为标准日期
    4. **缺失值处理**：
       - 末次月经缺失：尝试同孕妇填充
       - 染色体非整倍体NaN：语义为"无异常"
       - BMI缺失：用同孕妇中位数填充
    5. **异常值识别**：BMI 20.7~46.9、Y浓度 1%~23.4%、GC含量 38.6%~42.1% 均在合理范围
    6. **特征工程**：派生"孕周"（小数周）、"孕周阶段"、"BMI组"、"是否异常"二分类标签
    """)

    st.header("数据质量检查（由原始Excel实时计算）")
    c1, c2 = st.columns(2)
    with c1:
        st.subheader("男胎原始数据")
        miss = data["raw_male"].isna().sum()
        st.dataframe(pd.DataFrame({"列名": miss.index, "缺失数": miss.values})[miss.values > 0], width="stretch", height=280)
        st.info(f"原始记录 {len(data['raw_male'])} 条 → 清洗后 {len(data['male'])} 条，"
                f"缺失字段：末次月经 {int(data['raw_male']['末次月经'].isna().sum())} 条、"
                f"染色体非整倍体 {int(data['raw_male']['染色体的非整倍体'].isna().sum())} 条（语义=无异常）")
    with c2:
        st.subheader("女胎原始数据")
        miss = data["raw_female"].isna().sum()
        st.dataframe(pd.DataFrame({"列名": miss.index, "缺失数": miss.values})[miss.values > 0], width="stretch", height=280)
        st.info(f"原始记录 {len(data['raw_female'])} 条 → 清洗后 {len(data['female'])} 条，"
                f"缺失字段：孕妇BMI {int(data['raw_female']['孕妇BMI'].isna().sum())} 条、"
                f"染色体非整倍体 {int(data['raw_female']['染色体的非整倍体'].isna().sum())} 条")

    st.header("清洗前后字段对比")
    c1, c2 = st.columns(2)
    with c1:
        st.write("**原始字段（Excel工作表）**")
        st.write("、".join(list(data["raw_male"].columns)))
    with c2:
        st.write("**清洗后新增/派生字段**")
        derived = [c for c in data["male"].columns if c not in data["raw_male"].columns]
        st.write("、".join(derived))
    st.markdown("""
    > 数据质量：核心字段完整度高；样本量充足（1082+605条）；每人 1~8 次追踪检测；异常样本充足（男胎126条、女胎67条）。
    > 注意：数据以高BMI人群为主（均值≈32.3），低BMI组样本量较少；类别不平衡（异常占比约11%）。
    """)


# ---------------- 页面3：问题1 相关性分析 ----------------
def page_problem1(male):
    st.title("📈 问题1：Y染色体浓度与孕周、BMI的相关性分析")
    st.markdown("---")
    if len(male) < 10:
        st.warning(f"当前筛选下男胎样本仅 {len(male)} 条，不足以进行分析，请放宽侧边栏筛选条件。")
        return
    method = st.radio("相关系数方法", ["Pearson", "Spearman"], horizontal=True)
    render(analysis.fig_corr_heatmap(male, method))
    if method == "Pearson":
        corr = male[["Y浓度_pct", "孕周", "孕妇BMI"]].corr()
        st.markdown(
            f"**Pearson相关系数（n={len(male)}）：** Y浓度 vs 孕周 r = {corr.loc['Y浓度_pct','孕周']:+.3f}；"
            f"Y浓度 vs BMI r = {corr.loc['Y浓度_pct','孕妇BMI']:+.3f}；孕周 vs BMI r = {corr.loc['孕周','孕妇BMI']:+.3f}"
        )

    st.header("散点回归分析")
    render(analysis.fig_scatter_regression(male, st.session_state["threshold"]))

    st.header("线性回归模型（Y浓度 ~ 孕周 + BMI）")
    model = analysis.fit_ols(male)
    params = model.params
    st.markdown(
        f"**模型：** Y浓度 = {params['const']:.4f} {params['孕周']:+.4f}×孕周 {params['孕妇BMI']:+.4f}×BMI　"
        f"R² = {model.rsquared:.4f}，调整R² = {model.rsquared_adj:.4f}，"
        f"F = {model.fvalue:.2f}，p = {model.f_pvalue:.2e}"
    )
    ttab = pd.DataFrame({
        "变量": ["截距", "孕周", "BMI"],
        "系数": params.values,
        "t值": model.tvalues.values,
        "p值": model.pvalues.values,
    })
    ttab["p值"] = ttab["p值"].map(lambda v: f"{v:.2e}")
    st.dataframe(ttab, width="stretch", hide_index=True)
    render(analysis.fig_regression_diagnostics(model))

    st.header("非线性模型比较")
    fig4, m4 = analysis.fig_nonlinear_compare(male)
    render(fig4)
    m4["R²"] = m4["R²"].map(lambda v: f"{v:.4f}")
    m4["AIC"] = m4["AIC"].map(lambda v: f"{v:.2f}")
    m4["RSS"] = m4["RSS"].map(lambda v: f"{v:.2f}")
    st.dataframe(m4, width="stretch")

    st.markdown("""
    **结论：**
    1. Y浓度与孕周显著正相关：孕周每增1周，Y浓度平均增加约0.13%
    2. Y浓度与BMI显著负相关：BMI每增1，Y浓度平均减少约0.20%
    3. R²较低(约4.6%)反映个体间差异大，需通过分组分析细化
    """)


# ---------------- 页面4：问题2 BMI分组时点 ----------------
def page_problem2(male):
    st.title("📊 问题2：BMI分组与最佳NIPT时点")
    st.markdown("---")
    if len(male) < 10:
        st.warning(f"当前筛选下男胎样本仅 {len(male)} 条，不足以进行分析，请放宽侧边栏筛选条件。")
        return
    st.header("1. BMI分组与Y浓度趋势")
    render(analysis.fig_bmi_curves(male, st.session_state["threshold"]))
    st.info("图例可点击开关单个BMI组；鼠标悬浮可读数值；滚轮缩放。")

    st.header("2. 达标率分析")
    fig6, curves6 = analysis.fig_pass_rate(male, st.session_state["threshold"])
    render(fig6)

    st.header("3. 风险函数优化")
    st.markdown("""
    **风险函数**：风险(孕周) = 失败惩罚 × (1 − 达标率) + 时间惩罚 × 孕周风险系数 × max(0, 孕周−10)²
    - 孕周风险系数：早期(≤12周)=1，中期(13-27周)=5，晚期(≥28周)=20
    - 达标率曲线采用上方 0.5 周分箱平滑曲线
    """)
    c1, c2, c3 = st.columns(3)
    fail_pen = c1.slider("失败惩罚系数", 0.0, 2.0, 0.5, 0.05, help="未达标（检测失败）的成本权重")
    time_pen = c2.slider("时间惩罚系数", 0.0, 1.0, 0.1, 0.01, help="孕周延后的时间成本权重")
    show_ref = c3.toggle("叠加文件记录的最优解（星标）", value=True)
    fig7, best7 = analysis.fig_risk_tradeoff(male, st.session_state["threshold"], fail_pen, time_pen)
    if show_ref:
        opt2 = load_data()["opt2"]
        for g, row in opt2.iterrows():
            if g in analysis.GROUP_COLORS:
                fig7.add_vline(
                    x=row["best_week"],
                    line=dict(color=analysis.GROUP_COLORS[g], width=1.5, dash="dot"),
                    annotation_text=f"{g} 文件最优 {row['best_week']:.1f}周",
                    annotation_position="top right",
                    annotation_font_color=analysis.GROUP_COLORS[g],
                    annotation_font_size=10,
                )
    render(fig7)
    c1, c2 = st.columns(2)
    with c1:
        st.write("**当前参数重算的最优时点**")
        st.dataframe(best7.assign(达标率=best7["达标率"].map(fmt_pct)), width="stretch", hide_index=True)
    with c2:
        st.write("**文件记录的最优时点（原项目优化结果）**")
        st.dataframe(load_data()["opt2"].reset_index().rename(columns={"index": "BMI组"}),
                     width="stretch", hide_index=True)

    st.header("4. 检测误差影响")
    err = st.slider("测量误差幅度（±%，百分点）", 0.0, 2.0, 1.0, 0.1)
    render(analysis.fig_error_impact(male, st.session_state["threshold"], err))
    st.markdown("""
    **问题2结论：**
    - BMI越低，达标越早，可在较低孕周检测；高BMI组需要更晚检测
    - 检测误差对低BMI组影响更大（接近阈值的样本多）
    """)


# ---------------- 页面5：问题3 多因素优化 ----------------
def page_problem3(data):
    male, opt3 = filtered(data["male"]), data["opt3"]
    st.title("🎯 问题3：多因素综合最佳NIPT时点")
    st.markdown("---")
    if len(male) < 10:
        st.warning(f"当前筛选下男胎样本仅 {len(male)} 条，不足以进行分析，请放宽侧边栏筛选条件。")
        return
    st.header("1. 多因素预测模型（随机森林 → Y浓度）")
    fig9, imp9, cv9 = analysis.fig_rf_importance(male)
    render(fig9)
    st.dataframe(imp9.sort_values("重要性", ascending=False).reset_index(drop=True), width="stretch")
    st.markdown(f"> 特征含孕周、BMI、年龄、身高、体重、GC含量、读段数、比对比例、染色体Z值等；5折交叉验证 R²={cv9.mean():.3f}±{cv9.std():.3f}")

    st.header("2. 各BMI组预测达标率 vs 实测达标率")
    win = st.radio("实测口径", ["≥最优时点", "最优时点±0.5周"], horizontal=True)
    fig10, df10 = analysis.fig_multifactor_compare(male, opt3, st.session_state["threshold"], win)
    render(fig10)
    st.dataframe(df10.assign(实测达标率=df10["实测达标率"].map(fmt_pct),
                            模型预测达标率=df10["模型预测达标率"].map(fmt_pct)), width="stretch", hide_index=True)

    st.header("3. 风险最小化优化结果")
    render(analysis.fig_multifactor_optimal(opt3))
    c1, c2 = st.columns([1, 2])
    with c1:
        st.write("**最优时点明细**")
        st.dataframe(opt3.reset_index().rename(columns={"index": "BMI组"}), width="stretch", hide_index=True)
    with c2:
        st.markdown("""
        **问题3结论：**
        - 综合多因素后，高BMI组推荐检测时点延后更明显（40以上 → 15.5周）
        - 多因素模型比单纯BMI分组更精确
        - 40+ BMI组需延后至15周以上才能保证较高达标率
        """)
    st.info("数据来源：problem3_optimal_times.csv（原项目多因素优化输出）")


# ---------------- 页面6：问题4 女胎异常判定 ----------------
def page_problem4(data):
    female = filtered(data["female"])
    pred = data["pred"]
    st.title("🔬 问题4：女胎异常判定机器学习模型")
    st.markdown("---")
    if len(female) < 10:
        st.warning(f"当前筛选下女胎样本仅 {len(female)} 条，不足以进行分析，请放宽侧边栏筛选条件。")
        return
    st.markdown(f"**数据**：女胎 {len(female)} 条（异常 {female['是否异常'].sum()} 条）；"
                f"**测试集预测**：{len(pred)} 条（来自 model_predictions.csv）")
    st.markdown("**特征（10个）**：" + "、".join(analysis.F_FEATURES))

    st.header("1. 分类模型性能对比（5折交叉验证）")
    with st.spinner("训练 4 个模型并做 5 折交叉验证…"):
        metrics_df, X, y, rf = analysis.cv_classification_metrics(female)
    sel = st.multiselect("选择展示的指标", ["准确率", "精确率", "召回率", "F1", "AUC"],
                         default=["准确率", "召回率", "F1", "AUC"])
    render(analysis.fig_model_comparison(metrics_df, sel))
    st.dataframe(metrics_df.round(4), width="stretch")

    st.header("2. ROC 与 精确率-召回率曲线")
    render(analysis.fig_roc_pr(pred))
    st.markdown(f"> 测试集正例（异常）占比 {pred['真实标签'].mean():.1%}；传统 Z 值法(\\|Z\\|>3)召回率仅约 6%，漏检严重。")

    st.header("3. 混淆矩阵（阈值可调）")
    th = st.slider("判定阈值", 0.05, 0.95, 0.50, 0.05)
    fig14, m14 = analysis.fig_confusion(pred, th)
    render(fig14)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("准确率", f"{m14['准确率']:.3f}")
    c2.metric("精确率", f"{m14['精确率']:.3f}")
    c3.metric("召回率", f"{m14['召回率']:.3f}")
    c4.metric("F1", f"{m14['F1']:.3f}")

    st.header("4. 分类特征重要性")
    render(analysis.fig_clf_importance(rf))
    st.markdown("""
    **问题4结论：**
    1. 机器学习模型能综合多特征，显著提升异常检出能力（召回率）
    2. 18号/21号染色体Z值是最重要的判定特征
    3. 建议临床使用多特征综合判定，而非单一Z值阈值
    """)


# ---------------- 侧边栏与主入口 ----------------
def build_sidebar(data):
    with st.sidebar:
        st.title("🩺 NIPT分析系统")
        st.markdown("---")
        st.subheader("全局筛选")
        bmi = st.multiselect("BMI组", analysis.BMI_ORDER, default=st.session_state["bmi"])
        stage = st.multiselect("孕周阶段", analysis.STAGE_ORDER, default=st.session_state["stage"])
        ivf = st.multiselect("IVF妊娠", list(analysis.IVF_LABEL), default=st.session_state["ivf"],
                             format_func=lambda v: analysis.IVF_LABEL[v])
        abn = st.multiselect("是否异常", [0, 1], default=st.session_state["abn"],
                             format_func=lambda v: "异常" if v else "正常")
        threshold = st.slider("Y浓度达标阈值（%）", 2.0, 10.0, st.session_state["threshold"], 0.5)
        if st.button("重置筛选"):
            st.session_state["bmi"] = list(analysis.BMI_ORDER)
            st.session_state["stage"] = list(analysis.STAGE_ORDER)
            st.session_state["ivf"] = []
            st.session_state["abn"] = []
            st.session_state["threshold"] = 4.0
            st.rerun()
        for k, v in {"bmi": bmi, "stage": stage, "ivf": ivf, "abn": abn, "threshold": threshold}.items():
            st.session_state[k] = v
        m_f, f_f = filtered(data["male"]), filtered(data["female"])
        st.markdown("---")
        st.caption(f"当前筛选：男胎 {len(m_f)} 条 / 女胎 {len(f_f)} 条")

        st.markdown("---")
        page = st.radio("导航", [
            "📊 数据概览", "🧹 数据清洗", "📈 问题1：相关性分析",
            "📊 问题2：BMI分组时点", "🎯 问题3：多因素优化",
            "🔬 问题4：女胎异常判定",
        ])
        st.markdown("---")
        with st.expander("💡 使用说明"):
            st.markdown("""
- 所有图表为 **Plotly 交互式**：滚轮缩放、悬浮读数、图例点击开关、右上角可导出PNG
- 侧边栏筛选会同步作用于所有页面
- 风险函数、判定阈值、误差幅度等参数可在页面内调节并实时重算
- 数据来源：male_clean.csv / female_clean.csv / model_predictions.csv / optimal_ntip_times.csv / problem3_optimal_times.csv / 附件.xlsx
""")
        st.caption("NIPT时点选择与胎儿异常判定 · 2025年高教社杯C题实训项目（交互式升级版）")
        return page


def main():
    init_state()
    data = load_data()
    page = build_sidebar(data)
    if page == "📊 数据概览":
        page_overview(data)
    elif page == "🧹 数据清洗":
        page_cleaning(data)
    elif page == "📈 问题1：相关性分析":
        page_problem1(filtered(data["male"]))
    elif page == "📊 问题2：BMI分组时点":
        page_problem2(filtered(data["male"]))
    elif page == "🎯 问题3：多因素优化":
        page_problem3(data)
    elif page == "🔬 问题4：女胎异常判定":
        page_problem4(data)


if __name__ == "__main__":
    main()

