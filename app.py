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
    """按侧边栏全局筛选条件过滤（空列表表示不过滤）。"""
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
    st.caption("数据读取与清洗 → 指标构建 → 统计可视化 → 最佳检测时点统计优化（Streamlit 原生组件版）")
    st.divider()

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("男胎记录数", f"{len(male)}")
    c2.metric("女胎记录数", f"{len(female)}")
    c3.metric("男胎孕妇数", f"{male['孕妇代码'].nunique()}")
    c4.metric("女胎孕妇数", f"{female['孕妇代码'].nunique()}")
    c5.metric("男胎异常", f"{male['是否异常'].sum()}条" if len(male) else "-",
              delta=f"{male['是否异常'].mean():.1%}" if len(male) else None)
    c6.metric("女胎异常", f"{female['是否异常'].sum()}条" if len(female) else "-",
              delta=f"{female['是否异常'].mean():.1%}" if len(female) else None)

    st.subheader("数据基本信息（当前筛选范围）")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**男胎数据**")
        if len(male):
            st.write(f"- 孕周范围：{male['孕周'].min():.1f} ~ {male['孕周'].max():.1f} 周")
            st.write(f"- BMI范围：{male['孕妇BMI'].min():.1f} ~ {male['孕妇BMI'].max():.1f}（均值 {male['孕妇BMI'].mean():.1f}）")
            st.write(f"- Y染色体浓度：{male['Y浓度_pct'].min():.2f}% ~ {male['Y浓度_pct'].max():.2f}%")
            st.write(f"- Y浓度达标（≥{st.session_state['threshold']:.0f}%）："
                     f"{(male['Y浓度_pct'] >= st.session_state['threshold']).sum()} 条")
    with c2:
        st.markdown("**女胎数据**")
        if len(female):
            st.write(f"- 孕周范围：{female['孕周'].min():.1f} ~ {female['孕周'].max():.1f} 周")
            st.write(f"- BMI范围：{female['孕妇BMI'].min():.1f} ~ {female['孕妇BMI'].max():.1f}（均值 {female['孕妇BMI'].mean():.1f}）")
            st.write(f"- X染色体浓度：{female['X浓度_pct'].min():.2f}% ~ {female['X浓度_pct'].max():.2f}%")
            st.write(f"- 18/21号染色体Z值绝对值>3："
                     f"{((female['18号染色体的Z值'].abs() > 3) | (female['21号染色体的Z值'].abs() > 3)).sum()} 条")

    st.subheader("交互式分布图")
    tab1, tab2, tab3, tab4 = st.tabs(["孕周分布", "BMI分布", "异常构成", "检测次数分布"])
    with tab1:
        render(analysis.fig_week_hist(male, female))
    with tab2:
        render(analysis.fig_bmi_hist(male, female))
    with tab3:
        render(analysis.fig_abnormal_pie(male, female))
    with tab4:
        render(analysis.fig_times_dist(male, female))

    st.subheader("数据预览")
    tabA, tabB, tabC, tabD = st.tabs(["男胎（清洗后）", "女胎（清洗后）", "原始Excel·男胎", "原始Excel·女胎"])
    with tabA:
        st.dataframe(male, width="stretch", height=320)
    with tabB:
        st.dataframe(female, width="stretch", height=320)
    with tabC:
        st.dataframe(data["raw_male"], width="stretch", height=320)
    with tabD:
        st.dataframe(data["raw_female"], width="stretch", height=320)


# ---------------- 页面2：数据读取与清洗 ----------------
def page_cleaning(data):
    st.title("🧹 数据读取与清洗")
    st.divider()
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
    6. **类型统一**：怀孕次数"1/2/≥3"混型列统一转字符串；个别BMI组缺失按BMI值补填
    """)

    st.header("数据质量检查（由原始Excel实时计算）")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**男胎原始数据**")
        miss = data["raw_male"].isna().sum()
        st.dataframe(pd.DataFrame({"列名": miss.index, "缺失数": miss.values})[miss.values > 0],
                     width="stretch", height=280)
        st.info(f"原始记录 {len(data['raw_male'])} 条 → 清洗后 {len(data['male'])} 条，"
                f"缺失字段：末次月经 {int(data['raw_male']['末次月经'].isna().sum())} 条、"
                f"染色体非整倍体 {int(data['raw_male']['染色体的非整倍体'].isna().sum())} 条（语义=无异常）")
    with c2:
        st.markdown("**女胎原始数据**")
        miss = data["raw_female"].isna().sum()
        st.dataframe(pd.DataFrame({"列名": miss.index, "缺失数": miss.values})[miss.values > 0],
                     width="stretch", height=280)
        st.info(f"原始记录 {len(data['raw_female'])} 条 → 清洗后 {len(data['female'])} 条，"
                f"缺失字段：孕妇BMI {int(data['raw_female']['孕妇BMI'].isna().sum())} 条、"
                f"染色体非整倍体 {int(data['raw_female']['染色体的非整倍体'].isna().sum())} 条")

    st.header("清洗前后字段对比")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**原始字段（Excel工作表）**")
        st.write("、".join(list(data["raw_male"].columns)))
    with c2:
        st.markdown("**清洗后新增/派生字段**")
        derived = [c for c in data["male"].columns if c not in data["raw_male"].columns]
        st.write("、".join(derived))
    st.markdown("""
    > 数据质量：核心字段完整度高；样本量充足（1082+605条）；每人 1~8 次追踪检测。
    > 注意：数据以高BMI人群为主（均值≈32.3），低BMI组样本量较少；类别不平衡（异常占比约11%）。
    """)


# ---------------- 页面3：指标构建 ----------------
def page_indicators(data):
    male, female = filtered(data["male"]), filtered(data["female"])
    st.title("🧮 指标构建")
    st.divider()

    st.header("派生指标字典")
    st.table(pd.DataFrame([
        ["孕周", "小数孕周", '周 + 天/7（"11w+6" → 11.86）'],
        ["孕周阶段", "孕期三段划分", "≤12周 早期 / 13-27周 中期 / ≥28周 晚期"],
        ["BMI组规范", "BMI 五档分组", "[20,28) [28,32) [32,36) [36,40) 40以上"],
        ["是否异常", "二分类标签", "染色体非整倍体非空 → 1，否则 0"],
        ["Y浓度_pct / X浓度_pct", "染色体浓度百分数", "原始浓度 × 100"],
        ["达标", "Y浓度是否达标", "Y浓度_pct ≥ 侧边栏阈值"],
    ], columns=["派生字段", "含义", "计算方法"]))

    st.header("分组统计速览（当前筛选范围）")
    if len(male):
        grp = (male.groupby("BMI组规范", observed=True)
               .agg(记录数=("孕周", "size"),
                    平均孕周=("孕周", "mean"),
                    平均BMI=("孕妇BMI", "mean"),
                    平均Y浓度=("Y浓度_pct", "mean"),
                    达标率=("Y浓度_pct", lambda s: (s >= st.session_state["threshold"]).mean()))
               .reindex([g for g in analysis.BMI_ORDER if g in male["BMI组规范"].unique()]))
        st.dataframe(grp.assign(平均孕周=grp["平均孕周"].map(lambda v: f"{v:.1f}"),
                                平均BMI=grp["平均BMI"].map(lambda v: f"{v:.1f}"),
                                平均Y浓度=grp["平均Y浓度"].map(lambda v: f"{v:.2f}%"),
                                达标率=grp["达标率"].map(fmt_pct)),
                     width="stretch")
    else:
        st.warning("当前筛选下男胎无样本，请放宽侧边栏筛选条件。")

    st.header("清洗后数据预览")
    tabA, tabB = st.tabs(["男胎", "女胎"])
    with tabA:
        st.dataframe(male, width="stretch", height=360)
    with tabB:
        st.dataframe(female, width="stretch", height=360)


# ---------------- 页面4：统计分析与可视化 ----------------
def page_stats(male):
    st.title("📈 统计分析与可视化")
    st.divider()
    if len(male) < 10:
        st.warning(f"当前筛选下男胎样本仅 {len(male)} 条，不足以进行分析，请放宽侧边栏筛选条件。")
        return

    tab1, tab2 = st.tabs(["趋势与相关性", "分组达标率"])

    with tab1:
        method = st.radio("相关系数方法", ["Pearson", "Spearman"], horizontal=True)
        render(analysis.fig_corr_heatmap(male, method))
        if method == "Pearson":
            corr = male[["Y浓度_pct", "孕周", "孕妇BMI"]].corr()
            st.markdown(
                f"**Pearson相关系数（n={len(male)}）：** Y浓度 vs 孕周 r = {corr.loc['Y浓度_pct','孕周']:+.3f}；"
                f"Y浓度 vs BMI r = {corr.loc['Y浓度_pct','孕妇BMI']:+.3f}；孕周 vs BMI r = {corr.loc['孕周','孕妇BMI']:+.3f}"
            )
        st.subheader("散点与趋势")
        render(analysis.fig_scatter_regression(male, st.session_state["threshold"]))
        st.markdown("""
        **结论：** Y浓度与孕周显著正相关（孕周每增1周约 +0.13%），与BMI显著负相关（BMI每增1约 −0.20%）；
        个体差异较大，需通过分组分析细化（见下一页与达标率页签）。
        """)

    with tab2:
        st.subheader("各BMI组Y浓度拟合曲线")
        render(analysis.fig_bmi_curves(male, st.session_state["threshold"]))
        st.caption("图例可点击开关单个BMI组；鼠标悬浮读数；滚轮缩放。")
        st.subheader("各BMI组达标率曲线（0.5周分箱 + 平滑）")
        fig_pr, _ = analysis.fig_pass_rate(male, st.session_state["threshold"])
        render(fig_pr)


# ---------------- 页面5：最佳检测时点（统计优化） ----------------
def page_optimal(data):
    male = filtered(data["male"])
    st.title("🎯 最佳NIPT检测时点 · 风险函数优化")
    st.divider()
    if len(male) < 10:
        st.warning(f"当前筛选下男胎样本仅 {len(male)} 条，不足以进行分析，请放宽侧边栏筛选条件。")
        return

    st.header("方法说明（统计优化，非机器学习）")
    st.markdown("对每个候选孕周 w 计算风险分，取**最小值**对应的孕周为该 BMI 组最佳检测时点：")
    st.latex(r"R(w) = \lambda_{fail}\,(1 - p(w)) + \lambda_{time}\,c(w)\,\max(0,\; w-10)^2")
    st.markdown("""
    - p(w)：该孕周的 Y 浓度达标率（0.5 周分箱 + 平滑曲线）
    - c(w)：孕周风险系数——早期(≤12周)=1，中期(13-27周)=5，晚期(≥28周)=20
    - 第一项惩罚**检测失败**（太早测，浓度不够）；第二项惩罚**检测太晚**（错过干预窗口）
    """)

    st.header("参数调节")
    c1, c2, c3 = st.columns(3)
    fail_pen = c1.slider("失败惩罚系数 λ_fail", 0.0, 2.0, 0.5, 0.05,
                         help="未达标（检测失败）的成本权重")
    time_pen = c2.slider("时间惩罚系数 λ_time", 0.0, 1.0, 0.1, 0.01,
                         help="孕周延后的时间成本权重")
    show_ref = c3.toggle("叠加文件记录的最优解（虚线）", value=True)

    fig7, best7 = analysis.fig_risk_tradeoff(male, st.session_state["threshold"], fail_pen, time_pen)
    if show_ref:
        for g, row in data["opt2"].iterrows():
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
        st.markdown("**当前参数重算的最优时点**")
        st.dataframe(best7.assign(达标率=best7["达标率"].map(fmt_pct)), width="stretch", hide_index=True)
    with c2:
        st.markdown("**文件记录的最优时点（原项目优化结果）**")
        st.dataframe(data["opt2"].reset_index().rename(columns={"index": "BMI组"}),
                     width="stretch", hide_index=True)

    st.header("检测误差影响")
    err = st.slider("测量误差幅度（±%，百分点）", 0.0, 2.0, 1.0, 0.1)
    render(analysis.fig_error_impact(male, st.session_state["threshold"], err))
    st.markdown("""
    **结论：**
    - BMI越低，达标越早，可在较低孕周检测；高BMI组需要更晚检测
    - 检测误差对低BMI组影响更大（接近阈值的样本多）
    """)


# ---------------- 侧边栏与主入口 ----------------
def build_sidebar(data):
    with st.sidebar:
        st.title("🩺 NIPT分析系统")
        st.divider()
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
        st.divider()
        st.caption(f"当前筛选：男胎 {len(m_f)} 条 / 女胎 {len(f_f)} 条")

        st.divider()
        page = st.radio("导航", [
            "📊 数据概览", "🧹 数据读取与清洗", "🧮 指标构建",
            "📈 统计分析与可视化", "🎯 最佳检测时点",
        ])
        st.divider()
        with st.expander("💡 使用说明"):
            st.markdown("""
- 界面全部由 **Streamlit 原生组件**实现，无自定义 HTML/CSS/JS
- 所有图表为 **Plotly 交互式**：缩放、悬浮读数、图例开关、导出PNG
- 侧边栏筛选同步作用于所有页面；阈值滑块影响达标口径
- 时点优化页的 λ 参数与误差幅度可在页面内调节并实时重算
- 数据来源：male_clean.csv / female_clean.csv / optimal_ntip_times.csv / 附件.xlsx
""")
        st.caption("NIPT时点选择与胎儿异常判定 · 生产实习（数据分析/Python Web开发）交付物")
        return page


def main():
    init_state()
    data = load_data()
    page = build_sidebar(data)
    if page == "📊 数据概览":
        page_overview(data)
    elif page == "🧹 数据读取与清洗":
        page_cleaning(data)
    elif page == "🧮 指标构建":
        page_indicators(data)
    elif page == "📈 统计分析与可视化":
        page_stats(filtered(data["male"]))
    elif page == "🎯 最佳检测时点":
        page_optimal(data)


if __name__ == "__main__":
    main()
