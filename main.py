import streamlit as st
import pandas as pd
import numpy as np
import joblib
import re
from sklearn.linear_model import LinearRegression
from difflib import SequenceMatcher
import random
import matplotlib.pyplot as plt
import seaborn as sns

# 设置页面配置
st.set_page_config(
    page_title="钢板划痕风险预测系统",
    page_icon="🛡️",
    layout="wide"
)

# 页面标题
st.title("🛡️ 钢板划痕风险预测与优化系统")
st.markdown("通过选择钢板参数和加工设备，预测划痕风险并获取优化建议")


# 加载模型
@st.cache_resource
def load_model():
    try:
        with st.spinner("正在加载预测模型..."):
            # 请将模型路径替换为您的实际模型路径
            model_data = joblib.load("优化后PLS_RF最优模型.pkl")

            # 验证模型组件
            required_components = ['model', 'encoder_pls', 'pls_model', 'encoder_other',
                                   'feature_names', 'train_proc', 'best_pls_components']
            missing_components = [comp for comp in required_components if comp not in model_data]
            if missing_components:
                st.error(f"模型缺少关键组件: {missing_components}")
                return None

            st.success("模型加载成功！")
            return model_data
    except Exception as e:
        st.error(f"模型加载失败: {str(e)}")
        return None


# 数据转换函数
def convert_scratch_resistance(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, str):
        value = value.strip().replace('N', '').replace('牛顿', '')
        numbers = re.findall(r'\d+\.?\d*', value)
        if numbers:
            val = np.mean([float(num) for num in numbers])
            return max(0.5, val)
        return np.nan
    val = float(value)
    return max(0.5, val)


def unify_gloss_format(value):
    if pd.isna(value):
        return np.nan
    value = str(value).strip().lower()
    text_map = {'低光泽': 5, '中光泽': 20, '高光泽': 40,
                'low': 5, 'medium': 20, 'high': 40,
                '低': 5, '中': 20, '高': 40}
    if value in text_map:
        return text_map[value]
    nums = re.findall(r'\d+\.?\d*', value)
    if nums:
        val = float(nums[0])
        return max(0, min(100, val))
    return np.nan


# 预处理函数
def preprocess_data(input_data, model_data):
    df = pd.DataFrame([input_data])

    # 处理数值特征
    df['耐划痕标准'] = df['耐划痕标准'].apply(convert_scratch_resistance)
    df['光泽度'] = df['光泽度'].apply(unify_gloss_format)

    # 填充空值
    train_proc = model_data['train_proc']
    fill_values = {
        '耐划痕标准': train_proc['耐划痕标准'].median(),
        '光泽度': train_proc['光泽度'].median()
    }
    for col, fill_val in fill_values.items():
        df[col].fillna(fill_val, inplace=True)

    # 处理分类特征
    category_cols = ['钢板类型', '颜色深浅度']
    train_categories = {
        col: set(train_proc[col].astype(str).str.strip().str.replace(' ', '').str.replace('-', '').str.lower())
        for col in category_cols
    }

    for col in category_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(' ', '').str.replace('-', '').str.lower()
        unknown_cats = set(df[col].unique()) - train_categories[col]
        if unknown_cats:
            df[col] = df[col].apply(lambda x: '其他' if x in unknown_cats else x)

    # 处理设备类型
    equipment_cols = ['开料设备', '封边设备', '打孔设备']
    train_equipment = {
        col: set(train_proc[col].astype(str).str.strip().str.replace(' ', '').str.replace('-', '').str.lower())
        for col in equipment_cols
    }

    for col in equipment_cols:
        df[col] = df[col].astype(str).str.strip().str.replace(' ', '').str.replace('-', '').str.lower()

        unknown_equipment = set(df[col].unique()) - train_equipment[col]
        if unknown_equipment:
            fuzzy_matches = {}
            for unknown in unknown_equipment:
                max_similarity = 0
                best_match = None
                for train_eq in train_equipment[col]:
                    sim = SequenceMatcher(None, unknown, train_eq).ratio()
                    if sim > max_similarity and sim > 0.6:
                        max_similarity = sim
                        best_match = train_eq
                if best_match:
                    fuzzy_matches[unknown] = best_match

            for unknown, train_eq in fuzzy_matches.items():
                df[col] = df[col].replace(unknown, train_eq)

            remaining_unknown = set(df[col].unique()) - train_equipment[col]
            if remaining_unknown:
                df[col] = df[col].apply(lambda x: '其他' if x in remaining_unknown else x)

    # 创建设备组合特征
    df['开料-封边组合'] = df['开料设备'] + "_" + df['封边设备']

    return df


# 生成特征函数
def generate_features(preprocessed_data, model_data):
    coupled_cols = ['钢板类型', '颜色深浅度', '光泽度']
    X_pls = model_data['encoder_pls'].transform(preprocessed_data[coupled_cols])
    pls_features = model_data['pls_model'].transform(X_pls)
    pls_cols = [f'pls_{i + 1}' for i in range(pls_features.shape[1])]
    pls_df = pd.DataFrame(pls_features, columns=pls_cols)

    other_cols = ['开料设备', '封边设备', '打孔设备', '开料-封边组合', '耐划痕标准']
    X_other = model_data['encoder_other'].transform(preprocessed_data[other_cols])
    other_feature_names = model_data['encoder_other'].get_feature_names_out(other_cols)
    other_df = pd.DataFrame(X_other, columns=other_feature_names)

    sample_features = pd.concat([pls_df, other_df], axis=1)
    missing_features = [col for col in model_data['feature_names'] if col not in sample_features.columns]
    for col in missing_features:
        sample_features[col] = 0

    return sample_features[model_data['feature_names']]


# 分析设备性能
def analyze_equipment_performance(model_data):
    train_proc = model_data['train_proc']
    analysis = {}

    # 单设备性能分析
    for col in ['开料设备', '封边设备', '打孔设备']:
        eq_risk = train_proc.groupby(col, observed=True).agg({
            '划痕风险概率': ['mean', 'count']
        }).round(6)
        eq_risk.columns = ['平均风险', '样本数']
        analysis[col] = eq_risk[eq_risk['样本数'] >= 2].sort_values('平均风险')

    # 设备组合分析
    combo_risk = train_proc.groupby('开料-封边组合', observed=True).agg({
        '划痕风险概率': ['mean', 'count']
    }).round(6)
    combo_risk.columns = ['平均风险', '样本数']
    analysis['设备组合'] = combo_risk[combo_risk['样本数'] >= 3].sort_values('平均风险')

    # 颜色深浅度与风险关系
    color_risk = train_proc.groupby('颜色深浅度', observed=True).agg({
        '划痕风险概率': ['mean', 'count']
    }).round(6)
    color_risk.columns = ['平均风险', '样本数']
    analysis['颜色深浅度'] = color_risk[color_risk['样本数'] >= 2].sort_values('平均风险')

    # 耐划痕标准与风险关系
    scratch_risk = train_proc.groupby(pd.cut(train_proc['耐划痕标准'], bins=5), observed=True).agg({
        '划痕风险概率': 'mean'
    }).round(6)
    scratch_risk.columns = ['平均风险']
    analysis['耐划痕标准'] = scratch_risk

    return analysis


# 生成优化建议
def generate_optimization_suggestions(input_data, current_risk, model_data):
    analysis = analyze_equipment_performance(model_data)
    suggestions = []
    used_equipment = {'开料设备': [], '封边设备': [], '打孔设备': [], '开料-封边组合': []}

    # 确定风险等级
    if current_risk >= 0.003:
        risk_level = "高风险"
        target_reduction = 0.4
    elif current_risk >= 0.001:
        risk_level = "中风险"
        target_reduction = 0.3
    else:
        risk_level = "低风险"
        target_reduction = 0.1

    target_risk = current_risk * (1 - target_reduction)

    # 1. 设备组合优化建议
    current_combo = f"{input_data['开料设备']}_{input_data['封边设备']}"
    if current_combo in analysis['设备组合'].index:
        current_combo_risk = analysis['设备组合'].loc[current_combo, '平均风险']
        better_combos = analysis['设备组合'][analysis['设备组合']['平均风险'] < current_combo_risk]

        if not better_combos.empty:
            # 优先选择未推荐过的组合
            unused_combos = [combo for combo in better_combos.index
                             if combo not in used_equipment['开料-封边组合']]

            if unused_combos:
                best_combo = unused_combos[0]
            else:
                best_combo = better_combos.index[0]

            # 更新历史记录
            used_equipment['开料-封边组合'].append(best_combo)
            if len(used_equipment['开料-封边组合']) > 10:
                used_equipment['开料-封边组合'].pop(0)

            reduction = ((current_combo_risk - better_combos.loc[best_combo, '平均风险'])
                         / current_combo_risk * 100)
            suggestions.append({
                "priority": 1,
                "desc": f"优化设备组合：{current_combo} → {best_combo}（预计降低风险{reduction:.1f}%）",
                "type": "设备组合优化"
            })

    # 2. 单设备优化建议
    for eq_type in ['开料设备', '封边设备']:
        current_eq = input_data[eq_type]
        if current_eq in analysis[eq_type].index:
            current_eq_risk = analysis[eq_type].loc[current_eq, '平均风险']
            better_eq = analysis[eq_type][analysis[eq_type]['平均风险'] < current_eq_risk]

            if not better_eq.empty:
                # 优先选择未推荐过的设备
                unused_eq = [eq for eq in better_eq.index if eq not in used_equipment[eq_type]]

                if unused_eq:
                    best_eq = unused_eq[0]
                else:
                    best_eq = better_eq.index[0]

                # 更新历史记录
                used_equipment[eq_type].append(best_eq)
                if len(used_equipment[eq_type]) > 10:
                    used_equipment[eq_type].pop(0)

                reduction = ((current_eq_risk - better_eq.loc[best_eq, '平均风险'])
                             / current_eq_risk * 100)
                suggestions.append({
                    "priority": 2,
                    "desc": f"更换{eq_type}：{current_eq} → {best_eq}（预计降低风险{reduction:.1f}%）",
                    "type": f"{eq_type}优化"
                })

    # 3. 颜色调整建议
    current_color = input_data['颜色深浅度']
    if current_color in analysis['颜色深浅度'].index:
        current_color_risk = analysis['颜色深浅度'].loc[current_color, '平均风险']
        better_colors = analysis['颜色深浅度'][analysis['颜色深浅度']['平均风险'] < current_color_risk]

        if not better_colors.empty:
            best_color = better_colors.index[0]
            reduction = ((current_color_risk - better_colors.loc[best_color, '平均风险'])
                         / current_color_risk * 100)
            suggestions.append({
                "priority": 2,
                "desc": f"调整颜色深浅度：{current_color} → {best_color}（预计降低风险{reduction:.1f}%）",
                "type": "颜色优化"
            })

    # 4. 耐划痕标准优化建议
    current_scratch = input_data['耐划痕标准']
    target_scratch = current_scratch * (1 + min(target_reduction * 2, 0.5))  # 适度提高
    suggestions.append({
        "priority": 3,
        "desc": f"提高耐划痕标准：{current_scratch:.1f}N → {target_scratch:.1f}N（预计降低风险{target_reduction * 100:.0f}%）",
        "type": "耐划痕标准优化"
    })

    # 按优先级排序建议
    suggestions.sort(key=lambda x: x["priority"])

    # 根据风险等级选择适当数量的建议
    if risk_level == "高风险":
        selected_suggestions = suggestions[:3]
    elif risk_level == "中风险":
        selected_suggestions = suggestions[:2]
    else:
        selected_suggestions = suggestions[:1]

    return selected_suggestions, target_risk


# 预测函数
def predict_risk(input_data, model_data):
    try:
        # 预处理数据
        preprocessed_data = preprocess_data(input_data, model_data)

        # 生成特征
        features = generate_features(preprocessed_data, model_data)

        # 预测风险
        risk_probability = model_data['model'].predict(features)[0]

        # 生成优化建议
        suggestions, optimized_risk = generate_optimization_suggestions(
            input_data, risk_probability, model_data)

        return {
            "success": True,
            "risk_probability": risk_probability,
            "suggestions": suggestions,
            "optimized_risk": optimized_risk
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e)
        }


# 主函数
def main():
    # 加载模型
    model_data = load_model()
    if model_data is None:
        return

    # 创建两列布局
    col1, col2 = st.columns([1, 2])

    with col1:
        st.sidebar.header("参数设置")

        # 1. 钢板类型
        steel_type = st.sidebar.selectbox(
            "1. 钢板类型",
            options=["布纹", "针织布纹", "方格面", "高低光", "荔枝皮纹", "零度手抓纹",
                     "零度细直纹", "零度小木刺", "零度雨丝纹", "绒麻面", "沙贝面",
                     "细麻面", "细丝纹"]
        )

        # 2. 光泽度
        gloss = st.sidebar.slider(
            "2. 光泽度（数值越高光泽越强）",
            min_value=5,
            max_value=40,
            step=5,
            value=20,
            help="低光泽：5，中光泽：20，高光泽：40"
        )

        # 3. 颜色深浅度
        color_depth = st.sidebar.radio(
            "3. 颜色深浅度",
            options=["浅色", "中色", "深色"],
            horizontal=True
        )

        # 4. 耐划痕标准
        scratch_resistance = st.sidebar.number_input(
            "4. 耐划痕标准（单位：N）",
            min_value=1.0,
            max_value=10.0,
            value=3.0,
            step=0.5,
            help="常见值：3N（基础）、5N（中等）、7N（高耐刮）"
        )

        # 5. 开料设备
        cutting_machine = st.sidebar.selectbox(
            "5. 开料设备",
            options=["套铣", "南兴380电子据", "豪迈380电子锯", "豪迈300电子据",
                     "谢林电子锯", "推台锯", "极动电子据", "南兴大板套铣",
                     "欧登多精密推台锯", "马氏精密推台锯", "鼎力加工中心", "其他"]
        )

        # 6. 封边设备
        edge_banding_machine = st.sidebar.selectbox(
            "6. 封边设备",
            options=["KAL610封边机", "NKL210封边机", "豪迈210封边机", "豪迈350封边机",
                     "南兴封边机", "包覆机", "极东668J封边机", "极东自动封边机",
                     "南兴自动封边机", "南兴高速自动封边机", "其他"]
        )

        # 7. 打孔设备
        drilling_machine = st.sidebar.selectbox(
            "7. 打孔设备",
            options=["T-E9SL", "KN-2309P", "极东2309", "豪迈自动打孔机",
                     "拓雕单通道六面钻", "俊驭门铰机", "马氏钻孔机", "南兴六面钻",
                     "鼎力设备", "极东数控钻孔中心", "其他"]
        )

        # 预测按钮
        predict_button = st.sidebar.button("预测风险", key="predict", use_container_width=True)

    with col2:
        # 显示输入参数摘要
        st.subheader("参数摘要")
        input_data = {
            "钢板类型": steel_type,
            "光泽度": gloss,
            "颜色深浅度": color_depth,
            "耐划痕标准": scratch_resistance,
            "开料设备": cutting_machine,
            "封边设备": edge_banding_machine,
            "打孔设备": drilling_machine
        }

        # 以表格形式展示参数
        param_df = pd.DataFrame(list(input_data.items()), columns=["参数", "值"])
        st.dataframe(param_df, use_container_width=True, hide_index=True)

        # 预测结果展示
        if predict_button:
            with st.spinner("正在预测风险并生成优化建议..."):
                result = predict_risk(input_data, model_data)

                if result["success"]:
                    # 显示原始风险
                    st.subheader("📊 风险预测结果")

                    # 风险等级判断
                    risk_prob = result["risk_probability"]
                    if risk_prob >= 0.003:
                        risk_level = "高风险"
                        risk_color = "red"
                    elif risk_prob >= 0.001:
                        risk_level = "中风险"
                        risk_color = "orange"
                    else:
                        risk_level = "低风险"
                        risk_color = "green"

                    # 显示风险值和等级
                    col_risk1, col_risk2 = st.columns(2)
                    with col_risk1:
                        st.metric(
                            label="预测划痕风险概率",
                            value=f"{risk_prob:.6f}",
                            delta="",
                            delta_color="inverse"
                        )
                    with col_risk2:
                        st.markdown(
                            f"**风险等级**: <span style='color:{risk_color};font-size:1.2em'>{risk_level}</span>",
                            unsafe_allow_html=True)

                    # 可视化风险
                    fig, ax = plt.subplots(figsize=(8, 2))
                    ax.barh(["风险概率"], [risk_prob], color=risk_color, height=0.5)
                    ax.set_xlim(0, max(0.005, risk_prob * 1.5))
                    ax.set_xlabel('风险概率')
                    ax.grid(axis='x', linestyle='--', alpha=0.7)
                    st.pyplot(fig)

                    # 显示优化建议
                    st.subheader("💡 优化建议")
                    suggestions = result["suggestions"]

                    for i, suggestion in enumerate(suggestions, 1):
                        with st.expander(f"建议 #{i}: {suggestion['type']}", expanded=True):
                            st.info(suggestion["desc"])

                    # 显示优化后风险
                    st.subheader("✨ 优化后预期效果")
                    optimized_risk = result["optimized_risk"]

                    # 优化后风险等级
                    if optimized_risk >= 0.003:
                        optimized_level = "仍为高风险"
                        optimized_color = "red"
                    elif optimized_risk >= 0.001:
                        optimized_level = "降至中风险"
                        optimized_color = "orange"
                    else:
                        optimized_level = "降至低风险"
                        optimized_color = "green"

                    # 计算风险降低百分比
                    risk_reduction = ((risk_prob - optimized_risk) / risk_prob) * 100

                    col_opt1, col_opt2, col_opt3 = st.columns(3)
                    with col_opt1:
                        st.metric(
                            label="优化后风险概率",
                            value=f"{optimized_risk:.6f}",
                            delta=f"{risk_reduction:.2f}%",
                            delta_color="normal"
                        )
                    with col_opt2:
                        st.markdown(
                            f"**优化后等级**: <span style='color:{optimized_color};font-size:1.2em'>{optimized_level}</span>",
                            unsafe_allow_html=True)
                    with col_opt3:
                        st.metric(
                            label="风险降低幅度",
                            value=f"{risk_reduction:.2f}%",
                            delta="",
                            delta_color="normal"
                        )

                    # 优化前后对比图
                    fig, ax = plt.subplots(figsize=(8, 3))
                    bars = ax.bar(["优化前", "优化后"], [risk_prob, optimized_risk],
                                  color=[risk_color, optimized_color], width=0.6)
                    ax.set_ylim(0, max(0.005, risk_prob * 1.5))
                    ax.set_ylabel('风险概率')
                    ax.grid(axis='y', linestyle='--', alpha=0.7)

                    # 在柱状图上添加数值
                    for bar in bars:
                        height = bar.get_height()
                        ax.text(bar.get_x() + bar.get_width() / 2., height,
                                f'{height:.6f}',
                                ha='center', va='bottom')

                    st.pyplot(fig)
                else:
                    st.error(f"预测失败: {result['error']}")


if __name__ == "__main__":
    main()
