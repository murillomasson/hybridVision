import streamlit as st
import pandas as pd
from pathlib import Path
import altair as alt
import seaborn as sns
import matplotlib.pyplot as plt
import json
from streamlit_agraph import agraph, Node, Edge, Config
import numpy as np
import yaml


st.set_page_config(
    page_title="HybridVision Dashboard",
    page_icon="🧠",
    layout="centered"
)


st.markdown("""
<style>

/* Title */
h1 {
    font-size: 32px !important;
    font-weight: 600;
}

/* Header */
h2 {
    font-size: 24px !important;
    font-weight: 600;
}

/* Subheader */
h3 {
    font-size: 18px !important;
    font-weight: 600;
}

/* Sidebar titles */
section[data-testid="stSidebar"] h3 {
    font-size: 16px !important;
}

</style>
""", unsafe_allow_html=True)

st.sidebar.title("HybridVision")

st.sidebar.markdown("""
**Deep Analysis Dashboard**

_A Modular Framework for Multimodal Image Segmentation using Deep Representations and Unsupervised Clustering_
""")

@st.cache_data
def load_data(file_path):
    df = pd.read_csv(file_path)
    clean_columns = {col: col.replace('user_attrs_', '').replace('params_', '').replace('.', '_') for col in df.columns}
    df = df.rename(columns=clean_columns)
    df = df.loc[:, ~df.columns.duplicated(keep='first')]
    return df

@st.cache_data
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    
@st.cache_data
def load_markdown(md_file_path='experiments/index.md'):
    try:
        with open(md_file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return "The `experiments/index.md` file was not found."

def render_system_overview(svg_path="docs/system_overview.svg"):
    svg_content = load_svg(svg_path)

    if not svg_content:
        st.warning(f"System overview SVG not found at: {svg_path}")
        return

    st.markdown("""
    <div style="margin-top: 0.5rem; margin-bottom: 1.5rem;">
        <div style="
            font-size: 0.95rem;
            font-weight: 700;
            color: #f3f4f6;
            margin-bottom: 0.35rem;
            letter-spacing: 0.02em;
        ">
            System Overview
        </div>
        <div style="
            font-size: 0.90rem;
            color: #9ca3af;
            line-height: 1.5;
            max-width: 100%;
        ">
            High-level overview of the experimental multimodal segmentation framework, 
            including preprocessing, region generation, representation extraction, clustering, 
            semantic labeling, and validation stages.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
        <div style="
            width: 100%;
            overflow-x: auto;
            display: flex;
            justify-content: center;
            margin-top: 20px;
            margin-bottom: 10px;
        ">
            <div style="
                width: 100%;
                max-width: 100%;
            ">
                <style>
                    svg {{
                        max-width: 100% !important;
                        height: auto !important;
                        display: block;
                    }}
                </style>
                {svg_content}
            </div>
        </div>
        """, unsafe_allow_html=True)
            
    st.markdown(
    "<p style='text-align:center; opacity:0.65; font-size:0.9rem; margin-top:8px;'>"
    "Figure: high-level architecture of the HybridVision experimental segmentation pipeline."
    "</p>",
    unsafe_allow_html=True
)

def load_svg(svg_file_path: str):
    try:
        with open(svg_file_path, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return None
    

def find_experiments(base_dir='experiments'):
    experiment_map = {}
    base_path = Path(base_dir)
    if not base_path.is_dir(): return experiment_map
    for csv_file in sorted(base_path.rglob('*.csv')):
        friendly_name = str(csv_file.relative_to(base_path))
        experiment_map[friendly_name] = str(csv_file)
    return experiment_map

def find_ci_experiments(base_dir='experiments'):
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    ci_runs = [
        d for d in sorted(base_path.rglob("*"))
        if d.is_dir() and d.name.upper().startswith("CI")
    ]
    return ci_runs

def prettify_plot_title(file_path: Path) -> str:
    name = file_path.stem.replace("_", " ").strip()

    replacements = {
        "img0 img0": "",
        "dino semantic": "DINO Semantic",
        "resnet structural": "ResNet Structural",
        "fusion hybrid": "Fusion Hybrid",
        "scatter 2d": "2D Scatter",
        "silhouette": "Silhouette",
        "heatmap": "Heatmap",
        "debug validated": "Validation Debug",
        "dino k opt": "DINO k Optimization",
        "resnet k opt": "ResNet k Optimization",
        "final result": "Final Result"
    }

    name_lower = name.lower()
    for old, new in replacements.items():
        name_lower = name_lower.replace(old, new)

    title = " ".join(name_lower.split()).strip()
    return title.title()


def render_image_grid(image_paths, columns=2):
    if not image_paths:
        return

    cols = st.columns(columns)
    for i, img_path in enumerate(image_paths):
        with cols[i % columns]:
            st.caption(prettify_plot_title(img_path))
            st.image(str(img_path), use_container_width=True)

def load_yaml_data(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return None


def get_nested(cfg, path, default=None):
    cur = cfg
    for key in path:
        if not isinstance(cur, dict) or key not in cur:
            return default
        cur = cur[key]
    return cur

def summarize_ci_config(run_path: Path):
    config_candidates = sorted(run_path.rglob("config_used*.y*ml"))
    if not config_candidates:
        return {
            "experiment": run_path.name,
            "config_found": False
        }

    cfg = load_yaml_data(config_candidates[0])
    if not cfg:
        return {
            "experiment": run_path.name,
            "config_found": False
        }

    return {
        "experiment": run_path.name,
        "config_found": True,

        "run_dino": get_nested(cfg, ["run_dino_pipeline"]),
        "run_resnet": get_nested(cfg, ["run_resnet_pipeline"]),
        "run_fusion": get_nested(cfg, ["run_fusion_pipeline"]),

        "dino_clip_labeling": get_nested(cfg, ["dino_pipeline", "clustering", "run_clip_labeling"]),
        "resnet_clip_labeling": get_nested(cfg, ["resnet_pipeline", "clustering", "run_clip_labeling"]),
        "fusion_clip_labeling": get_nested(cfg, ["fusion_pipeline", "clustering", "run_clip_labeling"]),

        "semantic_threshold": get_nested(cfg, ["validator_settings", "semantic_threshold"]),
        "structural_threshold": get_nested(cfg, ["validator_settings", "structural_threshold"]),
        "final_threshold": get_nested(cfg, ["validator_settings", "final_threshold"]),

        "w_clip": get_nested(cfg, ["validator_settings", "weights", "clip_confidence"]),
        "w_dino": get_nested(cfg, ["validator_settings", "weights", "dino_iou"]),
        "w_resnet": get_nested(cfg, ["validator_settings", "weights", "resnet_iou"]),

        "sam_pred_iou": get_nested(cfg, ["heuristic_settings", "generator_params", "pred_iou_thresh"]),
        "sam_stability": get_nested(cfg, ["heuristic_settings", "generator_params", "stability_score_thresh"]),
        "sam_points_per_side": get_nested(cfg, ["heuristic_settings", "generator_params", "points_per_side"]),

        "semantic_base": get_nested(cfg, ["heuristic_settings", "k_heuristic_formulas", "semantic_base"]),
        "semantic_scale_factor": get_nested(cfg, ["heuristic_settings", "k_heuristic_formulas", "semantic_scale_factor"]),
        "structural_multiplier": get_nested(cfg, ["heuristic_settings", "k_heuristic_formulas", "structural_multiplier"]),
        "max_k": get_nested(cfg, ["heuristic_settings", "k_heuristic_formulas", "max_k"]),
    }

def build_active_branches_label(row):
    active = []
    if row.get("run_dino"):
        active.append("DINO")
    if row.get("run_resnet"):
        active.append("ResNet")
    if row.get("run_fusion"):
        active.append("Fusion")
    return " + ".join(active) if active else "None"

def render_summary_cards(df, fields, title_field=None):
    for _, row in df.iterrows():
        title = row[title_field] if title_field and title_field in df.columns else "Item"

        with st.expander(str(title), expanded=False):
            for field in fields:
                if field in df.columns:
                    value = row[field]
                    st.markdown(f"**{field.replace('_', ' ').title()}**: {value}")

st.markdown("""
<style>
    section[data-testid="stSidebar"] {
        width: 320px !important;
        min-width: 260px !important;
    }

    @media (max-width: 768px) {
        h1 { font-size: 24px !important; }
        h2 { font-size: 20px !important; }
        h3 { font-size: 16px !important; }

        section[data-testid="stSidebar"] {
            width: 85vw !important;
            min-width: 85vw !important;
        }
    }
</style>
""", unsafe_allow_html=True)

app_mode = st.sidebar.selectbox(
    "Select View",
    [ "🏠 Framework Overview", "🏛️ Architecture", "📖 Experiment Log", "🧪 Qualitative Single-Image Experiments", "📈 Hyperparameter Optimization Experiments"]
)

if app_mode == "🏠 Framework Overview":

    st.title("HybridVision")
    st.markdown("""
        ### A modular framework for multimodal image segmentation

        HybridVision combines semantic and structural deep representations with
        unsupervised clustering and validation strategies for experimental segmentation analysis.
        """)
    
    col1, col2 = st.columns(2)
    with col1:
        st.info("**Region Generation**: SAM-based proposal masks for candidate segments")
        st.info("**Feature Extraction**: DINOv2 and ResNet representations")
    with col2:
        st.info("**Clustering**: pectral and K-means based grouping strategies")
        st.info("**Validation**: CLIP labeling and multi-view consistency analysis")

    st.markdown("""
        This dashboard provides an interactive overview of the framework, its architecture,
        and the experimental results used to analyze its behavior.
        """)

    render_system_overview()

if app_mode == "🏛️ Architecture":
    st.title("🏛️ Framework Architecture")
    st.markdown("""
        This section presents the structural organization of the HybridVision framework,
        including both a high-level system overview and an implementation-level dependency graph.

        It is intended to show how the main modules, classes, and functions relate to each other
        across the segmentation, feature extraction, clustering, labeling, and validation stages.
        """)
    render_system_overview("docs/system_overview.svg")
    
    st.markdown("<hr style='margin: 1.8rem 0 1.2rem 0; border: none; border-top: 1px solid rgba(255,255,255,0.08);'>", unsafe_allow_html=True)

    st.markdown("""
        <div style="
        width: 100%;
        max-width: 100%;
        display:flex;
        justify-content:center;
    ">
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    **Code Dependency Graph**
        <div>
        This is an interactive, auto-generated diagram of the project's architecture.
    </div>
    """, unsafe_allow_html=True)
   

    legend_html = """
        <div style="
            margin-top: 14px;
            display: flex;
            justify-content: center;
            flex-wrap: wrap;
            gap: 16px;
            padding: 10px 14px;
            background-color: rgba(12,15,20,0.84);
            border: 1px solid rgba(255,255,255,0.06);
            border-radius: 12px;
            font-size: 12px;
            color: #d1d5db;
        ">
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; background:#007bff; border:1px solid rgba(255,255,255,0.20);"></div>
                <span>Class</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; background:#6c757d; border-radius:50%; border:1px solid rgba(255,255,255,0.20);"></div>
                <span>Public</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; background:#adb5bd; border-radius:50%; border:1px solid rgba(255,255,255,0.20);"></div>
                <span>Private</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:12px; height:12px; background:#ffc107; border:1px solid rgba(255,255,255,0.20);"></div>
                <span>Function</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:22px; height:2px; background:#17a2b8;"></div>
                <span>Inherits</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:22px; border-top:2px dashed #fd7e14;"></div>
                <span>Calls</span>
            </div>
            <div style="display:flex; align-items:center; gap:6px;">
                <div style="width:22px; border-top:2px dashed #e5e7eb;"></div>
                <span>Contains</span>
            </div>
        </div>
        """   
    
    architecture_data = load_json_data('docs/architecture_data.json')
    if architecture_data and 'nodes' in architecture_data and 'edges' in architecture_data:
        agraph_nodes = []
        for n in architecture_data['nodes']:
            group = n.get('group', 'class')
            color = '#007bff'
            shape = 'box'
            if group == 'public_method':
                color = '#6c757d'
                shape = 'ellipse'
            elif group == 'private_method':
                color = '#adb5bd'
                shape = 'ellipse'
            elif group == 'function' or group == 'script':
                color = '#ffc107'
                shape = 'diamond'
            
            agraph_nodes.append(Node(id=n['id'], label=n.get('label', n['id']), title=n.get('title', ''), shape=shape, color=color))

        agraph_edges = []
        for e in architecture_data['edges']:
            label = e.get('label', '')
            edge_color = '#6c757d'
            dashes = True
            if label == 'inherits':
                edge_color = '#17a2b8'; dashes = False
            elif label == 'contains':
                edge_color = '#e9ecef'; label = ''; dashes = True
            elif label == 'calls':
                edge_color = '#fd7e14'
            
            agraph_edges.append(Edge(source=e['source'], target=e['target'], label=label, color=edge_color, dashes=dashes))

        config = Config(
            width='100%',
            height=500,
            directed=True,
            physics=True,
            hierarchical=False,
            solver='barnesHut',
            barnesHut={
                'gravitationalConstant': -80000,
                'centralGravity': 0.1,
                'springLength': 300,
                'springConstant': 0.05,
                'nodeSpacing': 200
            }
        )
        st.markdown(legend_html, unsafe_allow_html=True)
        agraph(nodes=agraph_nodes, edges=agraph_edges, config=config)

    else:
        st.warning("Architecture data (`docs/architecture_data.json`) not found. Please run 'python -m scripts/generate_architecture' to create it.")

if app_mode == "📈 Hyperparameter Optimization Experiments":
    st.title("📈 Hyperparameter Optimization Experiments")
    st.markdown("""
        This section explores the hyperparameter optimization experiments conducted for the framework,
        highlighting how different parameter choices affect segmentation and validation performance.

        It includes performance-driven filtering, parameter-to-metric analysis,
        correlation inspection, and access to the raw experimental data.
        """)
    st.sidebar.markdown("---")
    st.sidebar.subheader("Experiment Controls")
    
    experiment_map = find_experiments('experiments')
    if not experiment_map:
        st.error("No valid *results.csv experiment files found in the 'experiments/' directory.")
        st.stop()
        
    selected_experiment_id = st.sidebar.selectbox("1. Select Experiment", options=list(experiment_map.keys()))
    df_full = load_data(experiment_map[selected_experiment_id])
    df = df_full.copy()
    
    st.sidebar.subheader("Analysis Settings")
    
    all_possible_metrics = sorted([
        'value', 'semantic_score', 'mean_iou', 'consistency_reward',
        'semantic_structural_alignment', 'best_dino_iou', 'best_resnet_iou',
        'iou_ratio_dino_resnet', 'avg_clip_confidence', 'num_segments',
        'num_correct_segments', 'avg_entropy_labels', 'iou_variance',
        'k_semantic_used', 'k_structural_used', 'num_sam_masks'
    ])

    available_metrics_in_df = [metric for metric in all_possible_metrics if metric in df.columns]
    
    if not available_metrics_in_df:
        st.error(f"The selected file '{selected_experiment_id}' does not contain any recognizable metric columns.")
        st.stop()
        
    metric_stdev = df_full[available_metrics_in_df].std().fillna(0)
    sorted_metric_options = metric_stdev.sort_values(ascending=False).index.tolist()
    
    selected_metric = st.sidebar.selectbox("2. Select Target Metric (Y-axis)", options=sorted_metric_options)
    top_n = st.sidebar.slider(f"3. Analyze Top N Trials (by {selected_metric})", 1, len(df), len(df))
    df = df.sort_values(by=selected_metric, ascending=False).head(top_n)
    
    st.header(f"Analyzing Experiment: `{selected_experiment_id}`")
    
    tab_dive, tab_corr, tab_data = st.tabs(["📊 Hyperparameter Deep Dive", "🔗 Correlation Analysis", "📋 Raw Data"])
    
    descriptions = load_json_data('experiments/parameter_descriptions.json') or {}
    
    with tab_dive:
        st.header("Hyperparameter Deep Dive")
        analysis_mode = st.radio("Select Analysis Mode", ["Guided (Most Relevant)", "Manual (Select from list)"], horizontal=True)
        
        if analysis_mode == "Guided (Most Relevant)":
            if pd.api.types.is_numeric_dtype(df[selected_metric]):
                numeric_params = df.select_dtypes(include=['float64', 'int64']).columns.drop(available_metrics_in_df + ['number'], errors='ignore')
                correlations = df[numeric_params].corrwith(df[selected_metric]).abs().sort_values(ascending=False)
                num_to_show = st.slider("Number of top parameters to display", 1, len(correlations), min(5, len(correlations)))
                top_params = correlations.head(num_to_show).index
                
                for param in top_params:
                    st.markdown("---")
                    description = descriptions.get(param, "No description available.")
                    st.info(f"**What is `{param}`?**\n\n{description}")
                    st.subheader(f"`{selected_metric}` vs. `{param}` (Abs. Correlation: {correlations[param]:.2f})")
                    chart = alt.Chart(df).mark_circle(size=80, opacity=0.6).encode(
                        x=alt.X(f'{param}:Q', scale=alt.Scale(zero=False)), 
                        y=alt.Y(f'{selected_metric}:Q'), 
                        tooltip=list(df.columns)
                    ).interactive()
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.warning(f"Guided analysis is only available for numeric metrics. '{selected_metric}' is not numeric.")
        
        else: 
            param_columns = sorted([col for col in df.columns if col not in all_possible_metrics + ['number', 'state', 'error', 'datetime_start', 'datetime_complete', 'duration']])
            selected_param = st.selectbox("Select Hyperparameter to Analyze (X-axis)", options=param_columns)
            
            if selected_param:
                description = descriptions.get(selected_param, "No description available.")
                st.info(f"**What is `{selected_param}`?**\n\n{description}")
                st.subheader(f"`{selected_metric}` vs. `{selected_param}`")
                
                if pd.api.types.is_numeric_dtype(df[selected_param]):
                    chart = alt.Chart(df).mark_circle(size=80, opacity=0.6).encode(
                        x=alt.X(f'{selected_param}:Q', scale=alt.Scale(zero=False)), 
                        y=alt.Y(f'{selected_metric}:Q'), 
                        tooltip=list(df.columns)
                    ).interactive()
                else:
                    chart = alt.Chart(df).mark_boxplot(extent='min-max').encode(
                        x=alt.X(f'{selected_param}:N', axis=alt.Axis(labelAngle=-45)), 
                        y=alt.Y(f'{selected_metric}:Q'), 
                        tooltip=list(df.columns)
                    ).interactive()
                st.altair_chart(chart, use_container_width=True)

    with tab_corr:
        st.header("Correlation Analysis")
        st.markdown("This heatmap shows the Pearson correlation between all numeric hyperparameters and performance metrics. It helps identify which parameters have the strongest influence (positive or negative) on the results.")

        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        if 'number' in numeric_cols: numeric_cols.remove('number')
        
        if len(numeric_cols) < 2:
            st.warning("Not enough numeric columns available for a correlation matrix.")
        else:
            corr_matrix = df[numeric_cols].corr()
            
            fig, ax = plt.subplots(figsize=(max(12, len(numeric_cols)*0.5), max(12, len(numeric_cols)*0.5)))
            sns.heatmap(corr_matrix, annot=True, cmap='viridis', fmt=".2f", linewidths=.5, ax=ax, annot_kws={"size": 10})
            plt.title("Full Correlation Matrix", fontsize=16)
            st.pyplot(fig)
    
    with tab_data:
        st.header("Raw Data Explorer")

        compact_raw_view = st.toggle("Compact mobile view", value=True, key="raw_compact_view")

        if compact_raw_view:
            preview_cols = st.multiselect(
                "Columns to display",
                options=df.columns.tolist(),
                default=df.columns.tolist()[:5],
                key="raw_preview_cols"
            )
            render_summary_cards(
                df[preview_cols].reset_index(drop=True),
                fields=preview_cols,
                title_field=None
            )
        else:
            st.dataframe(df, use_container_width=True)

elif app_mode == "📖 Experiment Log":
    st.title("📖 Experiment Log")
    st.markdown("""
        This section documents the experimental trajectory of the project,
        including decisions, adjustments, intermediate findings, and implementation notes.

        It serves as a running record of how the framework evolved throughout development and testing.
        """)
    log_content = load_markdown('experiments/index.md')
    st.markdown(log_content, unsafe_allow_html=True)

elif app_mode == "🧪 Qualitative Single-Image Experiments":
    st.title("🧪 Qualitative Single-Image Experiments")
    st.markdown("""
        This section presents qualitative results from single-image experiments,
        allowing visual inspection of segmentation behavior under different configurations.

        It includes final outputs, optimization plots, validation diagnostics,
        and intermediate cluster visualizations for individual runs.
        """)

    ci_runs = find_ci_experiments()
    if not ci_runs:
        st.warning("No CI experiments found.")
        st.stop()

    run_options = {
        str(run.relative_to("experiments")): run
        for run in ci_runs
    }

    st.sidebar.markdown("---")
    st.sidebar.subheader("Analysis Settings")

    ci_mode = st.sidebar.radio(
        "View Mode",
        ["Global Summary", "Inspect Individual Run"],
        key="ci_view_mode"
    )

    if ci_mode == "Global Summary":
        st.header("Global CI Summary")

        ci_summary_rows = [summarize_ci_config(run) for run in ci_runs]
        df_ci_summary = pd.DataFrame(ci_summary_rows)

        if not df_ci_summary.empty:
            df_ci_summary["active_branches"] = df_ci_summary.apply(
                build_active_branches_label, axis=1
            )

            default_cols = [
            "experiment",
            "active_branches",
            "final_threshold",
            "w_clip",
            "sam_pred_iou"
            ]

            selected_cols = st.multiselect(
                "Columns to compare",
                options=df_ci_summary.columns.tolist(),
                default=[c for c in default_cols if c in df_ci_summary.columns],
                key="ci_global_cols"
            )

            compact_view = st.toggle("Compact mobile view", value=True, key="ci_compact_view")

            if compact_view:
                render_summary_cards(
                    df_ci_summary[selected_cols],
                    fields=[c for c in selected_cols if c != "experiment"],
                    title_field="experiment"
                )
            else:
                st.dataframe(df_ci_summary[selected_cols], use_container_width=True)
        else:
            st.info("No CI configuration summaries found.")

    else:
        st.sidebar.markdown("---")
        st.sidebar.subheader("Experiment Controls")

        selected_run_label = st.sidebar.selectbox(
            "Select Experiment",
            options=list(run_options.keys()),
            key="ci_run_select"
        )

        run_path = run_options[selected_run_label]

        st.header(f"Run: {selected_run_label}")

        final_images = sorted(run_path.rglob("final_result_*.png"))
        dino_imgs = sorted(run_path.rglob("dino_k_opt*.png"))
        resnet_imgs = sorted(run_path.rglob("resnet_k_opt*.png"))
        debug_imgs = sorted(run_path.rglob("debug_validated*"))

        if final_images:
            st.subheader("Final Segmentation Results")
            render_image_grid(final_images, columns=2)

        if dino_imgs:
            st.subheader("DINO K Optimization")
            render_image_grid(dino_imgs, columns=2)

        if resnet_imgs:
            st.subheader("ResNet K Optimization")
            render_image_grid(resnet_imgs, columns=2)

        if debug_imgs:
            st.subheader("Validation Debug")
            render_image_grid(debug_imgs, columns=2)

        st.subheader("Cluster Diagnostics")

        cluster_dirs = [
            p for p in sorted(run_path.rglob("*"))
            if p.is_dir() and p.name.lower() in {"graph", "kmeans"}
        ]

        if cluster_dirs:
            for cluster_dir in cluster_dirs:
                st.markdown(f"### `{cluster_dir.relative_to(run_path)}`")

                cluster_images = sorted(
                    [
                        p for p in cluster_dir.iterdir()
                        if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"}
                    ]
                )

                if cluster_images:
                    render_image_grid(cluster_images, columns=3)
                else:
                    st.info(f"No images found in `{cluster_dir.name}`.")
        else:
            st.info("No cluster diagnostic folders found.")