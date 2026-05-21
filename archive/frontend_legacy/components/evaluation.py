import streamlit as st
import os
import base64
from backend.metrics import graph_path

def show_graph(path):
    if path and os.path.exists(path):
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        return f'<img src="data:image/png;base64,{b64}" class="w-full h-full object-contain mix-blend-multiply opacity-90"/>'
    return f'<div class="h-full w-full flex items-center justify-center text-slate-400 text-xs text-center p-4">Graph not generated</div>'

def render_evaluation(metrics, threshold):
    if not metrics:
        st.markdown(f'<div class="h-64 flex flex-col items-center justify-center text-slate-400"><span class="material-symbols-outlined text-6xl opacity-30">analytics</span><p class="font-bold mt-4">Run python evaluate_model.py first</p></div>', unsafe_allow_html=True)
        return

    acc  = metrics.get("accuracy", 0)
    auc  = metrics.get("auc_roc", 0)
    cm   = metrics.get("confusion_matrix", [[0,0],[0,0]])
    tn, fp_v, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    total_eval = tn + fp_v + fn + tp
    if total_eval == 0: total_eval = 1
    
    GDIR = "outputs/graphs"

    st.markdown(f"""
    <div class="flex justify-between items-end mb-8">
        <div>
            <h2 class="text-3xl font-serif text-navy">Core Performance Metrics</h2>
            <p class="text-slate-500">Model Version: v3.0 (Stable Release)</p>
        </div>
        <div class="flex gap-2">
            <button class="px-4 py-2 bg-white border border-slate-200 rounded-lg text-sm font-medium flex items-center gap-2 text-slate-700 hover:bg-slate-50 transition-colors shadow-sm"><span class="material-symbols-outlined text-sm">download</span> Export PDF</button>
            <button class="px-4 py-2 bg-primary text-white hover:bg-[#147A5C] transition-colors rounded-lg text-sm font-medium flex items-center gap-2 shadow-sm"><span class="material-symbols-outlined text-sm">refresh</span> Re-run Eval</button>
        </div>
    </div>

    <!-- Row 1: Key Charts -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-6">
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Confusion Matrix</h3>
            <div class="aspect-square bg-slate-50 rounded-lg overflow-hidden flex flex-col p-2">
                <div class="flex-1 grid grid-cols-2 grid-rows-2 gap-1 pb-2 border-b border-slate-100/50">
                    <div class="bg-primary/90 text-white flex flex-col items-center justify-center rounded-lg shadow-sm">
                        <span class="text-xs opacity-80 uppercase font-bold tracking-widest mb-1">TN</span>
                        <span class="text-2xl font-mono shadow-sm">{tn}</span>
                    </div>
                    <div class="bg-primary/10 text-primary flex flex-col items-center justify-center rounded-lg border border-primary/20">
                        <span class="text-xs opacity-80 uppercase font-bold tracking-widest mb-1">FP</span>
                        <span class="text-2xl font-mono">{fp_v}</span>
                    </div>
                    <div class="bg-primary/20 text-primary flex flex-col items-center justify-center rounded-lg border border-primary/20">
                        <span class="text-xs opacity-80 uppercase font-bold tracking-widest mb-1">FN</span>
                        <span class="text-2xl font-mono">{fn}</span>
                    </div>
                    <div class="bg-primary/70 text-white flex flex-col items-center justify-center rounded-lg shadow-sm">
                        <span class="text-xs opacity-80 uppercase font-bold tracking-widest mb-1">TP</span>
                        <span class="text-2xl font-mono">{tp}</span>
                    </div>
                </div>
                <div class="pt-2 text-[10px] flex justify-between text-slate-400 font-bold uppercase tracking-widest">
                    <span>Total N: {total_eval} Cases</span>
                    <span>Accuracy: {acc*100:.1f}%</span>
                </div>
            </div>
        </div>
        
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 relative">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">ROC Curve</h3>
            <div class="aspect-square flex flex-col items-center justify-center bg-slate-50 rounded-lg relative overflow-hidden">
                <div class="absolute inset-2">{show_graph(os.path.join(GDIR, "roc_curve.png"))}</div>
            </div>
            <div class="absolute bottom-10 right-10 text-right bg-white/80 p-2 rounded-lg backdrop-blur shadow-sm">
                <p class="text-[10px] uppercase font-bold text-slate-400 tracking-widest">AUC Score</p>
                <p class="text-2xl font-mono font-bold text-primary leading-none">{auc:.3f}</p>
            </div>
        </div>
        
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Performance Radar</h3>
            <div class="aspect-square bg-slate-50 rounded-lg flex items-center justify-center overflow-hidden p-2">
                {show_graph(os.path.join(GDIR, "metrics_radar.png"))}
            </div>
        </div>
    </div>
    
    <!-- Row 2: Secondary Charts -->
    <div class="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 relative">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Precision-Recall Curve</h3>
            <div class="h-48 bg-slate-50 rounded-lg relative p-2">
                {show_graph(os.path.join(GDIR, "precision_recall.png"))}
            </div>
            <div class="absolute top-10 right-10 px-3 py-1 bg-white border border-slate-100 shadow-sm text-xs font-mono text-primary font-bold rounded-lg">mAP: {metrics.get('average_precision', 0):.3f}</div>
        </div>
        
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Probability Distribution</h3>
            <div class="h-48 bg-slate-50 rounded-lg p-2">
                {show_graph(os.path.join(GDIR, "probability_histogram.png"))}
            </div>
        </div>
        
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Class Distribution</h3>
            <div class="h-48 bg-slate-50 rounded-lg p-2">
                {show_graph(os.path.join(GDIR, "class_distribution.png"))}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Threshold Calibration Table Generation
    ta = metrics.get("threshold_analysis", [])
    t_rows = ""
    for row in ta:
        is_active = abs(row['t'] - threshold) < 0.03
        tr_cls = "bg-primary/10 text-primary border-transparent" if is_active else "border-b border-slate-50"
        t_rows += f"""<tr class="{tr_cls}">
            <td class="py-2.5 px-4 font-bold">{"★ " if is_active else ""}{row["t"]:.2f}</td>
            <td class="py-2.5 px-4 font-mono">{row["acc"]*100:.1f}%</td>
            <td class="py-2.5 px-4 font-mono">{row["sens"]*100:.1f}%</td>
            <td class="py-2.5 px-4 font-mono">{row["spec"]*100:.1f}%</td>
            <td class="py-2.5 px-4 font-mono">{row["f1"]:.3f}</td>
        </tr>"""

    st.markdown(f"""
    <!-- Row 3: Threshold Calibration -->
    <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 mb-8">
        <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Threshold Calibration</h3>
        <div class="grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div class="h-64 bg-slate-50 rounded-lg flex items-center justify-center p-2">
                {show_graph(os.path.join(GDIR, "threshold_analysis.png"))}
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left text-sm">
                    <thead class="border-b border-slate-200">
                        <tr>
                            <th class="py-3 px-4 text-xs tracking-widest uppercase font-bold text-slate-400">Threshold</th>
                            <th class="py-3 px-4 text-xs tracking-widest uppercase font-bold text-slate-400">Accuracy</th>
                            <th class="py-3 px-4 text-xs tracking-widest uppercase font-bold text-slate-400">Sensitivity</th>
                            <th class="py-3 px-4 text-xs tracking-widest uppercase font-bold text-slate-400">Specificity</th>
                            <th class="py-3 px-4 text-xs tracking-widest uppercase font-bold text-slate-400">F1 Score</th>
                        </tr>
                    </thead>
                    <tbody>{t_rows}</tbody>
                </table>
            </div>
        </div>
    </div>
    
    <!-- Row 4: Risk and Fairness -->
    <div class="grid grid-cols-1 lg:grid-cols-5 gap-6 mb-8">
        <!-- Risk Distribution (2/5) -->
        <div class="lg:col-span-2 bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Risk Distribution</h3>
            <div class="h-48 bg-slate-50 rounded-lg flex items-center justify-center p-2">
                {show_graph(os.path.join(GDIR, "risk_distribution.png"))}
            </div>
        </div>
        
        <!-- Fairness Analysis (3/5) -->
        <div class="lg:col-span-3 bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider mb-4">Fairness Analysis (By Skin Tone)</h3>
            <div class="h-48 relative bg-slate-50 rounded-lg p-2">
                {show_graph(os.path.join(GDIR, "fairness_analysis.png"))}
                <div class="absolute bottom-4 inset-x-0 text-center text-[10px] text-slate-400">Metric: Accuracy per Fitzpatrick Scale classification</div>
            </div>
        </div>
    </div>
    
    <!-- Row 5: Training Curves -->
    <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 mb-8">
        <div class="flex justify-between items-center mb-6">
            <h3 class="text-sm font-bold text-slate-500 uppercase tracking-wider">Training Dynamics</h3>
            <div class="flex gap-4">
                <div class="flex items-center gap-2"><div class="w-3 h-3 bg-primary rounded-full"></div><span class="text-xs font-bold text-slate-600">Accuracy</span></div>
                <div class="flex items-center gap-2"><div class="w-3 h-3 bg-navy rounded-full"></div><span class="text-xs font-bold text-slate-600">Loss</span></div>
            </div>
        </div>
        <div class="h-64 bg-slate-50 rounded-lg p-2 relative">
            {show_graph(os.path.join("outputs", "training_curves.png"))}
        </div>
    </div>
    """, unsafe_allow_html=True)
