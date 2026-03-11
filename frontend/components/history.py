import streamlit as st
import numpy as np
import io, base64
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def fig_to_b64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()

def session_chart(history):
    if len(history) < 2: return None
    probs = [h["prob"] for h in history]
    risks = [h["risk_score"] for h in history]
    labels = [h["label"] for h in history]
    
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.5))
    fig.patch.set_alpha(0)
    c = ["#ef4444" if l == "MALIGNANT" else "#1b9d76" for l in labels]
    
    for ax in [a1, a2]:
        ax.set_facecolor("none")
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#e2e8f0")
        ax.spines['bottom'].set_color("#e2e8f0")
        
    a1.bar(range(len(probs)), probs, color=c, alpha=.9, width=0.6, edgecolor="none")
    a1.axhline(0.85, color="#f59e0b", ls="--", lw=1.5, label="Threshold")
    a1.set_ylim(0, 1.05)
    a1.set_xlabel("Scan Number", fontsize=8, color="#64748b")
    a1.set_ylabel("Probability", fontsize=8, color="#64748b")
    a1.legend(fontsize=7.5, frameon=False, loc="upper left")
    a1.set_title("Session Probabilities", fontsize=10, loc="left", color="#334155", fontweight="bold")
    a1.grid(True, axis="y", alpha=.4, color="#e2e8f0", linestyle="dashed")
    
    rc = ["#ef4444" if r >= 72 else "#f59e0b" if r >= 52 else "#eab308" if r >= 32 else "#1b9d76" for r in risks]
    a2.bar(range(len(risks)), risks, color=rc, alpha=.9, width=0.6, edgecolor="none")
    for t in [32, 52, 72]: a2.axhline(t, color="#e2e8f0", ls=":", lw=1.5)
    a2.set_ylim(0, 105)
    a2.set_xlabel("Scan Number", fontsize=8, color="#64748b")
    a2.set_ylabel("Risk Score", fontsize=8, color="#64748b")
    a2.set_title("Risk Score History", fontsize=10, loc="left", color="#334155", fontweight="bold")
    a2.grid(True, axis="y", alpha=.4, color="#e2e8f0", linestyle="dashed")
    
    plt.tight_layout()
    return fig_to_b64(fig)

def render_history(history):
    if not history:
        st.markdown("""
        <div class="h-64 flex flex-col items-center justify-center text-slate-400">
            <span class="material-symbols-outlined text-6xl opacity-30">history</span>
            <p class="font-bold mt-4">No scans yet this session</p>
        </div>
        """, unsafe_allow_html=True)
        return

    total  = len(history)
    mal_n  = sum(1 for h in history if h["label"] == "MALIGNANT")
    crit   = sum(1 for h in history if h.get("risk_score", 0) >= 72)
    mod_r  = sum(1 for h in history if 32 <= h.get("risk_score", 0) < 52)
    low_r  = sum(1 for h in history if h.get("risk_score", 0) < 32)
    fp_est = mod_r + low_r
    avg_p  = np.mean([h["prob"] for h in history])

    st.markdown(f"""
    <div class="grid grid-cols-1 md:grid-cols-3 lg:grid-cols-5 gap-4 mb-8">
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <p class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Total Scans</p>
            <p class="font-mono text-3xl font-bold text-navy">{total}</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <p class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Flagged Malignant</p>
            <p class="font-mono text-3xl font-bold text-malignant">{mal_n}</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <p class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Critical Cases</p>
            <p class="font-mono text-3xl font-bold text-malignant">{crit}</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <p class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Avg Probability</p>
            <p class="font-mono text-3xl font-bold text-navy">{avg_p*100:.0f}%</p>
        </div>
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100">
            <p class="text-slate-500 text-xs font-bold uppercase tracking-wider mb-2">Est False Positives</p>
            <p class="font-mono text-3xl font-bold text-navy">{fp_est}</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    b64 = session_chart(history)
    if b64:
        st.markdown(f"""
        <div class="bg-white p-6 rounded-xl shadow-sm border border-slate-100 mb-8">
            <div class="flex justify-between items-center mb-6">
                <h3 class="font-serif text-lg text-navy">Session Analysis</h3>
                <span class="text-xs font-bold text-slate-400 uppercase tracking-widest">Target: 85%</span>
            </div>
            <img src="data:image/png;base64,{b64}" class="w-full mix-blend-multiply opacity-90"/>
        </div>
        """, unsafe_allow_html=True)

    rows = ""
    for i, h in enumerate(reversed(history)):
        rs  = h.get("risk_score", 0)
        rl  = h.get("risk_level", "LOW")
        is_m = h["label"] == "MALIGNANT"
        
        rc_hex = "#ef4444" if rs >= 72 else "#f59e0b" if rs >= 52 else "#eab308" if rs >= 32 else "#1b9d76"
        v_cls  = "bg-malignant/10 text-malignant" if is_m else "bg-benign/10 text-benign"
        v_txt  = "Malignant" if is_m else "Benign"
        
        prob_pct = int(h["prob"] * 100)
        p_cls = "bg-malignant" if is_m else "bg-amber-risk" if prob_pct >= 50 else "bg-benign"
        
        if rs >= 72:
            a_cls = "bg-malignant"; a_txt = "Refer Now"
        elif rs >= 52:
            a_cls = "bg-amber-risk"; a_txt = "1-Week"
        elif rs >= 32:
            a_cls = "bg-yellow-risk"; a_txt = "Review"
        else:
            a_cls = "bg-benign"; a_txt = "Routine"

        rows += f"""
        <tr class="hover:bg-slate-50 transition-colors border-b border-slate-50 font-mono text-sm leading-relaxed">
            <td class="px-6 py-4 text-slate-500">{len(history)-i:03d}</td>
            <td class="px-6 py-4 font-sans text-slate-700">{h["time"]}</td>
            <td class="px-6 py-4"><span class="px-3 py-1 {v_cls} rounded-full font-bold text-xs">{v_txt}</span></td>
            <td class="px-6 py-4">
                <div class="flex items-center gap-2">
                    <div class="flex-1 h-1.5 bg-slate-100 rounded-full overflow-hidden max-w-[80px]">
                        <div class="h-full {p_cls}" style="width: {prob_pct}%"></div>
                    </div>
                    <span class="text-slate-700">{prob_pct}%</span>
                </div>
            </td>
            <td class="px-6 py-4 font-sans font-bold" style="color:{rc_hex}">{rl}</td>
            <td class="px-6 py-4" style="color:{rc_hex}">{rs:.2f}</td>
            <td class="px-6 py-4"><span class="px-4 py-1.5 {a_cls} text-white rounded-lg text-xs font-bold shadow-sm">{a_txt}</span></td>
        </tr>
        """

    st.markdown(f"""
    <div class="bg-white rounded-xl shadow-sm border border-slate-100 overflow-hidden mb-8">
        <div class="overflow-x-auto">
            <table class="w-full text-left">
                <thead class="bg-slate-50 border-b border-slate-100">
                    <tr>
                        <th class="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">#</th>
                        <th class="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Time</th>
                        <th class="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Verdict</th>
                        <th class="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Probability %</th>
                        <th class="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Risk Level</th>
                        <th class="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Score</th>
                        <th class="px-6 py-4 text-xs font-bold text-slate-400 uppercase tracking-widest">Action</th>
                    </tr>
                </thead>
                <tbody>{rows}</tbody>
            </table>
        </div>
        <div class="p-4 bg-slate-50 border-t border-slate-100 flex justify-between items-center text-xs">
            <span class="text-slate-500 font-medium">Displaying all {total} results</span>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    col1,col2,col3 = st.columns([6,2,2])
    with col3:
        if st.button("🗑 Clear Session", use_container_width=True):
            st.session_state.history = []
            st.rerun()
