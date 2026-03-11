import streamlit as st
import numpy as np
import os
from PIL import Image
from datetime import datetime
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.model import predict
from backend.pdf_report import generate_report

def render_diagnosis(model, threshold, metrics):
    c_left, c_right = st.columns([1, 1], gap="large")

    with c_left:
        st.markdown(
        '<div class="bg-white rounded-xl border border-slate-200 p-8 shadow-sm space-y-6">'
            '<h3 class="font-serif text-xl text-navy mb-6">Analysis Input</h3>'
            '<div class="border-2 border-dashed border-slate-200 rounded-xl p-10 flex flex-col items-center justify-center text-center cursor-pointer hover:border-primary transition-colors bg-slate-50 mb-6 relative">'
                '<span class="material-symbols-outlined text-4xl text-primary mb-3">cloud_upload</span>'
                '<p class="font-bold text-slate-700">Drop dermatoscopy image here</p>'
                '<p class="text-xs text-slate-500 mt-1">Supports JPEG, PNG up to 20MB</p>', 
        unsafe_allow_html=True)
        
        # Transparent overlay for file uploader to sit on top of styled area
        st.markdown('<div style="position:absolute;top:0;left:0;width:100%;height:100%;opacity:0;cursor:pointer">', unsafe_allow_html=True)
        img_arr = None
        up = st.file_uploader("Upload", type=["jpg","jpeg","png"], label_visibility="collapsed")
        st.markdown('</div></div>', unsafe_allow_html=True)
        
        if up:
            img_arr = np.array(Image.open(up).convert("RGB"))
        else:
            # Fallback test samples if no upload
            st.markdown('<p class="text-xs font-bold text-slate-500 uppercase tracking-wider mb-2">Or try test samples:</p>', unsafe_allow_html=True)
            bc, mc = st.columns(2)
            with bc:
                bd = "data/processed/benign"
                if os.path.exists(bd):
                    for f in sorted(os.listdir(bd))[:2]:
                        if f.lower().endswith(('.jpg','.jpeg','.png')):
                            if st.button(f"🟢 {f[:18]}", key=f"b_{f}"):
                                img_arr = np.array(Image.open(os.path.join(bd,f)).convert("RGB"))
            with mc:
                md = "data/processed/malignant"
                if os.path.exists(md):
                    for f in sorted(os.listdir(md))[:2]:
                        if f.lower().endswith(('.jpg','.jpeg','.png')):
                            if st.button(f"🔴 {f[:18]}", key=f"m_{f}"):
                                img_arr = np.array(Image.open(os.path.join(md,f)).convert("RGB"))

        st.markdown('<div class="space-y-6 mt-6">', unsafe_allow_html=True)
        thr_ui  = st.slider("Certainty Threshold", .20, .80, float(threshold), .01)
        
        col1, col2 = st.columns(2)
        with col1: use_rob = st.toggle("Robustness Layer", True)
        with col2: abcd_on = st.toggle("ABCD Analysis", True)
        st.markdown('</div></div>', unsafe_allow_html=True)

        if img_arr is not None:
            st.markdown(
            '<div class="bg-white rounded-xl border border-slate-200 overflow-hidden shadow-sm mt-6">'
                '<div class="p-4 border-b border-slate-100 flex justify-between items-center">'
                    '<span class="text-xs font-bold text-slate-500 uppercase">Image Preview</span>'
                    '<button class="text-primary hover:bg-primary/10 p-1 rounded-md transition-colors"><span class="material-symbols-outlined text-sm">fullscreen</span></button>'
                '</div>'
                '<div class="aspect-video bg-slate-100">',
            unsafe_allow_html=True)
            st.image(img_arr, use_container_width=True)
            st.markdown('</div></div>', unsafe_allow_html=True)

    with c_right:
        result = None
        if img_arr is not None and model is not None:
            result = predict(model, img_arr, thr_ui, use_rob)
            st.session_state.last_result = result
            st.session_state.history.append(dict(time=datetime.now().strftime("%I:%M %p"), prob=result["prob"], label=result["label"], risk_score=result["risk"]["score"], risk_level=result["risk"]["level"]))
            if len(st.session_state.history) > 30: st.session_state.history = st.session_state.history[-30:]

        if result:
            prob = result["prob"]; label = result["label"]
            risk = result["risk"]; is_m  = label == "MALIGNANT"
            rs   = risk["score"]
            v_cls = "bg-malignant/10 text-malignant border border-malignant/20" if is_m else "bg-primary/10 text-primary border border-primary/20"
            r_col = "#ef4444" if rs>=72 else "#f59e0b" if rs>=52 else "#eab308" if rs>=32 else "#1b9d76"

            st.markdown(f"""
            <div class="bg-white rounded-xl border border-slate-200 p-8 shadow-sm space-y-8">
                <div class="flex justify-between items-start">
                    <div>
                        <h3 class="font-serif text-2xl text-navy">Diagnostic Results</h3>
                        <p class="text-slate-500 text-sm">Patient ID: #SKN-{datetime.now().strftime('%H%M')}</p>
                    </div>
                    <div class="{v_cls} px-4 py-1.5 rounded-full font-bold text-sm tracking-wider uppercase">
                        Verdict: {label}
                    </div>
                </div>

                <div class="flex flex-col md:flex-row items-center gap-10 py-4">
                    <div class="relative w-40 h-40">
                        <svg class="w-full h-full -rotate-90" viewBox="0 0 36 36">
                            <path class="stroke-slate-100" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke-width="3"></path>
                            <path stroke="{r_col}" d="M18 2.0845 a 15.9155 15.9155 0 0 1 0 31.831 a 15.9155 15.9155 0 0 1 0 -31.831" fill="none" stroke-dasharray="{int(prob*100)}, 100" stroke-linecap="round" stroke-width="3"></path>
                        </svg>
                        <div class="absolute inset-0 flex flex-col items-center justify-center">
                            <span class="font-mono text-4xl font-bold" style="color:{r_col}">{int(prob*100)}%</span>
                            <span class="text-[10px] uppercase font-bold text-slate-400">Probability</span>
                        </div>
                    </div>
                    
                    <div class="flex-1 space-y-4 w-full">
                        <div class="flex justify-between items-center">
                            <span class="text-sm font-bold text-slate-700">Risk Assessment</span>
                            <span class="text-white text-[10px] font-bold px-2 py-0.5 rounded tracking-widest uppercase" style="background:{r_col}">{risk['level']}</span>
                        </div>
                        <div class="h-4 w-full bg-slate-100 rounded-full overflow-hidden flex relative border border-slate-200">
                            <div class="h-full" style="width:20%;background:#1b9d76"></div>
                            <div class="h-full" style="width:30%;background:#eab308"></div>
                            <div class="h-full" style="width:50%;background:#ef4444"></div>
                            <div class="absolute top-0 bottom-0 w-1 bg-navy/80 rounded" style="left:{min(rs,98)}%;box-shadow:0 0 4px rgba(0,0,0,0.5)"></div>
                        </div>
                        <div class="flex justify-between text-[10px] font-bold text-slate-400 uppercase tracking-tighter">
                            <span>Low</span><span>Moderate</span><span>High</span>
                        </div>
                    </div>
                </div>

                <div class="grid grid-cols-2 gap-4">
                    <div class="bg-slate-50 p-4 rounded-xl">
                        <p class="text-[10px] uppercase font-bold text-slate-400 mb-1">Skin Tone Indicator</p>
                        <div class="flex items-center gap-3">
                            <div class="w-4 h-4 rounded-full border border-slate-200" style="background-color:{result['tone_col']}"></div>
                            <span class="text-sm font-mono font-bold">ITA Angle: {result['ita']:.1f}°</span>
                        </div>
                    </div>
                    <div class="bg-slate-50 p-4 rounded-xl">
                        <p class="text-[10px] uppercase font-bold text-slate-400 mb-1">Preprocessing Status</p>
                        <div class="flex items-center gap-2">
                            <span class="material-symbols-outlined text-primary text-sm">check_circle</span>
                            <span class="text-sm font-bold">CLAHE Active</span>
                        </div>
                    </div>
                </div>

                <div class="space-y-3">
                    <h4 class="text-sm font-bold text-slate-700 uppercase tracking-wide">Model Rationale</h4>
                    <p class="text-sm text-slate-600 leading-relaxed bg-slate-50 p-4 rounded-lg">{risk['action']}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("🧠 Explain Features", use_container_width=True):
                    st.session_state.show_explainer = not st.session_state.get("show_explainer", False)
            with c2:
                report = generate_report(result, metrics, thr_ui)
                st.download_button("📄 Download Report", data=report, file_name=f"melanoma_report.txt", mime="text/plain", use_container_width=True)

            if st.session_state.get("show_explainer", False):
                st.markdown(f"""
                <div class="bg-white rounded-xl border border-slate-200 p-6 mt-4 shadow-sm">
                    <div class="text-sm font-bold text-navy mb-4 uppercase tracking-wider">Risk Score Breakdown</div>
                    <div class="space-y-4">
                        <div class="flex gap-3 pb-3 border-b border-slate-100">
                            <div class="w-2.5 h-2.5 rounded-full bg-primary mt-1.5 shrink-0"></div>
                            <div><div class="text-sm font-bold text-navy">Base Probability · {risk['factors']['base_probability']:.1f} pts</div>
                            <div class="text-xs text-slate-500 mt-1">Raw model output × 65</div></div>
                        </div>
                        <div class="flex gap-3 pb-3 border-b border-slate-100">
                            <div class="w-2.5 h-2.5 rounded-full bg-blue-500 mt-1.5 shrink-0"></div>
                            <div><div class="text-sm font-bold text-navy">Confidence Factor · {risk['factors']['confidence']:.1f} pts</div>
                            <div class="text-xs text-slate-500 mt-1">Distance from threshold.</div></div>
                        </div>
                        <div class="flex gap-3">
                            <div class="w-2.5 h-2.5 rounded-full bg-amber-500 mt-1.5 shrink-0"></div>
                            <div><div class="text-sm font-bold text-navy">Skin Reliability · {risk['factors']['skin_reliability']:.1f} pts</div>
                            <div class="text-xs text-slate-500 mt-1">Dark skin weighted lower ({result['reliability']*100:.0f}% reliable).</div></div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        else:
            st.markdown("""
            <div class="bg-white rounded-xl border border-slate-200 shadow-sm flex flex-col items-center justify-center p-24 text-center">
                <span class="material-symbols-outlined text-6xl text-slate-200 mb-4">analytics</span>
                <p class="font-bold text-slate-500">Upload an image to see diagnostic results</p>
            </div>
            """, unsafe_allow_html=True)
