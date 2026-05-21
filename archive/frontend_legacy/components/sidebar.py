import streamlit as st
import streamlit.components.v1 as components

def render_sidebar():
    components.html("""
    <script>
    const parentDoc = window.parent.document;
    
    // Inject TailwindCSS
    if (!parentDoc.getElementById('tailwind-cdn')) {
        const script = parentDoc.createElement('script');
        script.src = "https://cdn.tailwindcss.com?plugins=forms";
        script.id = 'tailwind-cdn';
        parentDoc.head.appendChild(script);
        
        const scriptConfig = parentDoc.createElement('script');
        scriptConfig.innerHTML = `
        tailwind.config = {
            darkMode: "class",
            theme: {
                extend: {
                    colors: {
                        "primary": "#1b9d76",
                        "navy": "#1A2B4A",
                        "background-light": "#f6f8f7",
                        "malignant": "#ef4444",
                        "benign": "#1b9d76",
                        "amber-risk": "#f59e0b",
                        "yellow-risk": "#eab308"
                    },
                    fontFamily: {
                        "display": ["Inter", "sans-serif"],
                        "serif": ["DM Serif Display", "serif"],
                        "mono": ["JetBrains Mono", "monospace"]
                    }
                }
            }
        }`;
        parentDoc.head.appendChild(scriptConfig);
    }
    
    function attachListeners() {
        const t1 = parentDoc.getElementById("nav-diagnosis");
        const t2 = parentDoc.getElementById("nav-eval");
        const t3 = parentDoc.getElementById("nav-history");

        if (!t1 || !t2 || !t3) {
            setTimeout(attachListeners, 200); return;
        }

        if (t1.dataset.jsAttached === "true") return;
        t1.dataset.jsAttached = "true";

        function switchTab(index) {
            const btns = [t1, t2, t3];
            btns.forEach(b => { 
                if(b) { 
                    b.classList.remove("text-primary", "bg-primary/10"); 
                    b.classList.add("text-slate-400", "hover:text-white"); 
                } 
            });
            if (btns[index]) { 
                btns[index].classList.remove("text-slate-400", "hover:text-white"); 
                btns[index].classList.add("text-primary", "bg-primary/10"); 
            }
            const tabs = parentDoc.querySelectorAll('[data-baseweb="tab"]');
            if (tabs && tabs[index]) { tabs[index].click(); }
        }

        t1.addEventListener("click", () => switchTab(0));
        t2.addEventListener("click", () => switchTab(1));
        t3.addEventListener("click", () => switchTab(2));
    }
    attachListeners();
    </script>
    """, height=0, width=0)

    return """
    <aside class="w-[72px] bg-navy flex flex-col items-center py-6 gap-8 shrink-0 fixed h-full z-50 top-0 left-0">
        <div class="p-2 bg-primary rounded-lg text-white">
            <span class="material-symbols-outlined text-3xl">medical_services</span>
        </div>
        <nav class="flex flex-col gap-6">
            <div id="nav-diagnosis" class="p-2 text-primary bg-primary/10 rounded-lg cursor-pointer transition-colors">
                <span class="material-symbols-outlined">dashboard</span>
            </div>
            <div id="nav-eval" class="p-2 text-slate-400 hover:text-white cursor-pointer transition-colors">
                <span class="material-symbols-outlined">analytics</span>
            </div>
            <div id="nav-history" class="p-2 text-slate-400 hover:text-white cursor-pointer transition-colors">
                <span class="material-symbols-outlined">history</span>
            </div>
            <div class="p-2 text-slate-400 hover:text-white cursor-pointer transition-colors">
                <span class="material-symbols-outlined">person</span>
            </div>
            <div class="p-2 text-slate-400 hover:text-white cursor-pointer transition-colors">
                <span class="material-symbols-outlined">settings</span>
            </div>
        </nav>
        <div class="mt-auto p-2 text-slate-400 hover:text-white cursor-pointer transition-colors">
            <span class="material-symbols-outlined">logout</span>
        </div>
    </aside>
    """
