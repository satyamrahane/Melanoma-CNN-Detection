def render_topbar(model_name, gpu_str, acc_str, auc_str):
    return f"""
    <header class="bg-white border-b border-slate-200 px-8 py-4 sticky top-0 z-40" style="margin-left:-32px; margin-right:-32px; margin-bottom: 24px;">
        <div class="flex items-center justify-between">
            <h1 class="font-serif text-2xl text-navy">MelanomaAI Diagnostic Portal</h1>
            <div class="flex items-center gap-4">
                <div class="flex items-center gap-2 bg-primary/10 px-3 py-1.5 rounded-full border border-primary/20">
                    <div class="w-2 h-2 bg-primary rounded-full animate-pulse"></div>
                    <span class="text-xs font-bold text-primary tracking-wider uppercase">Model Active</span>
                </div>
                <div class="flex flex-col items-end">
                    <span class="text-[10px] uppercase font-bold text-slate-400">Compute</span>
                    <span class="text-sm font-mono text-slate-700">{gpu_str}</span>
                </div>
                <div class="h-8 w-[1px] bg-slate-200 mx-2"></div>
                <div class="text-right">
                    <p class="text-sm font-semibold text-slate-900 leading-none">Dr. Sarah Jenkins</p>
                    <p class="text-[10px] text-slate-500 mt-1">Chief Dermatologist</p>
                </div>
                <div class="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-primary font-bold border-2 border-primary">SJ</div>
            </div>
        </div>
    </header>
    """
