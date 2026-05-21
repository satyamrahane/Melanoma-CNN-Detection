# Legacy frontend (archived)

Pre-multipage UI code has been moved to `archive/frontend_legacy/`.

**Use instead:**
- `app.py` (MelanomaAI v2 Streamlit dashboard)
- `pages/`, `stitch_shared.py`, `stitch_ui.css`
- `demo_samples/` for viva quick-load

Do not import from this folder.

**Note:** An experimental Flask UI may exist on the remote history under `frontend/`; production entry point for v2 is `streamlit run app.py` at repo root.
