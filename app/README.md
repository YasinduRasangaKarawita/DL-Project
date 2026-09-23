# Streamlit app

`streamlit_app.py` is a research-demonstration interface for a trained classifier.

Before a public demo, it must load a self-describing final release checkpoint, verify its checksum, fail closed when assets are missing, reject unsupported/OOD inputs where possible, calibrate confidence, and show an explicit non-diagnostic disclaimer.

Run locally with:

```powershell
streamlit run app/streamlit_app.py
```

