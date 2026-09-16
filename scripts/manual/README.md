# Manual diagnostics

These scripts contain the interactive plots, large map checks, and ad-hoc
diagnostics that previously lived inside production modules. Run them from the
repository root, for example:

```powershell
python .\scripts\manual\network_demo.py
python .\scripts\manual\simulator_demo.py
```

They intentionally remain manual because several require CUDA, map assets, or
an interactive Matplotlib window. Fast deterministic checks belong in `tests/`.
