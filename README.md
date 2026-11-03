# ARI510
A repo for ARI510 assignments and projects
Assignment 1 is an attempt to compare different classifiers k-nearest neighbors, logistic regression,
and support vector machines.
## Notebook setup on Windows

Install Python 3, then run from the repository root:

```bat
setup\setup.bat
```

The script creates `.venv` at the repository root and installs
`setup/requirements.txt`. Running it again reuses the environment.
The `.venv/` directory is excluded from Git.

Start Jupyter from the repository root:

```bat
.venv\Scripts\python.exe -m notebook
```

Open the notebook under `notebooks/`. In VS Code, you can instead open the
notebook and select `.venv\Scripts\python.exe` in the kernel picker (with the
Python and Jupyter extensions installed).

Place `heart_disease_uci.csv` alongside the assignment notebook before running
its cells. The optional Colab upload cell is skipped when running locally.
