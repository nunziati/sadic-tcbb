"""Emit the standalone Supplementary Material LaTeX document (table + figures)."""

import os
import sys

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from paths import SUPPLEMENTARY as SUPP

PREAMBLE = r"""\documentclass[10pt]{article}
\usepackage[margin=2.2cm]{geometry}
\usepackage{graphicx}
\usepackage{longtable}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\title{Supplementary Material\\
\large SADIC: Improving Protein Atom Depth Computation Using Morphological Transformations}
\author{G.~Nunziati, S.~Marziali, F.~Romagnoli, S.~Brunetti}
\date{}
\begin{document}
\maketitle

\section*{S1. Dataset overview}
The dataset consists of 1000 protein structures, processed as their biological
assembly (assembly~1) as deposited in the Protein Data Bank. Entries were restricted to
protein-only depositions with 200--40{,}000 non-hydrogen polymer atoms in the assembly.
Figures~\ref{fig:atoms}--\ref{fig:oligo} summarise the composition of the dataset;
Table~\ref{tab:dataset} lists all entries.

\begin{figure}[h!]\centering
\includegraphics[width=3.25in]{fig_atoms_hist.pdf}
\caption{Distribution of the number of atoms of the modelled assemblies.}\label{fig:atoms}
\end{figure}
\begin{figure}[h!]\centering
\includegraphics[width=3.25in]{fig_rg_hist.pdf}
\caption{Distribution of the radius of gyration.}\label{fig:rg}
\end{figure}
\begin{figure}[h!]\centering
\includegraphics[width=3.25in]{fig_method_bar.pdf}
\caption{Experimental methods of the selected entries.}\label{fig:method}
\end{figure}
\begin{figure}[h!]\centering
\includegraphics[width=3.25in]{fig_oligomer_bar.pdf}
\caption{Oligomeric state of the biological assemblies.}\label{fig:oligo}
\end{figure}

\clearpage
\section*{S2. Dataset table}
{\footnotesize
\setlength{\tabcolsep}{3.5pt}
\begin{longtable}{llllrrlr}
\caption{The 1000 PDB entries of the dataset. Chains are those of the modelled
assembly (after normalisation); $N$ is the number of modelled atoms; $R_g$ the radius of
gyration; resolution is reported when applicable.}\label{tab:dataset}\\
\toprule
\textbf{PDB} & \textbf{Chains} & \textbf{Oligomeric state} & \textbf{Method} &
\textbf{$N$} & \textbf{$R_g$ (\AA)} & \textbf{Res. (\AA)} \\
\midrule
\endfirsthead
\toprule
\textbf{PDB} & \textbf{Chains} & \textbf{Oligomeric state} & \textbf{Method} &
\textbf{$N$} & \textbf{$R_g$ (\AA)} & \textbf{Res. (\AA)} \\
\midrule
\endhead
\bottomrule
\endfoot
"""

CLOSING = r"""\end{longtable}
}

\clearpage
\section*{S3. Sensitivity to the grid resolution: discrepancy of the depth indices}
\begin{figure}[h]\centering
\includegraphics[width=3.25in]{fig_ablation_mad.pdf}
\caption{Mean absolute discrepancy (MAD) of the SADIC depth indices with respect to the
finest configuration ($0.3$~\AA), as a function of the grid resolution ($150$-protein
stratified subset, \textsc{MorphPC} + \textsc{TranSph}, all the other parameters at their default values).}\label{fig:mad}
\end{figure}
\end{document}
"""


def latex_escape(s):
    return str(s).replace("_", r"\_").replace("&", r"\&").replace("%", r"\%")


def main():
    df = pd.read_csv(os.path.join(SUPP, "dataset_table.csv"))
    rows = []
    for _, r in df.sort_values("PDB ID").iterrows():
        chains = r["Chains"]
        if isinstance(chains, str) and len(chains) > 24:
            chains = chains[:24] + r"\dots"
        res = "" if pd.isna(r["Resolution (A)"]) else f"{r['Resolution (A)']:.2f}"
        oligo = "" if pd.isna(r["Oligomeric state"]) else str(r["Oligomeric state"])
        if len(oligo) > 22:
            oligo = oligo[:22] + r"\dots"
        rows.append(
            f"{latex_escape(r['PDB ID']).upper()} & {latex_escape(chains)} & "
            f"{latex_escape(oligo)} & {latex_escape(r['Method'])} & "
            f"{int(r['N atoms'])} & {r['Rg (A)']:.1f} & {res} \\\\"
        )
    with open(os.path.join(SUPP, "supplementary.tex"), "w") as f:
        f.write(PREAMBLE)
        f.write("\n".join(rows) + "\n")
        f.write(CLOSING)
    print(f"Wrote {os.path.join(SUPP, 'supplementary.tex')} with {len(rows)} rows")


if __name__ == "__main__":
    main()
