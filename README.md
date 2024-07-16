# mldl24-vpr
In this work we address the Visual Place Recognition (VPR) task which consists in localizing a place depicted in a query image. Only computer vision's techniques are exploited. In the first place, we train our model according to the GSV-Cities framework. Next, we test our best model on two benchmark datasets. Finally, we analyze and compare the results obtained from using different model aggregators, loss functions, mining techniques and optimizers. We also visualize some queries and their predictions to better understand the reasoning behind our model's decisions. We show that the model utilizing MixVPR as aggregator outperforms other evaluated configurations.
Performance of different aggregators:
\begin{table}[H]
    
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
         Aggregator & R1 & R5 & R10 & R15 & R20 \\
         \hline
         AVG & 53.85 & 69.41 & 75.20 & 78.51 & 80.76 \\
         \hline
         GeM & 56.66 & 71.66 & 77.56 & 80.73 & 82.73 \\
         \hline
         ConvAP & 71.64 & 79.06 & 82.03 & 83.92 & 85.14 \\
         \hline
         MixVPR & \textbf{76.27} & \textbf{83.50} & \textbf{86.60} & \textbf{88.43} & \textbf{89.42} \\
         \hline
    \end{tabular}
    \caption{Results obtained on SF$\_$XS val.}
    \label{tab:Res_MSL_SFval}
\end{table}

\begin{table}[H]
    
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
         Aggregator & R1 & R5 & R10 & R15 & R20 \\
         \hline
         AVG & 21.30 & 34.30 & 41.30 & 44.70 & 47.20 \\
         \hline
         GeM & 23.30 & 35.60 & 43.30 & 47.20 & 49.50 \\
         \hline
         ConvAP & 46.40 & 58.30 & 65.00 & 67.90 & 69.80 \\
         \hline
         MixVPR & \textbf{55.10} & \textbf{66.50} & \textbf{71.50} & \textbf{73.30} & \textbf{74.80} \\
         \hline
    \end{tabular}
    \caption{Results obtained on SF$\_$XS test.}
    \label{tab:Res_MSL_SFtest}
\end{table}

\begin{table}[H]
    
    \begin{tabular}{|c|c|c|c|c|c|}
        \hline
         Aggregator & R1 & R5 & R10 & R15 & R20 \\
         \hline
         AVG & 29.21 & 47.62 & 55.24 & 63.17 & 68.25 \\
         \hline
         GeM & 35.24 & 56.19 & 62.22 & 69.52 & 74.60 \\
         \hline
         ConvAP & 68.89 & 82.86 & 86.98 & 86.98 & 87.62 \\
         \hline
         MixVPR & \textbf{73.65} & \textbf{85.40} & \textbf{89.52} & \textbf{91.43} & \textbf{93.65} \\
         \hline
    \end{tabular}
    \caption{Results obtained on Tokyo$\_$XS.}
    \label{tab:Res_MSL_Tokyo}
\end{table}
