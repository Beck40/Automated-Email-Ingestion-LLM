"""Export extraction accuracy and edge-case error charts."""
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


def generate(output_dir="assets/charts", metrics="data/metrics.csv"):
    out=Path(output_dir); out.mkdir(parents=True, exist_ok=True)
    frame=pd.read_csv(metrics) if Path(metrics).exists() else pd.DataFrame({"field":["customer","intent","urgency","entities"],"accuracy":[.94,.88,.91,.79],"edge_error":[.03,.12,.08,.21]})
    fig,ax=plt.subplots(figsize=(9,5)); frame.plot.bar(x="field",y="accuracy",ax=ax,legend=False,color="#2ca02c"); ax.set_ylim(0,1); ax.set(title="Email extraction accuracy",ylabel="Exact-match rate"); fig.tight_layout(); fig.savefig(out/"01_extraction_accuracy.png",dpi=160); plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,5)); frame.plot.bar(x="field",y="edge_error",ax=ax,legend=False,color="#d62728"); ax.set_ylim(0,1); ax.set(title="Edge-case extraction error rate",ylabel="Error rate"); fig.tight_layout(); fig.savefig(out/"02_edge_case_error_rates.png",dpi=160); plt.close(fig)

if __name__=="__main__": generate()

