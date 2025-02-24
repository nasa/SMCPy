import pandas as pd
import seaborn as sns

from pathlib import Path

FILEPATH = Path(__file__).parent / "smc_samples.csv"

df = pd.read_csv(FILEPATH)
sns.pairplot(
    df[["a", "b", "oe_cov0", "oe_cov1", "oe_cov2"]],
    diag_kind="kde",
    corner=True,
    plot_kws={"alpha": 1.0},
)
sns.mpl.pyplot.savefig(FILEPATH.parent / "smc_pairwise.png")
