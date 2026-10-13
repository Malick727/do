import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from part_preparation.paths import RESULTS_DIR

RL_RESULTS_DIR = RESULTS_DIR / "rl"
LOSS_COLUMNS = ["policy_loss", "value_loss", "entropy"]


def list_runs():
    return sorted(path.stem for path in (RL_RESULTS_DIR / "rewards").glob("*.csv"))


def load_logs(name):
    rewards = pd.read_csv(RL_RESULTS_DIR / "rewards" / f"{name}.csv")
    losses = pd.read_csv(RL_RESULTS_DIR / "losses" / f"{name}.csv")
    return rewards, losses


def run_color(index):
    return qualitative.Plotly[index % len(qualitative.Plotly)]


def phase_changes(log):
    changed = log["phase"].ne(log["phase"].shift())
    return log.loc[changed, ["episode", "phase"]].itertuples(index=False)


def plot_rewards(runs, window=20):
    fig = go.Figure()
    for index, name in enumerate(runs):
        rewards, _ = load_logs(name)
        color = run_color(index)
        fig.add_trace(go.Scatter(
            x=rewards["episode"], y=rewards["cost"], mode="markers",
            marker=dict(size=3, color=color, opacity=0.25),
            name=f"{name} (episode)", legendgroup=name,
        ))
        fig.add_trace(go.Scatter(
            x=rewards["episode"], y=rewards["cost"].rolling(window, min_periods=1).mean(), mode="lines",
            line=dict(color=color, width=2),
            name=f"{name} (moving average {window})", legendgroup=name,
        ))
        for episode, phase in phase_changes(rewards):
            fig.add_vline(x=episode, line=dict(color=color, dash="dot", width=1),
                          annotation_text=phase, annotation_position="top left")
    fig.update_layout(
        title="Total cost per training episode (= -total reward)",
        xaxis_title="episode", yaxis_title="cost", height=450,
    )
    return fig


def plot_losses(runs, window=20):
    fig = make_subplots(rows=len(LOSS_COLUMNS), cols=1, shared_xaxes=True, subplot_titles=LOSS_COLUMNS)
    for row, column in enumerate(LOSS_COLUMNS, start=1):
        for index, name in enumerate(runs):
            _, losses = load_logs(name)
            fig.add_trace(go.Scatter(
                x=losses["episode"], y=losses[column].rolling(window, min_periods=1).mean(), mode="lines",
                line=dict(color=run_color(index)), name=name, legendgroup=name, showlegend=row == 1,
            ), row=row, col=1)
    fig.update_yaxes(type="log", row=2, col=1)
    fig.update_xaxes(title_text="episode", row=len(LOSS_COLUMNS), col=1)
    fig.update_layout(title=f"Losses per training episode (moving average {window})", height=700)
    return fig
