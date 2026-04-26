import numpy as np
import matplotlib.pyplot as plt


def plot_da_schedule(
    battery_charge: np.ndarray,
    battery_discharge: np.ndarray,
    soc_values: np.ndarray,
    daprice: np.ndarray,
    pnl_values: np.ndarray,
    pv: np.ndarray = None,
    demand: np.ndarray = None,
    figsize=None,
    return_fig: bool = False,
    show_soc: bool = True,
    show_price: bool = True,
    show_pnl: bool = True,
    show_pv: bool = True,
    show_demand: bool = True,
    x = None,
):
    
    if x is None:
        x = np.arange(len(daprice))

    nplots = 1
    if show_soc:
        nplots += 1
    if show_price:
        nplots += 1
    if show_pnl:
        nplots += 1
    if show_pv and pv is not None:
        nplots += 1
    if show_demand and demand is not None:
        nplots += 1

    fig, ax = plt.subplots(nplots, 1, figsize=figsize)

    idx = 0
    ax[idx].bar(x=x, height=battery_charge, color='#2ecc71')
    ax[idx].bar(x=x, height=-battery_discharge, color='#e74c3c')
    ax[idx].axhline(y=0, color='black')
    ax[idx].set_ylabel('MWh')
    ax[idx].set_title('Charge / Discharge')

    if show_soc:
        idx += 1
        ax[idx].fill_between(x, soc_values, alpha=0.6, color='#3498db', step='post')
        ax[idx].step(x, soc_values, linewidth=0.8, color='#3498db', where='post')
        ax[idx].set_ylabel('MWh')
        ax[idx].set_title('State of Charge')

    if show_price:
        idx += 1
        ax[idx].step(x, daprice, color='#9b59b6', where='post')
        ax[idx].set_ylabel('€/MWh')
        ax[idx].set_title('DA Price')

    if show_pv and pv is not None:
        idx += 1
        ax[idx].step(x, pv, color='#f39c12', where='post')
        ax[idx].set_ylabel('MWh')
        ax[idx].set_title('PV Generation')

    if show_demand and demand is not None:
        idx += 1
        ax[idx].step(x, demand, color='#e67e22', where='post')
        ax[idx].set_ylabel('MWh')
        ax[idx].set_title('Demand')

    if show_pnl:
        idx += 1
        ax[idx].bar(x=x, height=pnl_values, color=np.where(pnl_values >= 0, '#2ecc71', '#e74c3c'))
        ax[idx].axhline(y=0, color='black')
        ax[idx].set_ylabel('€')
        ax[idx].set_title('PnL')
        ax2 = ax[idx].twinx()
        ax2.plot(x, np.cumsum(pnl_values), color='#2980b9', linewidth=1.5)
        ax2.set_ylabel('Cumulative €', color='#2980b9')
        ax2.tick_params(axis='y', labelcolor='#2980b9')

    for a in ax[:-1]:
        a.tick_params(labelbottom=False)

    fig.suptitle('Optimisation schedule', fontweight='bold')
    plt.tight_layout()

    if return_fig:
        return fig
    plt.show()
