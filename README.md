# bessyopt

This is a simple repository where I try to put I practice what I'm learning about BESS optimisation.

It's mostly about OR application. Here you will not find anything disruptive in particular about forecasts, but rather very clean and solid code to handle the decision making in bettery dispatch.

In `documentation` you will find the papers I'm studying about the topic. In case you know better reference please send me a message!


## Purpose
Learn OR by applying it in real world problems.

In papers folder you will find papers I studied and implemented.

## Roadmap 

✅ day ahead  
✅ rolling intraday  
✅ linearisation of utility function (for risk management)  
✅ co-located assets (pv/wind + bess dispatch)
⬜ stochastic optimisation (handle uncertainty in forecasts)

## Problem formulation

Indices: $t \in \{1,\dots,T\}$ time steps of length $\Delta t$.

Decision variables (shared by both problems):

- $c_t \ge 0$ — battery charge [MW]
- $d_t \ge 0$ — battery discharge [MW]
- $u_t \in \{0,1\}$ — charge/discharge mutual-exclusion switch
- $g_t^{\text{in}}, g_t^{\text{out}} \ge 0$ — grid sell/buy [MW]
- $s_t = s_0 + \sum_{k \le t}\!\big(\eta_c\, c_k - d_k/\eta_d\big)$ — state of charge [MWh]

Common physical constraints (battery + grid):

$$
\begin{aligned}
g_t^{\text{out}} + d_t + \mathrm{pv}_t &= g_t^{\text{in}} + c_t \\
c_t &\le u_t\, P_c\, \Delta t \\
d_t &\le (1-u_t)\, P_d\, \Delta t \\
\underline{s}\,C \le\; s_t &\;\le \overline{s}\,C \\
\sum_t d_t / C &\le N_{\text{cycles}}
\end{aligned}
$$

with optional terminal constraint $s_T \ge s_{\text{end}}$.

### Auction (day-ahead)

Standalone:

$$
\max_{\,c_t,\, d_t,\, u_t,\, g_t^{\text{in}},\, g_t^{\text{out}}} \;\; \sum_t p_t\,(g_t^{\text{in}} - g_t^{\text{out}}) \;-\; \kappa \sum_t d_t
$$

Residual mode — given an existing schedule $(\bar c_t, \bar d_t)$, decompose the deltas
$c_t - \bar c_t = \Delta c_t^{+} - \Delta c_t^{-}$ and
$d_t - \bar d_t = \Delta d_t^{+} - \Delta d_t^{-}$ with all $\Delta \ge 0$, define

$$
c_t^{r} = \Delta c_t^{+} + \Delta d_t^{-}, \qquad
d_t^{r} = \Delta d_t^{+} + \Delta c_t^{-},
$$

and price only the incremental trade:

$$
\max_{\,c_t,\, d_t,\, u_t,\, g_t^{\text{in}},\, g_t^{\text{out}},\, \Delta c_t^{\pm},\, \Delta d_t^{\pm}} \;\; \sum_t p_t\,(d_t^{r} - c_t^{r}) \;-\; \kappa \sum_t d_t
$$

### Intraday (rolling-intrinsic)

Same residual decomposition as above, against the DA position $(\bar c_t, \bar d_t)$. Imbalance long/short prices $p_t^{\ell}, p_t^{s}$ remunerate sell/buy residuals separately:

$$
\max_{\,c_t,\, d_t,\, u_t,\, g_t^{\text{in}},\, g_t^{\text{out}},\, \Delta c_t^{\pm},\, \Delta d_t^{\pm}} \;\; \sum_t \Big( p_t^{\ell}\, d_t^{r} - p_t^{s}\, c_t^{r} \Big) \;-\; \kappa \sum_t d_t
$$

Symbols: $p_t$ DA price, $p_t^{\ell}/p_t^{s}$ imbalance long/short prices, $C$ capacity, $P_c, P_d$ max charge/discharge power, $\eta_c, \eta_d$ efficiencies, $\underline{s}, \overline{s}$ SoC bounds (fractions of $C$), $N_{\text{cycles}}$ daily cycle cap, $\kappa$ degradation cost.



## Credits
[This is my personal page, have a look!](https://clarkmaio.github.io/)