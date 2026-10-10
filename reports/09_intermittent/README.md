# Phase 9 - Intermittent Demand Methods

## Item level: 500 sampled item-store series

Each model is fitted separately on every series, using the last
365 days before each origin (Christmas closures interpolated). RMSSE = per-series
RMSE scaled by that series' naive in-sample error, averaged; WAPE and bias are volume-weighted.

| model                    |   RMSSE (backtest) |   RMSSE (holdout) |   WAPE (backtest) |   bias (backtest) |
|:-------------------------|-------------------:|------------------:|------------------:|------------------:|
| TSB                      |              0.739 |             0.776 |             0.741 |            -0.024 |
| ADIDA                    |              0.740 |             0.779 |             0.744 |            -0.007 |
| IMAPA                    |              0.742 |             0.780 |             0.747 |            -0.001 |
| Weekday profile x MA(28) |              0.744 |             0.789 |             0.732 |            -0.025 |
| SBA                      |              0.756 |             0.781 |             0.777 |            -0.014 |
| Croston (classic)        |              0.760 |             0.783 |             0.790 |             0.039 |
| Croston (optimised)      |              0.766 |             0.784 |             0.803 |             0.063 |
| Seasonal naive (7)       |              0.963 |             1.029 |             0.864 |            -0.055 |

## By demand pattern (mean RMSSE over all 4 windows)

| model                    |   erratic |   intermittent |   lumpy |   smooth |
|:-------------------------|----------:|---------------:|--------:|---------:|
| ADIDA                    |     0.706 |          0.743 |   0.718 |    0.855 |
| Croston (classic)        |     0.675 |          0.764 |   0.729 |    0.854 |
| Croston (optimised)      |     0.698 |          0.767 |   0.735 |    0.860 |
| IMAPA                    |     0.706 |          0.744 |   0.723 |    0.855 |
| SBA                      |     0.673 |          0.760 |   0.725 |    0.851 |
| SES (alpha per series)   |     0.696 |          0.743 |   0.726 |    0.853 |
| Seasonal naive (7)       |     0.902 |          0.977 |   0.940 |    1.067 |
| TSB                      |     0.673 |          0.743 |   0.713 |    0.850 |
| Weekday profile x MA(28) |     0.672 |          0.751 |   0.719 |    0.848 |

Syntetos-Boylan class from the last year before the holdout (intermittent: 367, lumpy: 67, smooth: 52, erratic: 14
series). On **intermittent** series the best RMSSE (0.743) is shared by ADIDA, IMAPA, SES (alpha per series), TSB - within 0.002 of
each other; the weekday-profile baseline is at 0.751.

* Croston-type methods separate *how often* an item sells from *how much* it sells when it does, and forecast a flat
  rate ("0.4 units/day"): the right expected value, but no weekday shape. Phase 5 showed what that costs once
  forecasts are added up the hierarchy (Croston WRMSSE vs the weekday profile).
* **Croston's bias is visible**: classic +3.9% and optimised
  +6.3% over-forecast. Dividing size by interval is biased upwards;
  **SBA** multiplies by (1 - alpha/2) to correct it (-1.4%).
* **TSB** updates the *probability* of a sale every day, so it decays towards zero when an item stops selling
  (obsolescence); Croston only updates when a sale happens and keeps forecasting the old rate forever.
* **ADIDA / IMAPA** aggregate the series into buckets (where it is no longer intermittent), forecast, and
  disaggregate - temporal aggregation, the same idea as forecasting weekly for purchasing.

**Lesson:** specialised intermittent methods are a sensible per-series baseline for slow movers, but they do not
model drivers (price, SNAP, events) or weekly shape. Global ML models with a Tweedie/Poisson loss (Phase 11)
handle intermittency *and* drivers in one model.


