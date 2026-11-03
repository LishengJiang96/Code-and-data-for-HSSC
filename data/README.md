# Data dictionary

`panel_data.xlsx` has two sheets, one for each subsystem. Both sheets cover the
same 65 cities and the same 14 years (2010-2023), so each sheet has 910 rows.
There are no missing values.

- `City` uses the label `Province - City`, for example `Jiangsu - Nanjing`.
- `Year` is an integer from 2010 to 2023.
- All other columns are numeric indicator values.

The code standardises each indicator by min and max before weighting, so the
units below do not affect the results.

## Sheet `Ecological security`

| Indicator | Dimension | Attribute | Unit |
| --- | --- | --- | --- |
| Population density | Pressure | cost | persons per km² |
| Urbanization rate | Pressure | cost | fraction |
| Proportion of agricultural water consumption | Pressure | cost | fraction |
| GDP growth rate | Pressure | cost | percent |
| Arable land per capita | State | benefit | ha per person |
| Per capita park green space | State | benefit | m² per person |
| Green space coverage in built-up areas | State | benefit | percent |
| Fiscal expenditure on agriculture and forestry (100 million CNY) | Response | benefit | 10^8 CNY |
| Effective irrigated area ratio | Response | benefit | fraction |
| Environmental protection investment ratio | Response | benefit | fraction |

## Sheet `Food security`

| Indicator | Dimension | Attribute | Unit |
| --- | --- | --- | --- |
| Grain output per unit area | Quantitative | benefit | t per ha |
| Multiple cropping index | Quantitative | benefit | ratio |
| Fluctuation coefficient of grain production | Quantitative | cost | fraction |
| Fertilizer use intensity | Qualitative | cost | t per ha |
| Agricultural film use intensity | Qualitative | cost | t per ha |
| Pesticide use intensity | Qualitative | cost | t per ha |
| Gross power of agricultural machinery | Qualitative | benefit | 10^4 kW |
| Natural population growth rate | Consumption | cost | per mille |
| Grain possession per capita | Consumption | benefit | t per person |
| Grain self-sufficiency rate | Consumption | benefit | ratio |
| Water resource consumption per unit grain | Resource | cost | m³ per t |
| Arable land resource consumption per unit grain | Resource | cost | ha per t |
| Agricultural labor productivity | Resource | benefit | 10^4 CNY per person |
| Agricultural land productivity | Resource | benefit | as collected |

## Attributes

A `cost` indicator is multiplied by -1 before the outranking step, so a higher
value always means a better state. A `benefit` indicator is used as it is.
