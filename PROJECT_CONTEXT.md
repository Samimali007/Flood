## ADDENDUM: Flood Event Reconstruction & Replay
- The product shows OBSERVED flood extent at satellite acquisition dates, plus DEM-derived likely drainage
  pathways. It is not flood prediction. Between acquisitions nothing is observed; never animate smooth
  "water movement" as if it were observed.
- Sensor-agnostic design: Sentinel-1 and NISAR L-band behind one interface. Always record the sensor, product
  level, relative orbit, polarization and acquisition datetime per scene.
- Flood masks come from SAR backscatter (dB). Known limitations to state in the UI: flooded vegetation, wind-roughened
  water, radar shadow/layover, urban areas, threshold sensitivity.
- Depth is approximate and optional, labelled "rough estimate".
- Alerts are opt-in advisories or post-event "detected" reports, never official warnings.