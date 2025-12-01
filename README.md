# GeoTIFF Analyser

A Python tool to analyze and compare soil texture (Clay vs. Sand) between the Nile Delta and the Western Desert using SoilGrids GeoTIFF data.

## Requirements
*   Python 3.x
*   \asterio\ 
*   \
umpy\ 
*   \matplotlib\ 

## Usage
1.  Place your \.tif\ files in the project directory.
2.  Run the analysis script:
    \\\ash
    python analysis.py
    \\\`n
## Output
*   **Console**: Prints mean values for defined regions.
*   **Files**:
    *   \soil_analysis_summary.csv\: Summary of mean values.
    *   \soil_composition_comparison.png\: Bar chart comparing regions.
    *   \combined_soil_maps.png\: Visual maps of the analyzed areas.