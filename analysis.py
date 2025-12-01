import sys
import csv
from pathlib import Path

import rasterio
from rasterio.windows import from_bounds
import numpy as np
import matplotlib.pyplot as plt


def get_box_stats(dataset, bbox, nodata_threshold=-1e4):
    """Extract mean value inside a lat/lon bounding box.

    bbox: (min_lon, min_lat, max_lon, max_lat)
    nodata_threshold: values below this are treated as nodata.
    """

    min_lon, min_lat, max_lon, max_lat = bbox

    # Build a window in pixel coordinates from geographic bounds
    window = from_bounds(min_lon, min_lat, max_lon, max_lat, transform=dataset.transform)

    data = dataset.read(1, window=window, masked=True)

    # Mask very negative values (SoilGrids nodata is often -32768)
    data = np.ma.masked_less(data, nodata_threshold)

    if data.count() == 0:
        return np.nan

    return float(data.mean())


def plot_raster_map(dataset, title, cmap="viridis", vmin=None, vmax=None):
    data = dataset.read(1, masked=True)

    # Mask very negative nodata values
    data = np.ma.masked_less(data, -1e4)

    fig, ax = plt.subplots(figsize=(6, 4))
    img = ax.imshow(
        data,
        cmap=cmap,
        vmin=vmin,
        vmax=vmax,
        extent=(
            dataset.bounds.left,
            dataset.bounds.right,
            dataset.bounds.bottom,
            dataset.bounds.top,
        ),
        origin="upper",
    )
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(title)
    cbar = plt.colorbar(img, ax=ax)
    cbar.set_label("Content (scaled units)")
    plt.tight_layout()


def main():
    # Resolve file paths relative to this script's folder
    base_dir = Path(__file__).resolve().parent

    # Define locations to analyze
    # Grouped by region for averaging later
    # Note: Coordinates are approximate centers for the analysis box
    locations = [
        # Nile Delta Group
        {
            "group": "Nile Delta",
            "name": "Nile Delta 1",
            "clay_file": "ClayContentNile1.tif",
            "sand_file": "SandContentNile1.tif",
            "lat": 31.0, "lon": 31.0
        },
        {
            "group": "Nile Delta",
            "name": "Nile Delta 2",
            "clay_file": "ClayContentNile2.tif",
            "sand_file": "SandContentNile2.tif",
            "lat": 30.5, "lon": 31.2  # Adjusted coordinate for second Nile sample
        },
        # Western Desert Group
        {
            "group": "Western Desert",
            "name": "Western Desert 1",
            "clay_file": "ClayContentDesert1.tif",
            "sand_file": "SandContentDesert1.tif",
            "lat": 30.8, "lon": 30.2
        },
        {
            "group": "Western Desert",
            "name": "Western Desert 2",
            "clay_file": "ClayContentDesert2.tif",
            "sand_file": "SandContentDesert2.tif",
            "lat": 29.8462, "lon": 31.6474  # Adjusted coordinate for second Desert sample
        }
    ]

    box_half_size = 0.05
    scale = 10.0  # SoilGrids scale factor

    # Store results for aggregation
    group_stats = {}  # { "Nile Delta": {"clay": [], "sand": []}, ... }
    csv_rows = []     # List to store data for CSV export
    map_data = []     # Store map data for combined plotting: (name, clay_data, sand_data, extent)

    print("--- Processing Soil Data ---")

    for loc in locations:
        clay_path = base_dir / loc["clay_file"]
        sand_path = base_dir / loc["sand_file"]

        if not clay_path.exists() or not sand_path.exists():
            print(f"Skipping {loc['name']}: Files not found ({loc['clay_file']}, {loc['sand_file']})")
            continue

        # Define bounding box
        center_lat, center_lon = loc["lat"], loc["lon"]
        bbox = (
            center_lon - box_half_size,
            center_lat - box_half_size,
            center_lon + box_half_size,
            center_lat + box_half_size,
        )

        try:
            with rasterio.open(clay_path) as clay_ds, rasterio.open(sand_path) as sand_ds:
                clay_mean = get_box_stats(clay_ds, bbox)
                sand_mean = get_box_stats(sand_ds, bbox)
                
                # Apply scale
                clay_mean /= scale
                sand_mean /= scale

                # Store for group averaging
                if loc["group"] not in group_stats:
                    group_stats[loc["group"]] = {"clay": [], "sand": []}
                
                if not np.isnan(clay_mean):
                    group_stats[loc["group"]]["clay"].append(clay_mean)
                if not np.isnan(sand_mean):
                    group_stats[loc["group"]]["sand"].append(sand_mean)

                # Add to CSV rows
                csv_rows.append({
                    "Group": loc["group"],
                    "Location": loc["name"],
                    "Latitude": center_lat,
                    "Longitude": center_lon,
                    "Mean Clay (%)": f"{clay_mean:.2f}" if not np.isnan(clay_mean) else "NaN",
                    "Mean Sand (%)": f"{sand_mean:.2f}" if not np.isnan(sand_mean) else "NaN"
                })

                # Store data for combined map plot if valid
                if not np.isnan(clay_mean) and not np.isnan(sand_mean):
                    # Read data for plotting (re-reading window to ensure we have the array)
                    window = from_bounds(*bbox, transform=clay_ds.transform)
                    c_data = clay_ds.read(1, window=window, masked=True)
                    s_data = sand_ds.read(1, window=window, masked=True)
                    
                    # Mask nodata
                    c_data = np.ma.masked_less(c_data, -1e4)
                    s_data = np.ma.masked_less(s_data, -1e4)

                    # Apply scale factor to map data so it shows as %
                    c_data = c_data / scale
                    s_data = s_data / scale
                    
                    extent = (bbox[0], bbox[2], bbox[1], bbox[3])
                    map_data.append({
                        "name": loc["name"],
                        "clay": c_data,
                        "sand": s_data,
                        "extent": extent
                    })

                print(f"Location: {loc['name']}")
                print(f"  Coordinates: {center_lat}N, {center_lon}E")
                print(f"  Files: {loc['clay_file']}, {loc['sand_file']}")
                if np.isnan(clay_mean) or np.isnan(sand_mean):
                    print("  Result: No valid data inside box.")
                else:
                    print(f"  Mean Clay: {clay_mean:.2f}%")
                    print(f"  Mean Sand: {sand_mean:.2f}%")
                print("-" * 30)

        except Exception as e:
            print(f"Error processing {loc['name']}: {e}")

    if not group_stats:
        print("No data processed. Exiting.")
        return

    # --- Export CSV ---
    csv_path = base_dir / "soil_analysis_summary.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["Group", "Location", "Latitude", "Longitude", "Mean Clay (%)", "Mean Sand (%)"])
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"\nSummary CSV saved to: {csv_path}")

    # --- Combined Grid Plot (Compact Map View) ---
    if map_data:
        n_locs = len(map_data)
        fig, axes = plt.subplots(n_locs, 2, figsize=(10, 4 * n_locs), constrained_layout=True)
        
        if n_locs == 1: axes = [axes] # Handle single row case

        for i, data in enumerate(map_data):
            # Clay Map
            ax_clay = axes[i][0] if n_locs > 1 else axes[0]
            im_c = ax_clay.imshow(data["clay"], cmap="Oranges", extent=data["extent"], origin="upper")
            ax_clay.set_title(f"{data['name']} - Clay")
            plt.colorbar(im_c, ax=ax_clay, fraction=0.046, pad=0.04)
            
            # Sand Map
            ax_sand = axes[i][1] if n_locs > 1 else axes[1]
            im_s = ax_sand.imshow(data["sand"], cmap="YlOrBr", extent=data["extent"], origin="upper")
            ax_sand.set_title(f"{data['name']} - Sand")
            plt.colorbar(im_s, ax=ax_sand, fraction=0.046, pad=0.04)

        grid_plot_path = base_dir / "combined_soil_maps.png"
        plt.savefig(grid_plot_path, dpi=150)
        print(f"Combined map grid saved to: {grid_plot_path}")
        # plt.show() # Optional: comment out if running in batch

    # --- Calculate Averages per Group ---
    print("\n--- Group Averages ---")
    averaged_results = {}
    
    for group, data in group_stats.items():
        avg_clay = np.mean(data["clay"]) if data["clay"] else np.nan
        avg_sand = np.mean(data["sand"]) if data["sand"] else np.nan
        
        averaged_results[group] = {"clay": avg_clay, "sand": avg_sand}
        
        print(f"{group}:")
        print(f"  Average Clay: {avg_clay:.2f}%")
        print(f"  Average Sand: {avg_sand:.2f}%")

    # --- Combined bar chart: Group Averages ---
    labels = ["Clay", "Sand"]
    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 6))
    
    groups = list(averaged_results.keys())
    # Ensure consistent colors if we have exactly these two groups
    colors = ["tab:blue", "tab:orange"] if len(groups) == 2 else None
    
    for i, group in enumerate(groups):
        res = averaged_results[group]
        vals = [res["clay"], res["sand"]]
        
        # Handle NaNs for plotting
        vals = [0.0 if np.isnan(v) else v for v in vals]
        
        offset = (i - (len(groups) - 1) / 2) * width
        color = colors[i] if colors else None
        
        ax.bar(x + offset, vals, width, label=group, color=color)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean content (%)")
    ax.set_title("Soil Composition: Nile Delta vs Western Desert (Averaged)")
    ax.legend()
    plt.tight_layout()

    output_plot = base_dir / "soil_composition_comparison.png"
    plt.savefig(output_plot)
    print(f"\nComparison plot saved to: {output_plot}")
    plt.show()


if __name__ == "__main__":
    main()

