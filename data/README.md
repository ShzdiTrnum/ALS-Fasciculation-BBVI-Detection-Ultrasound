# data/

Place your input data files here before running the script.

## Required Files

| File | Description |
|------|-------------|
| `your_video.mp4` | Input ultrasound or synthetic video file |
| `Original_Groundtruth_Values.csv` | Initial rotation angle values per ring per frame |

## CSV Format

The `Original_Groundtruth_Values.csv` file must follow this structure:

```
Frame, Rectangle_0, Rectangle_1, ..., Rectangle_19
0,     0.0,         0.0,         ..., 0.0
1,     <angle>,     <angle>,     ..., <angle>
...
```

- **Frame**: integer frame index (0-based)
- **Rectangle_0 … Rectangle_19**: rotation angle in radians for each of the 20 rings

This file is produced by the [Fasciculation-Simulation](https://github.com/ShzdiTrnum/Fasciculation-Simulation) repository.

> **Note:** Video and CSV files are excluded from version control via `.gitignore` due to file size.
> Store them locally in this folder and update the paths in the script accordingly.
