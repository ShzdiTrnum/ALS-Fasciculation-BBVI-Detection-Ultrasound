# data/

Put the ultrasound video here (git-ignored). The initial-angle CSV is included.

| File | Description |
|------|-------------|
| `ALS - Trim20s.mp4` | ultrasound clip (980 × 748, grayscale or BGR). Pass another path with `--video`. |
| `Original_Groundtruth_Values.csv` | included: optical-flow values per ring for frames 6–24 (only the first row is used). Not ground truth. |

## Initial-angle CSV

Only the **first data row** is read. Its columns 2 … 21 must hold one value per ring, in the order of
`--radii` (5, 10, …, 100 px). The estimator uses them directly as angles in **radians**:

```
Frame,R5,R10,...,R100
6,<rad>,<rad>,...,<rad>
```

The column names are not used. The values come from the optical-flow / Green's-theorem baseline in the
[Optical-Flow](https://github.com/ShzdiTrnum/Optical-Flow) repository.

## Data availability

The clinical videos were recorded at Tokyo Metropolitan Neurological Hospital and are not
redistributed. To try the code without them, run it on the synthetic video from
[Fasciculation-Simulation](https://github.com/ShzdiTrnum/Fasciculation-Simulation) with a matching
`--roi`, `--radii` and `--init-csv`.
