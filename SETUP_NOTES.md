# Regenerating the shipped files

This note documents how `data/sample_patch.mat`, `pretrained/` and
`expected_outputs/metrics_reference.json` were produced from the author's
local archives. **A normal user does not need any of this**: the shipped
files already reproduce every number in the README. It is kept so that the
extraction is repeatable, and so that changing the shipped patch is a
mechanical operation rather than a hunt through hard-coded constants.

## Prerequisites

- MATLAB R2024a or later
- The source tree, not part of this repository:

  | What | Where |
  |---|---|
  | Patch HDF5 files | `D:\patch_36km_EASE\Unified_Patches_QC_2022_TO\patches_unified_h5_64x64\patch_<NNNN>.h5` |
  | Preprocessing metadata | `…\patches_unified_h5_64x64\unified_metadata.mat` |
  | EASE M36 grid | `inputs\grid_M36.mat` |
  | WBM global result | `outputs\WBM\Open-Loop_TO\WBM_Global_Results_QC.mat` |
  | Trained models | `outputs\MDN-ConvLSTM\outputs_individual_patches_QC_64x64_open_loop_TO_V1\Patch_<NNNN>\` |
  | In-situ representatives | `outputs\ISMN_era5_R30\timeseries\representatives_2024.csv` |

  Absolute paths are pre-filled in the `USER PATHS` block at the top of each
  script; edit them to match your machine.

## Steps

```matlab
>> cd PGMN/matlab

% (1) ~30 s  -> ../data/sample_patch.mat
>> extract_sample_patch

% (2) ~5 s   -> ../pretrained/wbm_params_patch<N>.mat
>> extract_wbm_params

% (3) ~10 s  -> ../pretrained/best_model_M<M>.mat
>> trimBestModel          % preview
>> trimBestModel(true)    % write

% (4) sanity check: do the three files agree with each other?
>> verify_loaded_files

% (5) ~1 min -> ../expected_outputs/metrics_reference.json
>> make_reference         % preview
>> make_reference(true)   % write

% (6) confirm the package reproduces its own reference
>> runExample_quick
>> cd ../expected_outputs
>> verify_reproduction
```

**Run them in that order.** Steps 2, 3 and 5 read the patch index and extent
out of `sample_patch.mat` rather than repeating them, so step 1 is the single
place a patch is chosen.

## Changing the shipped patch

Edit one line, `PATCH_IDX` near the top of `extract_sample_patch.m`, then
re-run the whole sequence above. Everything else follows:

- the WBM parameter file is named and cut from `sample_patch.mat`'s extent;
- the model file is found by globbing `Patch_<NNNN>/models/best_model_M*.mat`,
  so a patch with a different mixture size just produces a differently named
  file, and the run-time scripts glob for it too;
- `make_reference` re-derives every reference value and the scored-cell count;
- `plotReproduction` picks up the in-situ reference cell if the new patch
  contains one, and falls back to the domain median if it does not.

There is no version control on the source tree, so `trimBestModel` and
`make_reference` never overwrite silently: an existing destination is renamed
to `<name>.bak_<timestamp>` first. Delete those backups once you are satisfied.

## Expected output

| File | Size | Contents |
|---|---|---|
| `data/sample_patch.mat` | ~14 MB | `XVal` int32 [64 64 3 1004], `ResAvailVal` uint8, `YVal_Direct` int16, `MaskVal` logical, `scale_factor`, `fill_value`, `LAT`, `LON`, `land_mask`, `time`, `split`, `station`, `meta` |
| `pretrained/best_model_M2.mat` | ~0.7 MB | `M`, `input_size`, `normParams`, `patch_info`, `save_method`, learnables and state tables plus values, trimmed `performance`, `train_info` |
| `pretrained/wbm_params_patch20.mat` | ~40 KB | `alpha`, `Z`, `beta`, `modeling_mask`, per-cell WBM skill, `meta` |
| `expected_outputs/metrics_reference.json` | ~3.6 KB | reference medians, IQRs, domain counts, tolerances |

## Two things worth knowing

**The evaluation block is shipped quantized.** `XVal` and `YVal_Direct` are
scaled integers, exactly as the preprocessing wrote them, and
`mdn_reconstruct.m` dequantizes at run time. Reconstructing physical arrays
from the source archives instead would re-introduce that rounding differently
and the reproduction would drift in the fourth decimal for no benefit.

**The trained model is shipped as weights, not as a network.** The custom
`ConvLSTMLayer` does not survive being saved inside a `dlnetwork`: MATLAB warns,
substitutes a default layer, and hands back a network that runs and predicts
nonsense. `trimBestModel` therefore drops the `net` field and keeps the
Learnables and State tables; `mdn_load_model` rebuilds the architecture with
`mdn_build_network` and matches the stored values back by (Layer, Parameter)
name. `verify_loaded_files` checks that this round trip actually works before
you trust a run.

## Where the reference values come from

`make_reference` does **not** read the package's own output; a target the
package generated would pass by construction. Accuracy is taken from
`performance.*_per_grid` inside the production model file, written by the
evaluation code; ubRMSE, which the paper reports, is derived per cell from
the stored RMSE and Bias (ubRMSE² = RMSE² − Bias²). Calibration is recomputed from that patch's
`predictions.mat` with `uncertainty_metrics.m`, rather than read off the
published global maps: 53 % of land cells belong to two or more patches, so
inside any one patch's box those maps are blends of that patch and its
neighbors, correct for a global map but wrong as a single-patch target.
