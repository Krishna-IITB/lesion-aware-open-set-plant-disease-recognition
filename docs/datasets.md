# Datasets and taxonomy

No dataset is committed. Confirm the current terms at the source before download or redistribution.

## Recommended roles

| Dataset | Role | Access and citation notes |
|---|---|---|
| PlantVillage | controlled/lab source | 54k leaf images; use the [maintainer repository](https://github.com/spMohanty/PlantVillage-Dataset) and cite Hughes & Salathé (2015) plus Mohanty et al. (2016) |
| PlantWild | primary in-the-wild target | multimodal disease images/descriptions from the [MVPDR authors](https://github.com/tqwei05/MVPDR); cite Wei et al. (ACM MM 2024) |
| PlantDoc | optional smaller field target | use the [authors' repository](https://github.com/pratikkayal/PlantDoc-Dataset) and cite Singh et al. (2020) |
| PlantSeg | lesion supervision | pixel-level in-the-wild masks; use [Zenodo DOI 10.5281/zenodo.17719108](https://doi.org/10.5281/zenodo.17719108) and cite Wei et al. (Scientific Data, 2026) |

The strongest default classification story is PlantVillage → PlantWild because PlantWild supplies
disease descriptions aligned with the MVPDR question. PlantDoc is a useful fallback only for the
taxonomically compatible subset. PlantSeg is a separate segmentation source; it is not silently
treated as the same classification taxonomy.

## Manifest schema

Each JSONL row contains `sample_id`, absolute or manifest-relative `image_path`, `label`, one of
`train|validation|test`, `domain`, and optional `mask_path`. Dataset preparation never copies images.
`validate-data` opens all images/masks and hashes image bytes to detect duplicates crossing splits.

## Taxonomy protocol

Build a reviewed source-label → canonical-label JSON mapping. A target of `null` means excluded;
missing keys fail rather than map implicitly. Record the mapping version with every run. Do not
merge diseases merely because crop names look similar. Healthy classes should be treated explicitly,
not as generic unknown disease.

## PlantSeg masks

Convert annotations to one binary lesion mask per image without antialiased label interpolation.
Images without usable lesion regions may remain in classification manifests but are excluded from
supervised decoder batches. Preserve an original-data checksum and document any polygon/raster
conversion. Split by the released protocol where available and never use test masks as inputs.
