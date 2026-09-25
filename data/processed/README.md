# Processed data

This directory is for generated caches and temporary indexes. Its contents are ignored and must be reproducible from the source dataset, committed configuration, and split manifests.

Legacy `split_indices.json` and class-mapping caches are not used by the training loader.
The authoritative partitions and class mapping are tracked under `data/splits/`.
