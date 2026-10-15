# iDriving pipeline

Project layout:

```text
run_pipeline.py      Pipeline entry point
pipeline_config.py   Shared configuration and path resolution
config/              Pipeline YAML settings
kafka/               Consumer and producer
kafka/templates/     Incoming example and outgoing message template
minIO/               MinIO downloader
preprocess/          Frame extraction and reconstruction
train/               Splatfacto training
export/              Model export and SPLAT conversion
viewer/              HTTPS viewer, certificates, and scene links
dataset/             One folder per notification record; video and generated outputs
tests/               Offline checks
```

Built-in script, configuration, and template paths are resolved relative to
the project files, independently of the terminal's working directory. Paths
you explicitly pass on the command line are relative to your current directory.

The complete pipeline reads its defaults from [`config/config.yaml`](config/config.yaml).
Set `paths.dataset_dir` once and leave the other output paths as `null` to get:

```text
<dataset_dir>/
├── images/
├── colmap/
├── transforms.json
├── sparse_pc.ply
├── splatfacto_output/
└── export/
    ├── splat.ply
    └── splat.splat
```

Run the pipeline from the repository root:

```bash
python iDriving_pipeline/run_pipeline.py
```

Use another configuration file with `--config`. Every path can also be
overridden for one run; command-line values take precedence over YAML values:

```bash
python iDriving_pipeline/run_pipeline.py \
  --dataset_dir /data/road_0047 \
  --video_path /incoming/REC_0047.mp4 \
  --splat_output_path /published/road_0047.splat
```

To inspect the resolved paths without starting any processing:

```bash
python iDriving_pipeline/run_pipeline.py --dry_run
```

## Download MinIO uploads from Kafka

Install the consumer dependencies in the Python environment used to run it:

```bash
python -m pip install aiokafka minio rich
python iDriving_pipeline/kafka/consumer.py --topic YOUR_ACCELI_UPLOAD_TOPIC
```

Replace `YOUR_ACCELI_UPLOAD_TOPIC` with the incoming topic agreed with ACCELI.
The default broker is `157.90.19.188:9092`; override it with `--broker`.
The MinIO bucket is read from each record's `body.bucket`. If absent, `--bucket`
provides the fallback, defaulting to `uc2.1-certh-3d-smart`, using the connection
settings in `minIO/download_minio.py`.

Downloads display a progress bar with percentage, transferred/total bytes,
transfer speed, and estimated time remaining. The bar starts once MinIO responds
with the file data; connection checks are logged beforehand. This uses Rich,
which is already a Nerfstudio dependency. The same display appears when running
`minIO/download_minio.py` directly.

For a standalone download, run from the repository root:

```bash
python iDriving_pipeline/minIO/download_minio.py uc2.1-certh-3d-smart REC_0057.mp4
```

Standalone downloads go to `Downloaded/` under the Nerfstudio repository.
The consumer supplies its own per-dataset destination under `dataset/`.

The downloader uses a 10-second connection timeout and a 30-second socket-read
timeout, with automatic HTTP retries disabled, so network stalls fail visibly.
These are network-wait limits, not a limit on the total download duration or
file size. A failure leaves an incomplete temporary file; retrying currently
downloads the object from the beginning.

The assumed message format is in
[`kafka/templates/file_upload_ready.json`](kafka/templates/file_upload_ready.json).
The consumer checks `records[].header.producer-id == "ACCELI"` and
`records[].body.file_upload_status == "ready"`. The assumed filename field is
`records[].body.file_name`: it must contain the exact MinIO object key,
including any prefix (for example, `videos/REC_0057.mp4`). Adjust `ready_files()`
if the agreed partner schema differs. Send the JSON envelope directly as the
Kafka value; do not wrap it inside another `body`.

The consumer only reads notifications and downloads files. It does not launch
`run_pipeline.py`, reconstruction, training, the viewer, or outgoing messages.
For each matching record it saves the file directly under:

```text
iDriving_pipeline/dataset/<topic>-<partition>-<offset>-<index>/<filename>
```

Use `--work_dir /path/to/datasets` to choose another parent directory. The
consumer accepts any file type, including small JSON files for download tests.
The old `--download_only` and `--config` consumer options have been removed.
Existing downloads are not moved, and committed Kafka offsets are unchanged.

### Run reconstruction separately

After the video has downloaded, run these commands from inside `iDriving_pipeline`:

```bash
dataset_dir="dataset/idriving_certh_3d_smart_uc2.1-0-13-0"
python run_pipeline.py \
  --video_path "$dataset_dir/REC_0057.mp4" \
  --dataset_dir "$dataset_dir" \
  --extract_images
```

Use the dataset directory printed by the consumer and the actual video name.
The consumer creates the dataset directory and downloads the video; the pipeline
creates the processing subfolders when you run it:

```text
dataset/
└── idriving_certh_3d_smart_uc2.1-0-13-0/
    ├── REC_0057.mp4
    ├── images/
    ├── colmap/
    ├── transforms.json
    ├── sparse_pc.ply
    ├── splatfacto_output_hloc_matched5/
    └── export/
        ├── splat.ply
        └── splat.splat
```

Keep YAML output overrides null or relative so generated outputs stay in that
same dataset directory. Pass `--config` to `run_pipeline.py` for a different
configuration. Pipeline settings still control the outgoing ready notification
and viewer startup; add `--no_start_viewer` to leave the viewer running separately.

Run one consumer instance. Downloads are sequential. Kafka progress is committed
only after all downloads for a message succeed; unrelated messages are ignored
and committed. Validation or download failures stop the consumer without committing
the failed message. Restart with the same `--group_id` to retry. A crash before
commit can repeat downloads, including completed records in a multi-record message.
A later manual pipeline failure does not affect the consumer's committed offsets;
rerun the pipeline directly on the downloaded video.

A new group starts at the earliest retained message; subsequent starts resume
from committed progress. The default `--max_poll_interval_ms 86400000` allows
24 hours to download all files in one Kafka message. Downloads run in a worker
thread so the event loop remains free for Kafka heartbeats.

The consumer has no manual offset-skip option. A failed upload notification
stops processing and is retried when the same consumer group restarts. A later
corrected notification does not replace the earlier message. Previously
committed progress is preserved; removing the old skip option does not rewind
the consumer group. Messages unrelated to ACCELI ready uploads are still
ignored and committed, including this tool's outgoing reconstruction messages.

### Dummy upload notification from another VPN machine

The existing `kafka/producer.py` can send the ACCELI message
without code changes. It reads the producer ID and body from the JSON file;
its `CLIENT_ID = "CERTH"` only identifies the Kafka client connection and does
not override the JSON producer ID. It sends to broker `157.90.19.188:9092`,
topic `idriving_certh_3d_smart_uc2.1`, so the test consumer must use that topic.

1. Upload a small test file (for example, `small.json`) to MinIO bucket
   `uc2.1-certh-3d-smart` and wait for the
   upload to finish. Set `body.file_name` in `file_upload_ready.json` to its
   exact object key. The producer sends a notification; it does not upload a file.
   The consumer accepts any file type; use a video when running reconstruction.
2. On the receiving machine, from the repository root, run:

   ```bash
   python iDriving_pipeline/kafka/consumer.py \
     --topic idriving_certh_3d_smart_uc2.1 \
     --group_id certh-3d-smart-download-test
   ```

3. On the sending VPN machine, install `aiokafka` and send:

   ```bash
   python -m pip install aiokafka
   python iDriving_pipeline/kafka/producer.py \
     --message iDriving_pipeline/kafka/templates/file_upload_ready.json
   ```

   If that machine has no repository checkout, copy just `producer.py` and
   the edited `file_upload_ready.json` there, then run
   `python producer.py --message file_upload_ready.json`.

Both machines need network access to the Kafka broker; the receiver also needs
MinIO access. The receiver prints `Download complete:` followed by the downloaded
file path and dataset directory, commits the message, and keeps listening.
The separate test group preserves the normal download group's progress, but a
new group reads retained messages too. Other consumer groups on the same topic
also receive the test notification.

To test reconstruction after downloading a video, run `run_pipeline.py` as shown
above. No new Kafka notification is needed to process an already downloaded video.

## Train from an existing COLMAP/VGGT reconstruction

Use [`preprocess/convert_colmap_to_nerfstudio.py`](preprocess/convert_colmap_to_nerfstudio.py)
to generate `transforms.json` and `sparse_pc.ply` without rerunning hloc,
COLMAP reconstruction, or frame extraction. From inside `iDriving_pipeline`:

```bash
python preprocess/convert_colmap_to_nerfstudio.py \
  --recon_dir /data/vggt/colmap/sparse/0 \
  --image_dir /data/vggt/images \
  --output_dir /data/vggt/nerfstudio_dataset

python train/train_splatfacto.py \
  --dataset_dir /data/vggt/nerfstudio_dataset \
  --output_dir /data/vggt/nerfstudio_dataset/splatfacto_output \
  --downscale_factor 1
```

The model folder must contain `cameras.bin`, `images.bin`, and `points3D.bin`,
or the corresponding three `.txt` files. The utility supports per-image
intrinsics and records paths to the supplied images without copying them.
Use the same images (including any resizing/cropping) that VGGT used; their
dimensions must match the reconstructed cameras. Camera poses and points
are converted together using Nerfstudio's coordinate conventions.

If no paths are supplied, the utility uses the existing YAML defaults:
`colmap_dir` for the reconstruction, `image_dir` for images, and
`nerfstudio_dir` for outputs. Existing output files are preserved unless
`--overwrite` is supplied. `--dry_run` checks the model location without
creating files.

## Export and view an already trained model

The complete pipeline checks registration before starting training. Defaults in
`config/config.yaml` require at least `reconstruction.min_registered_images: 10`
and `reconstruction.min_registered_fraction: 0.5` (50% of the input images).
Both checks must pass. These are adjustable heuristics, not a guarantee of good
geometry. CLI overrides are `--min_registered_images` and
`--min_registered_fraction`. A failed check stops before training or notification.

The iDriving trainer saves directly inside `splatfacto_output_dir`:

```text
dataset/<dataset-name>/splatfacto_output_hloc_matched5/
├── config.yml
└── nerfstudio_models/
    └── step-000029999.ckpt
```

There are no additional experiment, method, or timestamp directories. The saved
config records `output_dir_is_run_dir: true`, so export finds checkpoints in the
same directory layout. Older configs without that setting keep their original
nested paths; existing model files are not moved.

Once training returns, the wrapper verifies that its final checkpoint exists.
If `training_config_path` points elsewhere, it also copies the config there.
Intermediate checkpoints from interrupted training do not count as completion.
Starting a fresh run in a directory that already contains checkpoints is rejected
to avoid mixing models. Choose a new `--splatfacto_output_dir` for a new run, or
move the previous output directory first.

The Kafka ready message is sent only after training, PLY export, SPLAT conversion,
and viewer-link publication succeed. Saving a training configuration or opening
the training viewer does not mean training has finished. An interrupted run does
not produce a ready notification.

After Splatfacto training, [`export/export_splat.py`](export/export_splat.py) performs the
remaining work in one command:

1. reads `training_config_path` (by default
   `<dataset_dir>/splatfacto_output/config.yml`);
2. runs the Nerfstudio Gaussian Splat export to create `splat.ply`;
3. runs `export/convert_ply_to_splat.py` to create `splat.splat`;
4. starts the HTTPS viewer when `viewer.start_server` is `true`.

From inside `iDriving_pipeline`, run:

```bash
python export/export_splat.py
```

The same utility is called automatically by `run_pipeline.py` immediately
after training. Generated PLY and SPLAT files remain under `dataset_dir`.
After conversion, the utility updates
`viewer/splat_files/current.splat`, a small symbolic link pointing to
the configured `splat_output_path`. The viewer loads that stable static path,
so the large SPLAT file is not duplicated.

`index.html` always loads `splat_files/current.splat`. Each successful pipeline
export updates this link to the new dataset's exported SPLAT, including exports
run with `--no_start_viewer`. Refresh an already open viewer page after exporting
to load the new scene. Keep the source SPLAT in place, since the link refers to
it rather than making a copy. The standalone `export/convert_ply_to_splat.py`
utility only converts the file; automatic viewer publication is performed by
`export/export_splat.py` and the full pipeline.

To export without starting the viewer:

```bash
python export/export_splat.py --no_start_viewer
```

To inspect the paths and export command without doing any work:

```bash
python export/export_splat.py --dry_run
```

### Crop unwanted Gaussians during export

Copy the oriented bounding-box values generated by the Nerfstudio viewer into
the `export` section of `config/config.yaml`:

```yaml
export:
  obb_center: [0.0, 0.0, 0.0]
  obb_rotation: [0.0, 0.0, 0.0]
  obb_scale: [1.0, 5.0, 1.0]
```

Nerfstudio requires all three values to apply the crop. Set all three to
`null` to export every Gaussian. Values can also be overridden for one run:

```bash
python export/export_splat.py \
  --obb_center 0 0 0 \
  --obb_rotation 0 0 0 \
  --obb_scale 1 5 1
```

Relative dataset output paths in YAML, such as `image_dir`, `colmap_dir`, and
`splatfacto_output_dir`, are relative to `dataset_dir`. Relative input paths,
such as `video_path`, are relative to the YAML file. Relative command-line
paths are relative to the directory where the command is run. The individual
stage scripts keep accepting their explicit path arguments and can still be
used independently.

## How `pipeline_config.py` works

[`pipeline_config.py`](pipeline_config.py) is the shared path resolver used by
the orchestration script:

1. `load_pipeline_config()` reads the YAML safely and verifies that its top
   level is a mapping.
2. `get_config_section()` reads sections such as `paths`, `training`, and
   `reconstruction`, while rejecting invalid section types.
3. `PipelinePaths.from_config()` applies this path precedence:
   command-line override, explicit YAML value, then a default derived from
   `dataset_dir`.
4. Relative YAML output paths are resolved from `dataset_dir`; relative YAML
   input paths are resolved from the configuration file's folder. CLI-relative
   paths are resolved from the current working directory.
5. The resulting immutable `PipelinePaths` object is passed to every pipeline
   stage. Its `display()` method produces the path summary used by `--dry_run`.

For example, with only `dataset_dir: /data/road_0047`, `image_dir` becomes
`/data/road_0047/images` and `colmap_dir` becomes
`/data/road_0047/colmap`. Setting `image_dir` explicitly changes only that
path.

## Configuration tests

The tests are contained in
[`tests/test_pipeline_config.py`](tests/test_pipeline_config.py). They verify:

- all default generated paths are derived under `dataset_dir`;
- individual YAML and command-line paths override only their corresponding
  defaults;
- malformed YAML structures raise `PipelineConfigError`;
- pipeline CLI arguments accept underscore names and reject dash names.

The tests use Python's built-in `unittest`; installing `pytest` is not
required. Run them from inside `iDriving_pipeline` with:

```bash
python -m unittest discover -s tests -q
```

Alternatively, run them from the repository root with:

```bash
python -m unittest discover -s iDriving_pipeline/tests -q
```
