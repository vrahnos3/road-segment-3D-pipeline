# 3D Reconstruction & Gaussian Splatting Pipeline

A modular Python pipeline that turns drone video into interactive 3D scenes through frame extraction, camera reconstruction with hloc/COLMAP, Gaussian Splatting training, and PLY/SPLAT export.

Engineering highlights:

- **Pipeline orchestration:** reusable processing stages, YAML configuration, CLI overrides, consistent path resolution, and dry-run inspection.
- **Data integration:** asynchronous Kafka notification handling, MinIO downloads, and offset commits after successful downloads.
- **Reliability:** reconstruction quality checks, checkpoint validation, failure handling, and unit tests for configuration, processing, and integration logic.
- **Web integration:** a JavaScript 3D viewer served over HTTPS, connecting model outputs to browser-based visualization.

## Run

With the required dependencies installed, run from the directory containing this file:

```bash
# Inspect available options and resolved configuration
python run_pipeline.py --help
python run_pipeline.py --dry_run

# Process a local video and export the trained scene
python run_pipeline.py \
  --video_path /path/to/drone_video.mp4 \
  --dataset_dir /path/to/output/scene \
  --converter_executable 3dgsconverter \
  --extract_images \
  --no_send_kafka_message \
  --no_start_viewer
```

Adjust reconstruction and training settings in [`config/config.yaml`](config/config.yaml). Exported models are saved under `<dataset_dir>/export/`. Add `--dry_run` to the processing command to preview its paths without running it.

Built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) for Splatfacto training and Gaussian export, [3dgsconverter](https://github.com/francescofugazzi/3dgsconverter) for PLY-to-SPLAT conversion, and [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) by Mark Kellogg for the Three.js-based web viewer. The custom development focuses on orchestration, configuration, validation, and integration of these tools.
