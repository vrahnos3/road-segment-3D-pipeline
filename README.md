# 3D Reconstruction & Gaussian Splatting Pipeline

A modular Python workflow connecting partner data uploads to 3D visualization in a main platform through Kafka messaging, MinIO storage, reconstruction, and Gaussian Splatting.

Engineering highlights:

- **Pipeline orchestration:** reusable processing stages, YAML configuration, CLI overrides, consistent path resolution, and dry-run inspection.
- **Data integration:** asynchronous Kafka notification handling, MinIO downloads, and offset commits after successful downloads.
- **Reliability:** reconstruction quality checks, checkpoint validation, failure handling, and unit tests for configuration, processing, and integration logic.
- **Web integration:** a JavaScript 3D viewer served over HTTPS, connecting model outputs to browser-based visualization.

## Communication flow & execution

With the required dependencies installed, run both terminals from the directory containing this file. Configure MinIO access in [`minIO/download_minio.py`](minIO/download_minio.py), reconstruction/training in [`config/config.yaml`](config/config.yaml), and the outgoing Kafka broker/topic in [`kafka/producer.py`](kafka/producer.py). Set the visualization URL in [`kafka/templates/3d_reconstruction_ready_v2.json`](kafka/templates/3d_reconstruction_ready_v2.json) to the viewer address reachable by the main platform.

**1. Receive the partner's upload notification — Terminal 1**

```bash
python kafka/consumer.py \
  --broker YOUR_KAFKA_HOST:9092 \
  --topic YOUR_PARTNER_UPLOAD_TOPIC
```

The partner uploads data to MinIO and sends a Kafka notification identifying the bucket and object key. The consumer reads matching ready notifications, downloads each file into its own dataset directory, and commits Kafka progress after successful downloads. It prints the downloaded file and dataset paths and keeps listening.

**2. Reconstruct and train — Terminal 2**

Start the pipeline manually using the paths printed by the consumer:

```bash
python run_pipeline.py \
  --video_path /path/printed/by/consumer/drone_video.mp4 \
  --dataset_dir /path/printed/by/consumer \
  --converter_executable 3dgsconverter \
  --extract_images \
  --send_kafka_message \
  --start_viewer
```

The pipeline extracts frames, reconstructs camera poses with hloc/COLMAP, trains Splatfacto, and exports PLY/SPLAT models under `<dataset_dir>/export/`. Add `--dry_run` to preview the configuration or use `--help` to inspect options.

**3. Notify the main platform and visualize**

After training, export, conversion, and viewer-link publication succeed, the pipeline automatically invokes the Kafka producer to send a `3d_reconstruction_status: ready` message containing the configured visualization URL. The receiving platform component can consume this notification and use the URL to display the 3D scene. The pipeline then starts the HTTPS viewer; keep it running for platform access. Configure [`viewer/index.html`](viewer/index.html) to load the generated scene through `splat_files/current.splat`.

Built on [Nerfstudio](https://github.com/nerfstudio-project/nerfstudio) for Splatfacto training and Gaussian export, [3dgsconverter](https://github.com/francescofugazzi/3dgsconverter) for PLY-to-SPLAT conversion, and [GaussianSplats3D](https://github.com/mkkellogg/GaussianSplats3D) by Mark Kellogg for the Three.js-based web viewer. The custom development focuses on orchestration, configuration, validation, and integration of these tools.

