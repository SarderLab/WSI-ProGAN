## Progressive Growing of GANs for Whole Slide Images<br><i>– A modified version of ProGAN for optimized training with histological WSIs</i>


This code was forked from https://github.com/tkarras/progressive_growing_of_gans and modified to work optimally on Whole Slide images WSIs by [Brendon Lutnick](https://github.com/brendonlutnick)

![Representative image](https://github.com/SarderLab/WSI-ProGAN/blob/master/representitive-image.png)<br>
**Picture:** A patch of kidney tissue that was dreamed up by a random number generator.

**Abstract:**<br>
*We modify the progressive_growing_of_gans code to work with hitological whole slide images. This uses a custom input pipeline to stochastically feed images patches at training time: [avalable here](https://github.com/SarderLab/tf-WSI-dataset-utils) .*

## Resources

* [ProGAN Paper (NVIDIA research)](http://research.nvidia.com/publication/2017-10_Progressive-Growing-of)
* [Additional material (UBBox)](https://buffalo.box.com/s/8sl2k01svciu1a5qex4g4ziyox39204c)
  * [Pre-trained networks (human kidney biopsies)](https://buffalo.box.com/s/2jtuzqudgs27mvo6izqosib1h979hmtn)
  * [1000 generated images](https://buffalo.box.com/s/ra5gp06kwcadpd9cefnqq0p103utip9x)
  * [Video interpolation (latent walk)](https://buffalo.box.com/s/88cxodei9u65suwxpt30a5pczj7p2d65)

## System requirements

* Linux.
* 64-bit Python 3.6 installation with numpy 1.13.3 or newer.
* One or more high-end NVIDIA GPU with 8GB of DRAM. To use the full resolution network 16GB DRAM is needed.
* NVIDIA driver 391.25 or newer, CUDA toolkit 9.0 or newer, cuDNN 7.1.2 or newer.
* Additional Python packages listed in `requirements-pip.txt`

## Importing and using pre-trained networks

All pre-trained networks found on UBBox, are stored as Python PKL files. They can be imported using the standard `pickle` mechanism as long as two conditions are met: (1) The directory containing the Progressive GAN code repository must be included in the PYTHONPATH environment variable, and (2) a `tf.Session()` object must have been created beforehand and set as default. Each PKL file contains 3 instances of `tfutil.Network`:

```
# Import kindey network.
with open('network-snapshot-001657.pkl', 'rb') as file:
    G, D, Gs = pickle.load(file)
    # G = Instantaneous snapshot of the generator, mainly useful for resuming a previous training run.
    # D = Instantaneous snapshot of the discriminator, mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator, yielding higher-quality results than the instantaneous snapshot.
```

Once you have imported the networks, you can call `Gs.run()` to produce a set of images for given latent vectors, or `Gs.get_output_for()` to include the generator network in a larger TensorFlow expression. For further details, please consult the original ProGAN [documentation](https://github.com/tkarras/progressive_growing_of_gans)

## Preparing datasets for training

The Progressive GAN code repository contains a file "dataset.py" which has a custom class "WSIDataset" used to create a dataset from a folder of WSI images:

```
usage: dataset.py ...
    wsi_ext             file extension of the WSIs in the dataset folder.

usage: config.py ...
    tfrecord_dir        the location of the dataset folder.

```

## Training networks

Once the necessary datasets are set up, you can proceed to train your own networks. The general procedure is as follows:

1. Edit `config.py` to specify the dataset and training configuration by uncommenting/editing specific lines.
2. Run the training script with `python train.py`.
3. The results are written into a newly created subdirectory under `config.result_dir`
4. Wait several days (or weeks) for the training to converge, and analyze the results.

By default, `config.py` is configured to train a 512x512 network for CelebA-HQ using a single-GPU. This is expected to take about two weeks even on the highest-end NVIDIA GPUs. The key to enabling faster training is to employ multiple GPUs and/or go for a lower-resolution dataset. To this end, `config.py` contains several examples for commonly used datasets, as well as a set of "configuration presets" for multi-GPU training. All of the presets are expected to yield roughly the same image quality for CelebA-HQ, but their total training time can vary considerably:

* `preset-v1-1gpu`: Original config that was used to produce the CelebA-HQ and LSUN results shown in the paper. Expected to take about 1 month on NVIDIA Tesla V100.
* `preset-v2-1gpu`: Optimized config that converges considerably faster than the original one. Expected to take about 2 weeks on 1xV100.
* `preset-v2-2gpus`: Optimized config for 2 GPUs. Takes about 1 week on 2xV100.
* `preset-v2-4gpus`: Optimized config for 4 GPUs. Takes about 3 days on 4xV100.
* `preset-v2-8gpus`: Optimized config for 8 GPUs. Takes about 2 days on 8xV100.

Other noteworthy config options:

* `fp16`: Enable [FP16 mixed-precision training](http://docs.nvidia.com/deeplearning/sdk/mixed-precision-training/index.html) to reduce the training times even further. The actual speedup is heavily dependent on GPU architecture and cuDNN version, and it can be expected to increase considerably in the future.
* `BENCHMARK`: Quickly iterate through the resolutions to measure the raw training performance.
* `BENCHMARK0`: Same as `BENCHMARK`, but only use the highest resolution.
* `syn1024rgb`: Synthetic 1024x1024 dataset consisting of just black images. Useful for benchmarking.
* `VERBOSE`: Save image and network snapshots very frequently to facilitate debugging.
* `GRAPH` and `HIST`: Include additional data in the TensorBoard report.

## Analyzing results

Training results can be analyzed in several ways:

* **Manual inspection**: The training script saves a snapshot of randomly generated images at regular intervals in `fakes*.png` and reports the overall progress in `log.txt`.
* **TensorBoard**: The training script also exports various running statistics in a `*.tfevents` file that can be visualized in TensorBoard with `tensorboard --logdir <result_subdir>`.
* **Generating images and videos**: At the end of `config.py`, there are several pre-defined configs to launch utility scripts (`generate_*`). For example:
  * Suppose you have an ongoing training run titled `010-pgan-celebahq-preset-v1-1gpu-fp32`, and you want to generate a video of random interpolations for the latest snapshot.
  * Uncomment the `generate_interpolation_video` line in `config.py`, replace `run_id=10`, and run `python train.py`
  * The script will automatically locate the latest network snapshot and create a new result directory containing a single MP4 file.
* **Quality metrics**: Similar to the previous example, `config.py` also contains pre-defined configs to compute various quality metrics (Sliced Wasserstein distance, Fréchet inception distance, etc.) for an existing training run. The metrics are computed for each network snapshot in succession and stored in `metric-*.txt` in the original result directory.
