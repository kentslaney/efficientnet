import sys, os, unittest
import tensorflow as tf
import tensorflow_datasets as tfds
from src.preprocessing import (
        RandAugmentCropped, RandAugmentPadded, PrepStretched)
from src.utils import (
        strftime, helper, ArgumentParser, cli_builder, relpath, Default)
from src.base import TFDSTrainer
from src.trainers import cli_names

# tf.config.optimizer.set_jit(True)
# os.environ["TF_XLA_FLAGS"] = "--tf_xla_cpu_global_jit"

def train_cli(parser):
    subparsers = parser.add_subparsers()
    for k, v in cli_names.items():
        sub = subparsers.add_parser(k)
        v.cli(sub)

@cli_builder
def preview(dataset="imagenette/320px-v2", pad=False, augment=True, size=224,
            data_dir=None, job_dir=None, fname="{time}.png", **kw):
    data, info = TFDSTrainer.as_dataset(dataset, data_dir, split="train")

    if augment:
        aug = RandAugmentPadded if pad else RandAugmentCropped
        aug = aug((size, size), data_format="channels_last")
        data = data.map(lambda x: {**x, "image": tf.clip_by_value(
            aug(x["image"]) / 5 + 0.5, 0., 1.)})

    fig = tfds.show_examples(data, info)
    if job_dir is not None:
        formatted = fname.format(time=strftime())
        path = os.path.join(job_dir, formatted)
        if formatted != fname:
            print(f"saving figure to {path}")
        with tf.io.gfile.GFile(path, "wb") as fp:
            fig.savefig(fp)

def preview_cli(parser):
    parser.add_argument("dataset", nargs="?", help=(
        'choose a TFDS dataset to augment, must have "image" key and a "train" '
        'split'))
    parser.add_argument("--job-dir", help=(
        "directory to write the output image to"))
    parser.add_argument("--fname", help="output file name")
    parser.add_argument("--data-dir", help=(
        "default directory for TFDS data, supports GCS buckets"))
    parser.add_argument("--size", metavar="N", type=int, help=(
        "set the output image size for each image previewed"))

    group = parser.add_mutually_exclusive_group(required=False)
    group.add_argument("--pad", action="store_true", help=(
        "pads the augmented images instead of cropping them"))
    group.add_argument(
        "--no-augment", dest="augment", action="store_false",
        help="don't augment the input")

    parser.set_defaults(call=preview)

@cli_builder
def download(dataset="imagenette/320px-v2", data_dir=None):
    return tfds.data_source(
            dataset, split="train", data_dir=data_dir,
            download_and_prepare_kwargs=TFDSTrainer.builder(dataset, data_dir))

def download_cli(parser):
    parser.add_argument("dataset", help="choose a TFDS dataset to download")
    parser.add_argument("--job-dir", dest="data_dir", help=(
        "default directory for TFDS data, supports GCS buckets"))
    parser.set_defaults(call=download)

def predict_cli(parser):
    parser.add_argument("model")
    parser.add_argument("file_input", nargs="?")
    parser.add_argument("--dataset-split", nargs=2)
    parser.add_argument("--job_dir")
    parser.add_argument("--data_dir")
    parser.add_argument(
            "--ckpt", dest="resume", help="defaults to latest in ./jobs")
    parser.add_argument("--dest", help=(
            "output npy location, defaults to ckpt/../logs, accepts filenames "
            "that end in npy"))

    parser.set_defaults(call=predict)

@cli_builder
def predict(
        model, file_input=None, dataset_split=(Default, "validation"),
        resume=None, dest=None, job_dir=None, data_dir=None, idx="*"):
    if job_dir is None:
        job_dir = os.path.join(os.path.dirname(__file__), "jobs")
    if resume is None:
        job = max(os.listdir(job_dir))
        query = os.path.join(job_dir, job, "ckpts", f"ckpt-{idx}.index")
        ckpts = tf.io.gfile.glob(query)
        idx = list(map(int, (os.path.basename(i)[5:-6] for i in ckpts)))
        resume = ckpts[max(enumerate(idx), key=lambda x: x[1])[0]]
        # until idx is supported
        resume = os.path.join(os.path.dirname(resume), os.path.pardir)
    if dest is None:
        # dest = os.path.join(os.path.dirname(resume), os.path.pardir, "logs")
        dest = os.path.join(resume, "logs")
    model = cli_names[model](
            resume=resume, augment=False, dataset=dataset_split[0])
    if file_input is None:
        ds, info = model.as_dataset(
                model.info.name, data_dir, split=dataset_split[1])
        ex = next(ds.take(1).as_numpy_iterator())
        im, mask = PrepStretched(shape=model.size)(ex["image"], ex["mask"])
        breakpoint()
        if not dest.endswith(".npy"):
            name = ex.get("filename", ex.get("image/id", strftime())) + ".npy"
            dest = os.path.join(dest, name)
    else:
        import cv2
        im = cv2.imread(file_input)
        im = PrepStretched(shape=model.size)(im)
        if not dest.endswith(".npy"):
            dest = os.path.join(dest, os.path.basename(file_input) + ".npy")
    res = model.model(im[None], training=False)[0]
    import numpy as np
    np.save(dest, res)
    print(f"saved result to {dest}")

def test():
    runner = unittest.TextTestResult(sys.stderr, True, 1)
    unittest.defaultTestLoader.discover(relpath()).run(runner)
    print()

def main(parser):
    subparsers = parser.add_subparsers()
    train_cli(subparsers.add_parser("train", help="Train a network"))
    preview_cli(subparsers.add_parser("preview", help=(
        "Preview an augmented dataset")))
    download_cli(subparsers.add_parser("download", help="Download a dataset"))
    predict_cli(subparsers.add_parser("predict", help="Run a model"))
    subparsers.add_parser("test", help="Run tests").set_defaults(call=test)
    parser.set_defaults(call=helper(parser))

    parser.parse_args().caller()

if __name__ == "__main__":
    main(ArgumentParser())
