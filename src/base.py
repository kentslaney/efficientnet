import os
import tensorflow as tf
import tensorflow_datasets as tfds
from src.utils import (
    tpu_prep, WarmedExponential, LRTensorBoard, strftime, parse_strategy,
    PresetFlag, cli_builder, relpath, Checkpointer, RequiredLength, Default
)
from src.preprocessing import (
    PrepStretched, RandAugmentCropped, RandAugmentPadded
)

class Trainer:
    path, _format, length, validation = None, None, None, None
    tb_callback, ckpt_callback = LRTensorBoard, Checkpointer
    opt = tf.keras.optimizers.Adam

    # creates the CLI interface; arguments are passed to init and default to
    # the function defaults by passing the Default flag to cli_builder
    @classmethod
    def cli(cls, parser):
        parser.add_argument("--id", dest="suffix", help=(
            "name template for the training directory, compiled using "
            "python's string formatting; time is the only currently supported "
            "variable"))
        parser.add_argument(
            "--batch", metavar="SIZE", type=int, help=(
                "batch size per visible device (1 unless distributed)"))
        parser.add_argument("--epochs", metavar="N", type=int, help=(
            "how many epochs to run"))
        parser.add_argument(
            "--lr", dest="learning_rate", type=float, help=(
                "model learning rate per example per batch"))
        parser.add_argument("--decay-warmup", type=int, help=(
            "number of epochs to warm up learning rate"))
        parser.add_argument("--decay-factor", type=float, help=(
            "lr decay per epoch after warmup"))
        parser.add_argument("--profile", type=int, action=RequiredLength(1, 2),
                            metavar="N", help="batches to profile")

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--no-decay", dest="decay", action="store_false",
                           help="don't decay the learning rate")
        group.add_argument(
            "--set-decay", dest="decay", action="store_true", help=(
                "set the decay flag (this is the default behavior in base "
                "class but can be overriden by subclasses, this allows it to "
                "be set at the CLI level)"))

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "--channels-last", dest="data_format", action="store_const",
            const="channels_last", help=(
                "forces the image data format to be channels last, which is "
                "the default for CPUs and TPUs"))
        group.add_argument(
            "--channels-first", dest="data_format", action="store_const",
            const="channels_first", help=(
                "forces the image data format to be channels first, which is "
                "the default for GPUs"))

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "--no-base", action="store_const", const=None, dest="base", help=(
                "prevents saving checkpoints or tensorboard logs to disk"))
        group.add_argument("--job-dir", dest="base", metavar="PATH", help=(
            "prefix for training directory"))
        group.add_argument("--resume", metavar="PATH", help="load from path")

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument(
            "--distribute", metavar=("STRATEGY", "DEVICE"), nargs="+", help=(
                "what distribution strategy to use from tf.distribute, and "
                "what devices to distribute to (usually no specified device "
                "implies all visable devices); leaving this unspecified "
                "results in no strategy, and uses tensorflow's default "
                "behavior"))
        group.add_argument(
            "--tpu", metavar="DEVICE", nargs="*", dest="distribute",
            action=PresetFlag("TPUStrategy"), help=(
                "use TPUStrategy for distribution; equivalent to --distribute "
                "TPUStrategy [DEVICE...]"))
        group.add_argument(
            "--mirror", metavar="DEVICE", nargs="*", dest="distribute",
            action=PresetFlag("MirroredStrategy"), help=(
                "use MirroredStrategy for distribution; equivalent to "
                "--distribute MirroredStrategy [DEVICE...]"))

        parser.set_defaults(call=cls.train)

    # see the CLI help for a complete list of parameters
    @cli_builder
    def __init__(self, base=relpath("jobs"), data_format=None, batch=64,
                 distribute=None, epochs=1000, decay=True, suffix="{time}",
                 learning_rate=6.25e-5, decay_warmup=5, decay_factor=0.99,
                 resume=None, profile=(0,), supervised_mapping="one_hot"):
        self.batch, self.learning_rate = batch, learning_rate
        self._format, self.epochs = data_format, epochs
        self.decay_warmup, self.decay_factor = decay_warmup, decay_factor
        self.should_decay, self.callbacks = decay, []
        self.step = tf.Variable(0, dtype=tf.int32, name="step")
        self.epoch = tf.Variable(0, dtype=tf.int32, name="epoch")
        self.profile = profile[0] if len(profile) == 1 else tuple(profile)
        self.mapper = supervised_mapping

        self.distribute(*parse_strategy(distribute))
        if resume is not None:
            self.path = os.path.abspath(resume)
        elif base is not None:
            self.path = os.path.join(base, suffix.format(time=strftime()))

        if self.should_decay:
            assert self.length is not None
            self.callbacks.append(WarmedExponential(
                self.learning_rate, self.length / self.batch,
                self.decay_warmup, self.decay_factor, self.step))

    # decides default data format based on device
    @property
    def data_format(self):
        if self._format is None:
            return "channels_last" if self.tpu or not bool(
                tf.config.experimental.list_physical_devices('GPU')) \
                else "channels_first"
        return self._format

    # sets the tf distribute strategy and adjusts the optimizer and batch size
    # accordingly
    def distribute(self, strategy, tpu=False):
        self.strategy, self.tpu = strategy, tpu
        if tf.distribute.in_cross_replica_context():
            self.batch *= strategy.num_replicas_in_sync
        self.learning_rate *= self.batch
        self.opt = self.opt(self.learning_rate)
        self.opt = tf.tpu.CrossShardOptimizer(self.opt) if tpu else self.opt

    # given a preprocessing function, sets the behavior for how to apply it
    # defaults to classification mapping
    mapper = "one_hot"

    @property
    def preprocessor(self):
        return getattr(self, "_mapper_" + self.mapper) \
                if isinstance(self.mapper, str) else self.mapper

    def _mapper_one_hot(self, f):
        return lambda x, y: (f(x), tf.one_hot(y, self.outputs))

    def _mapper_mask(self, f):
        return f

    # batches and preprocesses to train and optionally validation sets
    def preprocess(self, train, validation=None):
        tune = tf.data.experimental.AUTOTUNE
        options = tf.data.Options()
        options.experimental_deterministic = False
        if self.tpu:
            train, validation = map(tpu_prep, (train, validation))

        self.dataset = self.dataset.shuffle(self.batch * 8).map(
            self.preprocessor(train), tune).batch(self.batch, True).prefetch(
                tune).with_options(options)

        if validation is not None and self.validation is not None:
            self.validation = self.validation.map(self.preprocessor(
                validation), tune).batch(self.batch, True).prefetch(
                    tune).with_options(options)

    # sets up the checkpoint and logging directories and adds callbacks for
    # self.ckpt_callback and self.tb_callback
    def register(self):
        if self.path is None:
            return

        # create ckpts and logs directories inside self.path
        ckpts, logs = (os.path.join(self.path, i) for i in ("ckpts", "logs"))
        tf.io.gfile.makedirs(ckpts)

        ckptr = self.ckpt_callback(os.path.join(ckpts, "ckpt"), self.epoch,
                                   self.model, opt=self.opt, step=self.step)
        # load latest if there are already checkpoints
        if tf.io.gfile.listdir(ckpts):
            path = ckptr.restore(ckpts)
            print(f"Loading model from checkpoint {path}")
        else:
            print(f'Writing to training directory {self.path}')

        self.callbacks += [self.tb_callback(
            logs, update_freq=64, profile_batch=self.profile), ckptr]

    # calls outputs and compiles the model using self.opt
    def compile(self, *args, **kw):
        self.register()
        self.model.compile(self.opt, *args, **kw)

    # creates a class instance using the input keywords and starts training
    @classmethod
    def train(cls, **kw):
        cls(**kw).fit()

    # calls the keras model class' fit function using the appropiate properties
    def fit(self):
        self.model.fit(
            self.dataset, validation_data=self.validation, epochs=self.epochs,
            callbacks=self.callbacks, initial_epoch=self.epoch.numpy().item())

class TFDSTrainer(Trainer):
    @classmethod
    def cli(cls, parser):
        parser.add_argument("--dataset", help=(
            "choose which TFDS dataset to train on; must be classification "
            "and support as_supervised"))
        parser.add_argument("--data-dir", help=(
            "default directory for TFDS data, supports GCS buckets"))
        super().cli(parser)

    @property
    def outputs(self):
        return self.info.features["label"].num_classes

    ds_defaults = {"ref_coco": {"supervised_mapping": "mask"}}
    @cli_builder
    def __init__(self, dataset, data_dir=None, **kw):
        data, info = self.as_dataset(
                dataset, data_dir=data_dir, as_supervised=True)
        self.dataset, self.validation = data["train"], data["validation"]
        self.info, self.length = info, info.splits["train"].num_examples
        defaults = self.ds_defaults.get(dataset, {})
        defaults = {
                k: v for k, v in defaults.items()
                if k not in kw or kw[k] is Default}
        super().__init__(**{**kw, **defaults})

    @classmethod
    def builder(cls, dataset, data_dir):
        if hasattr(cls, "_tfds_" + dataset) and not tf.io.gfile.exists(
                tfds.builder(dataset, data_dir=data_dir).data_path):
            return getattr(cls, "_tfds_" + dataset)(data_dir)

    @staticmethod
    def gfile_download(data_dir, url_dict):
        data_dir = os.path.expanduser(
            tfds.core.constants.DATA_DIR if data_dir is None else data_dir)
        data_base = os.path.join(data_dir, "downloads", "manual")
        downloader = tfds.download.DownloadManager(download_dir=data_base)
        paths = []
        for url_base in sorted(url_dict.keys()):
            same_name = isinstance(url_dict[url_base], tuple)
            for fname in (url_dict[url_base],)[0 if same_name else slice(None)]:
                paths.append(tfds.download.Resource(
                    path=os.path.join(data_base, fname), url=url_base + (
                        fname if same_name else "")))
        return downloader, downloader.download(paths), paths

    @classmethod
    def _tfds_imagenet2012(cls, data_dir):
        url_base = "https://image-net.org/data/ILSVRC/2012/"
        files = ("ILSVRC2012_img_train.tar", "ILSVRC2012_img_val.tar")
        cls.gfile_download(data_dir, {url_base: files})

    # relative to `pathlib.Path(tfds.__file__).parent` or
    # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/
    # datasets/ref_coco/manual_download_process.py
    @staticmethod
    def ref_coco_manual_download_process(
            COCO, REFER, ref_data_root, coco_annotations_file, out_file):
        import json
        all_refs = []
        for dataset, split_bys in [
            ('refcoco', ['google', 'unc']),
            ('refcoco+', ['unc']),
            ('refcocog', ['google', 'umd']),
        ]:
          for split_by in split_bys:
            refer = REFER(ref_data_root, dataset, split_by)
            for ref_id in refer.getRefIds():
              ref = refer.Refs[ref_id]
              ann = refer.refToAnn[ref_id]
              ref['ann'] = ann
              ref['dataset'] = dataset
              ref['dataset_partition'] = split_by
              all_refs.append(ref)

        coco = COCO(coco_annotations_file)
        ref_image_ids = set(x['image_id'] for x in all_refs)
        coco_anns = {
            image_id: {
                'info': coco.imgs[image_id], 'anns': coco.imgToAnns[image_id]}
            for image_id in ref_image_ids
        }

        with open(out_file, 'w') as f:
          json.dump({'ref': all_refs, 'coco_anns': coco_anns}, f)

    @classmethod
    def _tfds_ref_coco(cls, data_dir):
        import sys, re, pathlib
        from setuptools import sandbox
        unc_url_base = (
                "https://web.archive.org/web/20220515000000/"
                "https://bvisionweb1.cs.unc.edu/licheng/referit/data/")
        unc_files = {
                unc_url_base: (
                    "refclef.zip", "refcoco.zip", "refcoco+.zip",
                    "refcocog.zip"),
                unc_url_base + "images/": ("saiapr_tc-12.zip",)}
        train2014 = {
                (
                    "http://images.cocodataset.org/annotations/"
                    "annotations_trainval2014.zip"): "annotations",
                "http://images.cocodataset.org/zips/": ("train2014.zip",)}
        github_url = "https://github.com/{}/archive/refs/heads/master.zip"
        coco_apis = {
                **unc_files, **train2014,
                github_url.format("lichengunc/refer"): "refer-master.zip",
                github_url.format("cocodataset/cocoapi"): "cocoapi-master.zip"}
        dl, out, sym = cls.gfile_download(data_dir, coco_apis)
        mapping = {name.path.stem: path for name, path in zip(sym, out)}
        data_root = dl.download_dir / "data"
        rename = {"train2014": "coco_train2014"}
        for path in (
                ("refcoco",), ("refcoco+",), ("refcocog",), ("refclef",),
                ("annotations",), ("refer-master",), ("cocoapi-master",),
                ("images", "saiapr_tc-12"),
                ("images", "mscoco", "images", "train2014")):
            res = data_root
            for subfolder in path[:-1]:
                res /= subfolder
            tf.io.gfile.makedirs(res)
            res /= rename.get(path[-1], path[-1])
            if not tf.io.gfile.exists(res):
                extracted = dl.extract(mapping[path[-1]])
                (extracted / path[-1]).rename(res)
                tf.io.gfile.rmtree(extracted)
        cocoapi = data_root / "cocoapi-master" / "PythonAPI"
        refer_path = data_root / "refer-master"
        with open(cocoapi / "setup.py", "r") as fp:
            diff = fp.read().replace("'-Wno-cpp', '-Wno-unused-function', ", "")
        diff = re.sub(
                "ext_modules ?= ?ext_modules",
                "ext_modules = cythonize(ext_modules)", diff)
        diff = re.sub(
                "^from setuptools ",
                "from Cython.Build import cythonize\nfrom setuptools ", diff)
        with open(cocoapi / "setup.py", "w") as fp:
            fp.write(diff)
        with open(refer_path / "refer.py", "r") as fp:
            py2 = fp.read().replace("cPickle", "pickle")\
                    .replace("import skimage.io as io", "")\
                    .replace("pickle.load(open(ref_file, 'r'))",
                             "pickle.load(open(ref_file, 'rb'))")\
                    .replace("images/mscoco/images/train2014",
                             "images/mscoco/images/coco_train2014")
        py3 = re.sub("^([ \t]*print) (.*)$", r"\1(\2)", py2, flags=re.MULTILINE)
        with open(refer_path / "refer.py", "w") as fp:
            fp.write(py3)
        out_file = data_root / "images" / "mscoco" / "images" / "refcoco.json"
        sandbox.run_setup(cocoapi / "setup.py", ['build_ext', '--inplace'])
        so = tf.io.gfile.glob(str(cocoapi / "pycocotools" / "_mask.*.so"))
        basename = pathlib.Path(so[0]).name
        tf.io.gfile.copy(so[0], refer_path / "external" / basename, True)
        extra_paths = (refer_path, cocoapi)
        for path in extra_paths:
            sys.path.insert(0, str(path))
        if not tf.io.gfile.exists(out_file):
            from pycocotools.coco import COCO
            from refer import REFER
            annotations = data_root / "annotations" / "instances_train2014.json"
            cls.ref_coco_manual_download_process(
                    COCO, REFER, data_root, annotations, out_file)
        return {"download_config": tfds.download.DownloadConfig(
                manual_dir=str(data_root / "images" / "mscoco" / "images"))}

    @classmethod
    def as_dataset(cls, dataset, data_dir, **kw):
        if hasattr(cls, "_tf_dataset_" + dataset):
            return getattr(cls, "_tf_dataset_" + dataset)(data_dir, **kw)
        else:
            cls.builder(dataset, data_dir)
            return tfds.load(
                    dataset, data_dir=data_dir, with_info=True, try_gcs=True,
                    shuffle_files=True, **kw)

    @classmethod
    def _tf_dataset_ref_coco(
            cls, data_dir, as_supervised=False, split=None, **kw):
        def body(data_source):
            info = data_source.dataset_info
            data = tf.data.Dataset.from_generator(
                    lambda: (
                        {
                            "image": i["image"], "mask": i["objects"]["mask"],
                            "label": i["objects"]["label"]}
                        for i in data_source),
                    output_signature={
                        "image": tf.TensorSpec((None, None, 3), dtype=tf.uint8),
                        "mask": tf.TensorSpec(
                            (None, None, None, 3), dtype=tf.uint8),
                        "label": tf.TensorSpec((None,), dtype=tf.int64)})
            data = data.map(lambda x: {**x, "mask": tf.transpose(
                    tf.keras.ops.any(x['mask'], -1), (1, 2, 0))})
            if as_supervised:
                data = data.map(lambda x: (x["image"], x["mask"]))
            return data, info
        data_source = tfds.data_source(
                "ref_coco", split=split, data_dir=data_dir, try_gcs=True,
                download_and_prepare_kwargs=cls.builder("ref_coco", data_dir))
        if isinstance(split, str):
            return body(data_source)
        else:
            data = {k: body(v) for k, v in data_source.items()}
            info = next(iter(data.values()))[1]
            data = {k: v[0] for k, v in data.items()}
            return data, info

class RandAugmentTrainer(Trainer):
    @classmethod
    def cli(self, parser):
        parser.add_argument("--size", metavar="N", type=int, help=(
            "force the input image to be a certain size, will default to the "
            "recommended size for the preset if unset"))

        group = parser.add_mutually_exclusive_group(required=False)
        group.add_argument("--pad", action="store_true", help=(
            "pads the augmented images instead of cropping them, this is the "
            "default behavior for the original RandAugment implementation"))
        group.add_argument(
            "--no-augment", dest="augment", action="store_false",
            help="don't augment the input")
        super().cli(parser)

    @cli_builder
    def __init__(self, size=None, augment=True, pad=False, **kw):
        super().__init__(**kw)
        self.size = (size, size) if type(size) == int else size
        self.augment, self.pad = augment, pad
        train = PrepStretched if not self.augment else \
            RandAugmentPadded if self.pad else RandAugmentCropped
        train = train(self.size, data_format=self.data_format)
        validation = PrepStretched(self.size, data_format=self.data_format)
        super().preprocess(train, validation)
