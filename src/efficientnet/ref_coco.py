import tensorflow as tf
import tensorflow_datasets as tfds
import json
from tqdm import tqdm
import sys, re, pathlib
from setuptools import sandbox
from .utils import serialize_tensor_features
from .preprocessing import Reformat

class RefCOCOTrainer:
    # relative to `pathlib.Path(tfds.__file__).parent` or
    # https://github.com/tensorflow/datasets/blob/master/tensorflow_datasets/
    # datasets/ref_coco/manual_download_process.py
    @staticmethod
    def ref_coco_manual_download_process(
            COCO, REFER, ref_data_root, coco_annotations_file, out_file):
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
    def _tf_dataset_ref_coco(
            cls, data_dir, as_supervised=False, split=None, **kw):
        output_signature={
                "image": tf.TensorSpec((None, None, 3), dtype=tf.uint8),
                "mask": tf.TensorSpec(
                    (None, None, None, 3), dtype=tf.uint8),
                "label": tf.TensorSpec((None,), dtype=tf.int64)}
        parse_sig = {
                **output_signature, "mask": tf.TensorSpec(
                    (None, None), dtype=tf.int32)}
        def decode_fn(record_bytes):
            res = tf.io.parse_single_example(record_bytes, {
                k: tf.io.FixedLenFeature([], tf.string)
                for k in output_signature.keys()})
            return {
                    k: tf.io.parse_tensor(v, parse_sig[k].dtype)
                    for k, v in res.items()}
        def from_tfrecord(split, info):
            fname = f"ref_coco-{split}.tfrecord*"
            file_path = str(tf.io.gfile.join(info.data_dir, fname))
            file_path = tf.io.gfile.glob(file_path)
            if file_path:
                return tf.data.TFRecordDataset(file_path).take(
                        info.splits[split].num_examples).map(decode_fn)
        def split_tfrecord(split, iterator, info):
            total, shards = info.splits[split].num_examples, 10
            iterator = iter(iterator.as_numpy_iterator())
            def subset(i):
                if i == shards - 1:
                    yield from iterator
                    return
                for _ in range(total // shards * i, total // shards * (i + 1)):
                    yield next(iterator)
            for i in tqdm(range(shards)):
                fname = f"ref_coco-{split}.tfrecord-{i}-of-{shards}"
                write_tfrecord(subset(i), fname, info)
            return from_tfrecord(split, info)
        def write_tfrecord(iterator, fname, info):
            file_path = tf.io.gfile.join(info.data_dir, fname)
            with tf.io.TFRecordWriter(str(file_path)) as fp:
                for kw in iterator:
                    fp.write(serialize_tensor_features(kw).SerializeToString())
        def to_tfrecord(split, iterator, info):
            if split == "train":
                return split_tfrecord("train", iterator, info)
            write_tfrecord(iterator, f"ref_coco-{split}.tfrecord", info)
            return from_tfrecord(split, info)
        def body(data_source, bitpack=True):
            info = data_source.dataset_info
            data = tf.data.Dataset.from_generator(
                    lambda: (
                        {
                            "image": i["image"], "mask": i["objects"]["mask"],
                            "label": i["objects"]["label"]}
                        for i in data_source),
                    output_signature=output_signature)
            data = data.map(lambda x: {**x, "mask": tf.transpose(
                    tf.keras.ops.any(x['mask'], -1), (1, 2, 0))})
            if bitpack:
                fmt = Reformat()
                data = data.map(lambda x: {**x, "mask": fmt.recall(x["mask"])})
            return data, info
        sig = tf.function(input_signature=[tf.data.DatasetSpec(parse_sig)])
        @sig
        def tupled(data):
            return data.map(lambda x: (x["image"], x["mask"]))
        data_source = tfds.data_source(
                "ref_coco", split=split, data_dir=data_dir, try_gcs=True,
                download_and_prepare_kwargs=cls.builder("ref_coco", data_dir))
        mapping = tupled if as_supervised else sig(lambda x: x)
        if isinstance(split, str):
            info = data_source.dataset_info
            res = from_tfrecord(split, info)
            if res is not None:
                return mapping(res), info
            return mapping(to_tfrecord(split, *body(data_source))), info
        else:
            res = {
                    k: from_tfrecord(k, v.dataset_info)
                    for k, v in data_source.items()}
            info = next(iter(data_source.values())).dataset_info
            if all(i is not None for i in res.values()):
                return {k: mapping(v) for k, v in res.items()}, info
            data = {k: body(v) for k, v in data_source.items()}
            data = {k: mapping(to_tfrecord(k, *v)) for k, v in data.items()}
            return data, info
