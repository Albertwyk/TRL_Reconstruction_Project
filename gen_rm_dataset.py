import json
import datasets

class WeiboCommentsConfig(datasets.BuilderConfig):
    def __init__(self, **kwargs):
        super(WeiboCommentsConfig, self).__init__(**kwargs)

class WeiboComments(datasets.GeneratorBasedBuilder):
    VERSION = datasets.Version("1.0.0")
    BUILDER_CONFIGS = [
        WeiboCommentsConfig(name="weibo_comments", version=VERSION, description="Weibo comments dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "weibo": datasets.Value("string"),
                    "text_j": datasets.Value("string"),
                    "text_k": datasets.Value("string"),
                    "text_j_like": datasets.Value("int32"),
                    "text_k_like": datasets.Value("int32"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": "RM/train.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": "RM/dev.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={
                    "filepath": "RM/test.json",
                }
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8-sig") as f:
            for id_, line in enumerate(f):
                data = json.loads(line)
                yield id_, {
                    "weibo": data["weibo"],
                    "text_j": data["text_j"],
                    "text_k": data["text_k"],
                    "text_j_like": data["text_j_like"],
                    "text_k_like": data["text_k_like"],
                }