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
                    "resp": datasets.Value("string"),
                }
            )
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={
                    "filepath": "train.jsonl",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={
                    "filepath": "dev_mod.jsonl",
                }
            ),
        ]

    def _generate_examples(self, filepath):
        with open(filepath, encoding="utf-8-sig") as f:
            for id_, line in enumerate(f):
                data = json.loads(line)
                yield id_, {
                    "weibo": data["weibo"],
                    "resp": data["resp"],
                }