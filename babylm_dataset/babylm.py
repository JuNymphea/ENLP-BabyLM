"""
This is the script that makes data into a huggingface dataset!

You can call 
    datasets.load_dataset("path/to/this/script/on/your/machine.py", name)
where name is one of the strings in the class Synthetic (eg. flat-parens_vocabsize-500_deplength-en)
And access any of the synthetic corpora as you would any other dataset. 

You can add new entries to the BUILDER_CONFIGS list following the pattern

Instructions about what this type of script is are here https://huggingface.co/docs/datasets/dataset_script
"""

import os
import datasets
import zipfile

_DESCRIPTION = """\
    BabyLM Dataset.
"""
# TODO Change the directory to be absolute, I think it can cause some problems 
# otherwise. This directory should point to the data directory in this dir, but the 
# absolute path from your machine
_DATA_DIR = "babylm_dataset"

class BabyLMConfig(datasets.BuilderConfig):

    def __init__(self, data_dir, **kwargs):
        """BuilderConfig for IzParens

        Args:
          data_dir: `string`, directory of the dataset (word or raw level)
          **kwargs: keyword arguments forwarded to super.
        """
        super(BabyLMConfig, self).__init__(
            **kwargs,
        )
        self.data_dir = data_dir

class BabyLM(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        BabyLMConfig(
            description="BabyLM Data.",
            data_dir = _DATA_DIR
        ),
        
        ]

    def _info(self):
        return datasets.DatasetInfo(
            # This is the description that will appear on the datasets page.
            description=_DESCRIPTION,
            # datasets.features.FeatureConnectors
            features=datasets.Features(
                {
                    "text": datasets.Value("string")
                    # These are the features of your dataset like images, labels ...
                }
            ),
            # If there's a common (input, target) tuple from the features,
            # specify them here. They'll be used if as_supervised=True in
            # builder.as_dataset.
            supervised_keys=None,
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN,
                gen_kwargs={"data_file": os.path.join(self.config.data_dir, "train_100M.zip"), "split": "train"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST,
                gen_kwargs={"data_file": os.path.join(self.config.data_dir, "test.zip"), "split": "test"},
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION,
                gen_kwargs={"data_file": os.path.join(self.config.data_dir, "dev.zip"), "split": "valid"},
            )
        ]

    def _generate_examples(self, data_file, split):
        with zipfile.ZipFile(data_file, "r") as zipf:
            for name in zipf.namelist():
                with zipf.open(name) as f:
                    for idx, line in enumerate(f):
                        yield idx, {"text": line.decode("utf-8").strip()}