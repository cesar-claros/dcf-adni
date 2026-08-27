import logging
import hydra
from omegaconf import DictConfig, OmegaConf
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import dcf_adni.utils as utils
from dcf_adni.preprocessing.adni_preprocess import ADNIPreprocess

log = logging.getLogger(__name__)
OmegaConf.register_new_resolver("tuple", lambda lst: tuple(lst))


@hydra.main(config_path="../configs/", config_name="config.yaml")
def main(config: DictConfig):
    # Pretty print config using Rich library
    if config.get("print_config"):
        utils.print_config(config, resolve=True)

    log.info("Instantiating preprocessing pipeline")
    preprocessing_pipeline = hydra.utils.instantiate(
        config.preprocessing_pipeline, _recursive_=False
    )

    # Run ADNI data preprocessing
    log.info("Running ADNI data preprocessing")
    preprocessor = ADNIPreprocess(**config.preprocessing_pipeline)
    preprocessor.run()


if __name__ == "__main__":
    main()
