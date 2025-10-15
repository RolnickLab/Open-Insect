from openood.utils.config import setup_config
from openood.pipelines import get_pipeline

config = setup_config()
pipeline = get_pipeline(config)
pipeline.run()
