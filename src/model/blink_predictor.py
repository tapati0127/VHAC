import logging
from typing import List, Dict

from params.algo_params import C4IBLINKInferenceConfig
from src.model.blink import load_models
from src.model.blink.inference import get_context, get_contexts

logger = logging.getLogger(__name__)


class BLINKPredictor:
    def __init__(self, configs: C4IBLINKInferenceConfig):
        self._logger = logging.getLogger(__name__)
        self._configs = configs
        self.models = load_models(self._configs, logger)

    def get_context(self, blink_input: List):
        try:
            _id, entity_title, text = get_context(
                self._configs,
                logger,
                *self.models,
                test_data=blink_input)
            return _id, entity_title, text

        except Exception as ex:
            self._logger.error(ex)
            return None, None, None

    def get_contexts(self, blink_input: List, top_k=1):
        try:
            contexts = get_contexts(
                self._configs,
                logger,
                *self.models,
                test_data=blink_input,
                top_k=top_k)
            return contexts

        except Exception as ex:
            self._logger.error(ex)
            return None, None, None