import os
from typing import List


class C4IBLINKInferenceConfig:
    def __init__(self, **kwargs):
        self.labels_to_link = kwargs.get('labels_to_link', ['VEH'])
        self.mode = kwargs.get('mode', '')
        self.device = kwargs.get('device', 'auto')
        self.top_candidates = kwargs.get('top_candidates', 16)
        self.find_thresh = kwargs.get('find_thresh', 0.75)
        self.bert_base_dir = kwargs.get('bert_base_dir')
        self.biencoder_model = os.path.join(
            kwargs.get('biencoder_dir'), 'pytorch_model.bin'
        )
        self.biencoder_config = os.path.join(
            kwargs.get('biencoder_dir'), 'training_params.txt'
        )
        self.crossencoder_model = os.path.join(
            kwargs.get('crossencoder_dir'), 'pytorch_model.bin'
        )
        self.crossencoder_config = os.path.join(
            kwargs.get('crossencoder_dir'), 'training_params.txt'
        )
        self.entity_catalogue = os.path.join(
            kwargs.get('entity_catalogue_dir'), 'documents.jsonl'
        )
        self.entity_encoding = os.path.join(
            kwargs.get('candidate_encode_dir'), 'cand_encode.t7'
        )
