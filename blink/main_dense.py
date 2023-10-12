# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import json
import ast

import torch
import numpy as np
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset

from src.model.blink.biencoder.biencoder import load_biencoder
from src.model.blink.crossencoder.crossencoder import load_crossencoder
from src.model.blink.biencoder.data_process import process_mention_data
from src.model.blink.crossencoder.train_cross import evaluate
from src.model.blink.indexer.faiss_indexer import DenseFlatIndexer, DenseHNSWFlatIndexer


def _load_candidates(
    entity_catalogue, entity_encoding, faiss_index=None, index_path=None, logger=None
):
    # only load candidate encoding if not using faiss index
    if faiss_index is None:
        candidate_encoding = torch.load(entity_encoding)
        indexer = None
    else:
        if logger:
            logger.info('Using faiss index to retrieve entities.')
        candidate_encoding = None
        assert index_path is not None, 'Error! Empty indexer path.'
        if faiss_index == 'flat':
            indexer = DenseFlatIndexer(1)
        elif faiss_index == 'hnsw':
            indexer = DenseHNSWFlatIndexer(1)
        else:
            raise ValueError('Error! Unsupported indexer type! Choose from flat,hnsw.')
        indexer.deserialize_from(index_path)

    # load all the 5903527 entities
    title2id = {}
    id2title = {}
    id2text = {}
    wikipedia_id2local_id = {}
    local_idx = 0
    with open(entity_catalogue, 'r') as fin:
        lines = fin.readlines()
        for line in lines:
            entity = json.loads(line)

            if 'idx' in entity:
                split = entity['idx'].split('curid=')
                if len(split) > 1:
                    wikipedia_id = int(split[-1].strip())
                else:
                    wikipedia_id = entity['idx'].strip()

                assert wikipedia_id not in wikipedia_id2local_id
                wikipedia_id2local_id[wikipedia_id] = local_idx

            title2id[entity['title']] = local_idx
            id2title[local_idx] = entity['title']
            id2text[local_idx] = entity['text']
            local_idx += 1
    return (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        indexer,
    )


def _process_biencoder_dataloader(samples, tokenizer, biencoder_params):
    _, tensor_data = process_mention_data(
        samples,
        tokenizer,
        biencoder_params['max_context_length'],
        biencoder_params['max_cand_length'],
        silent=True,
        logger=None,
        debug=biencoder_params['debug'],
    )
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=biencoder_params['eval_batch_size']
    )
    return dataloader


def _run_biencoder(biencoder, dataloader, candidate_encoding, top_k=100, indexer=None):
    biencoder.model.eval()
    labels = []
    nns = []
    all_scores = []
    for batch in dataloader:
        context_input, _, label_ids = batch
        with torch.no_grad():
            if indexer is not None:
                context_encoding = biencoder.encode_context(context_input).numpy()
                context_encoding = np.ascontiguousarray(context_encoding)
                scores, indicies = indexer.search_knn(context_encoding, top_k)
            else:
                scores = biencoder.score_candidate(
                    context_input, None, cand_encs=candidate_encoding  # .to(device)
                )
                scores = torch.nn.Softmax(dim=1)(scores)
                scores, indicies = scores.topk(top_k)
                scores = scores.data.cpu().numpy()
                indicies = indicies.data.cpu().numpy()

        labels.extend(label_ids.data.numpy())
        nns.extend(indicies)
        all_scores.extend(scores)
    return labels, nns, all_scores


def _process_crossencoder_dataloader(context_input, label_input, crossencoder_params):
    tensor_data = TensorDataset(context_input, label_input)
    sampler = SequentialSampler(tensor_data)
    dataloader = DataLoader(
        tensor_data, sampler=sampler, batch_size=crossencoder_params['eval_batch_size']
    )
    return dataloader


def _run_crossencoder(crossencoder, dataloader, logger, context_len, \
        device='cuda'):
    crossencoder.model.eval()
    accuracy = 0.0
    crossencoder.to(device)

    res = evaluate(crossencoder, dataloader, device, logger, context_len, \
        zeshel=False, silent=True)
    accuracy = res['normalized_accuracy']
    logits = res['logits']

    if accuracy > -1:
        predictions = np.argsort(logits, axis=1)
    else:
        predictions = []

    return accuracy, predictions, logits


def load_models(args, logger=None):
    # load biencoder model
    if logger:
        logger.info('BLINK | Loading biencoder model')
    with open(args.biencoder_config) as json_file:
        biencoder_params = ast.literal_eval(json_file.read())
        biencoder_params['path_to_model'] = args.biencoder_model
        biencoder_params['bert_base_dir'] = args.bert_base_dir
    biencoder = load_biencoder(biencoder_params)

    crossencoder = None
    crossencoder_params = None
    if True:
        # load crossencoder model
        if logger:
            logger.info('BLINK | Loading crossencoder model')
        with open(args.crossencoder_config) as json_file:
            crossencoder_params = ast.literal_eval(json_file.read())
            crossencoder_params['path_to_model'] = args.crossencoder_model
            crossencoder_params['data_parallel'] = False
            crossencoder_params['bert_base_dir'] = args.bert_base_dir
            crossencoder_params['add_linear'] = True
        crossencoder = load_crossencoder(crossencoder_params)

    # load candidate entities
    if logger:
        logger.info('BLINK: Loading candidate entities')
    (
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    ) = _load_candidates(
        args.entity_catalogue, 
        args.entity_encoding, 
        faiss_index=getattr(args, 'faiss_index', None), 
        index_path=getattr(args, 'index_path' , None),
        logger=logger,
    )

    return (
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer,
    )
