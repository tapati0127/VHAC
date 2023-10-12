# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import json
import logging
import torch
from tqdm import tqdm

from blink.biencoder.zeshel_utils import WORLDS, Stats


def get_topk_predictions(
    reranker,
    train_dataloader,
    candidate_pool,
    cand_encode_list,
    silent,
    logger,
    top_k=10,
    is_zeshel=False,
    save_predictions=False,
):
    reranker.model.eval()
    device = reranker.device
    logger.info("Getting top %d predictions." % top_k)
    if silent:
        iter_ = train_dataloader
    else:
        iter_ = tqdm(train_dataloader)

    nn_context = []
    nn_candidates = []
    nn_labels = []
    nn_worlds = []
    stats = {}

    if is_zeshel:
        world_size = len(WORLDS)
    else:
        # only one domain
        world_size = 1
        candidate_pool = [candidate_pool]
        cand_encode_list = [cand_encode_list]

    logger.info("World size : %d" % world_size)

    for i in range(world_size):
        stats[i] = Stats(top_k)
    
    # oid = 0
    for step, batch in enumerate(iter_):
        batch = tuple(t.to(device) for t in batch)
        if len(batch) == 4:
            context_input, _, srcs, label_ids = batch
        elif len(batch) == 3:
            context_input, _, label_ids = batch
        else:
            assert len(batch) == 3 or len(batch) == 4, 'Error processing data mention'
        scores = reranker.score_candidate(
            context_input, 
            None, 
            cand_encs=cand_encode_list[0].to(device)
        )
        values, indicies = scores.topk(top_k)
        src = 0
        for i in range(context_input.size(0)):
            inds = indicies[i].cpu()

            pointer = -1
            for j in range(top_k):
                if inds[j].item() == label_ids[i].item():
                    pointer = j
                    break
            stats[src].add(pointer)

            if pointer == -1:
                continue

            if not save_predictions:
                continue

            # add examples in new_data
            cur_candidates = candidate_pool[src][inds]
            nn_context.append(context_input[i].cpu().tolist())
            nn_candidates.append(cur_candidates.cpu().tolist())
            nn_labels.append(pointer)
            nn_worlds.append(src)

    res = Stats(top_k)
    for src in range(world_size):
        if stats[src].cnt == 0:
            continue
        if is_zeshel:
            logger.info("In world " + WORLDS[src])
        output = stats[src].output()
        logger.info(output)
        res.extend(stats[src])

    logger.info(res.output())

    nn_context = torch.LongTensor(nn_context)
    nn_candidates = torch.LongTensor(nn_candidates)
    nn_labels = torch.LongTensor(nn_labels)
    nn_data = {
        'context_vecs': nn_context,
        'candidate_vecs': nn_candidates,
        'labels': nn_labels,
    }

    if is_zeshel:
        nn_data["worlds"] = torch.LongTensor(nn_worlds)
    
    return nn_data
