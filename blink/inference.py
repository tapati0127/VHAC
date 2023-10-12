from typing import List, Tuple

from scipy.special import softmax

from src.model.blink.main_dense import _process_biencoder_dataloader, \
    _run_biencoder, _process_crossencoder_dataloader, _run_crossencoder
from src.model.blink.crossencoder.data_process import prepare_crossencoder_data
from src.model.blink.crossencoder.train_cross import modify


def inference(
        args,
        logger,
        biencoder,
        biencoder_params,
        crossencoder,
        crossencoder_params,
        candidate_encoding,
        title2id,
        id2title,
        id2text,
        wikipedia_id2local_id,
        faiss_indexer=None,
        test_data=None
        ) -> Tuple[List, List]:
    biencode_dataloader = _process_biencoder_dataloader(
        test_data, biencoder.tokenizer, biencoder_params
    )

    labels, nns, _ = _run_biencoder(
        biencoder, biencode_dataloader, candidate_encoding, \
        args.top_candidates, faiss_indexer
    )

    context_input, candidate_input, label_input = prepare_crossencoder_data(
        crossencoder.tokenizer, test_data, labels, nns, id2title, id2text, 
        keep_all=True
    )

    context_input_2 = modify(
        context_input, 
        candidate_input, 
        crossencoder_params['max_context_length'], 
        crossencoder_params['max_cand_length'], 
        crossencoder_params["max_seq_length"]
    )

    crossencode_dataloader = _process_crossencoder_dataloader(
                context_input_2, label_input, crossencoder_params
    )

    _, index_array, unsorted_scores = _run_crossencoder(
        crossencoder,
        crossencode_dataloader,
        logger,
        context_len=crossencoder_params["max_context_length"],
        device=crossencoder.device
    )

    score = softmax(unsorted_scores[0])[index_array[0][-1]]
    if score < args.find_thresh:
        entity_id = None
        entity_title = None
    else:
        entity_id = nns[0][index_array[0][-1]]
        entity_title = id2title[entity_id]
    return entity_id, entity_title
