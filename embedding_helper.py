import pandas as pd
from tqdm import tqdm
import numpy as np
from ripser import ripser
import torch
import torch.nn.functional as F
import os
import analysis_functions as topo

def get_embeddings(text, model, tokenizer):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            do_sample=False,
            return_dict_in_generate=True,
            output_hidden_states=True,
            output_scores=True,
        )

    # Full generated answer
    llm_answer = tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True
    )
    
    all_hidden_states = outputs.hidden_states

    # outputs.hidden_states is a tuple containing:
    # index 0: output of the embedding layer
    # index > 1 output of each respective transformer decoder layer
    
    # hidden states from requested layer
    # outputs.hidden_states: generation step -> layer -> tensor
    # each generation step generally has shape:
    #   [batch, 1, hidden_size]
    embeddings = torch.cat(
        [step[13] for step in outputs.hidden_states],
        dim=1
    )
    print(f"Total number of layers captured (embeddings + blocks): {len(all_hidden_states)}")
    print(f"Shape of hidden states at layer {13}: {embeddings.shape}")

    choice_tokens = {}
    for choice in ["A", "B", "C", "D"]:
        token_ids = tokenizer.encode(
            choice,
            add_special_tokens=False
        )

        if len(token_ids) != 1:
            raise ValueError(
                f"{choice!r} is not a single token for this tokenizer: "
                f"{token_ids}"
            )

        choice_tokens[choice] = token_ids[0]

    # get logits
    first_step_logits = outputs.scores[0][0]
    first_step_logprobs = F.log_softmax(first_step_logits, dim=-1)
    choice_logprobs = {choice: first_step_logprobs[token_id].item() for choice, token_id in choice_tokens.items()}

    # determine highest likelihood choice
    likelihood_answer = max(choice_logprobs, key=choice_logprobs.get)

    return all_hidden_states, llm_answer, likelihood_answer


def compute_emb_tda_feat(embeddings):
    """
    Run persistent homology directly on the
    activations!
    """
    embeddings = embeddings.detach().cpu().numpy().astype(np.float64).squeeze(0)

    if embeddings.ndim != 2:
        raise ValueError(
            f"Expected embeddings with shape (N, D), got {embeddings.shape}"
        )

    if not np.all(np.isfinite(embeddings)):
        raise ValueError(
            "Embeddings contain NaN or infinite values."
        )

    diagrams = ripser(
        embeddings,
        distance_matrix=False,
        maxdim=1
    )["dgms"]

    h0 = diagrams[0]

    h1 = (diagrams[1] if len(diagrams) > 1 else np.empty((0, 2)))

    # h0 features
    h0_lifetimes = h0[:, 1] - h0[:, 0]

    finite_h0 = h0_lifetimes[
        np.isfinite(h0_lifetimes)
        & (h0_lifetimes > 1e-10)
    ]

    finite_h0 = np.sort(
        finite_h0
    )[::-1]

    num_h0 = len(finite_h0)

    max_h0 = (
        float(finite_h0[0])
        if num_h0 > 0
        else 0.0
    )

    second_h0 = (
        float(finite_h0[1])
        if num_h0 > 1
        else 0.0
    )

    max_minus_second_h0 = (
        max_h0 - second_h0
        if num_h0 > 1
        else max_h0
    )

    mean_h0 = (
        float(np.mean(finite_h0))
        if num_h0 > 0
        else 0.0
    )


    betti_curve_0_array = topo.compute_betti_curve(
        h0,
        num_bins=50
    )
    
    betti_curve_0 = topo.summarize_betti_curve(
        betti_curve_0_array
    )

    persistence_entropy_0 = topo.persistence_entropy(
        finite_h0
    )

    # h1 features
    h1_lifetimes = (
        h1[:, 1] - h1[:, 0]
        if len(h1) > 0
        else np.array([])
    )

    finite_h1 = h1_lifetimes[
        np.isfinite(h1_lifetimes)
        & (h1_lifetimes > 1e-10)
    ]

    finite_h1 = np.sort(
        finite_h1
    )[::-1]

    num_h1 = len(finite_h1)

    max_h1 = (
        float(finite_h1[0])
        if num_h1 > 0
        else 0.0
    )

    second_h1 = (
        float(finite_h1[1])
        if num_h1 > 1
        else 0.0
    )

    max_minus_second_h1 = (
        max_h1 - second_h1
        if num_h1 > 1
        else max_h1
    )

    mean_h1 = (
        float(np.mean(finite_h1))
        if num_h1 > 0
        else 0.0
    )

    betti_curve_1_array = topo.compute_betti_curve(
        h1,
        num_bins=50
    )

    betti_curve_1 = topo.summarize_betti_curve(
        betti_curve_1_array
    )

    persistence_entropy_1 = topo.persistence_entropy(
        finite_h1
    )

    return [num_h0, max_h0, max_minus_second_h0, mean_h0, betti_curve_0, persistence_entropy_0,
            num_h1, max_h1, max_minus_second_h1, mean_h1, betti_curve_1, persistence_entropy_1,
            diagrams]


# def process_emb(texts, answer, model_id, layers):
#     model, tokenizer = topo.load_model(model_id)
#     data = []
#     dfs = []
#     columns = [
#             "Num_0dim", "Max_0dim", "Max_0dim_Minus_Second", "Mean_0dim", "betti_curve_0", "persistence_entropy_0",
#             "Num_1dim", "Max_1dim", "Max_1dim_Minus_Second", "Mean_1dim", "betti_curve_1", "persistence_entropy_1",
#             "diagrams", "correctness",
#         ]

#     for text_ind, text in enumerate(tqdm(texts)):
#         hidden_states, llm_answer = get_embeddings(text, model, tokenizer)
#         real_answer = answer[text_ind]
#         print("answer: ", real_answer)
#         for layer in range(0, layers + 1):
#             embeddings = torch.cat([step[layer] for step in hidden_states], dim=1)
#             tda_features = compute_emb_tda_feat(embeddings)
#             correctness = topo.evaluate_model(real_answer, llm_answer)
            
#             # Format TDA features properly into a list
#             if isinstance(tda_features, dict):
#                 row_features = list(tda_features.values())
#             elif isinstance(tda_features, (list, tuple)):
#                 row_features = list(tda_features)
#             elif hasattr(tda_features, "tolist"):
#                 row_features = tda_features.tolist()
#             else:
#                 row_features = [tda_features]
    
#             # Combine into EXACTLY ONE row per loop iteration
#             row = row_features + [correctness]
#             data.append(row)
    
#             topo_feat = pd.DataFrame(data, columns=columns)
#             dfs.append(topo_feat)
#     return dfs

def process_emb(texts, answer, model_id, layers):
    model, tokenizer = topo.load_model(model_id)

    columns = [
        "Num_0dim", "Max_0dim", "Max_0dim_Minus_Second", "Mean_0dim",
        "betti_curve_0", "persistence_entropy_0",
        "Num_1dim", "Max_1dim", "Max_1dim_Minus_Second", "Mean_1dim",
        "betti_curve_1", "persistence_entropy_1",
        "diagrams", "correctness", "gen_correctness", "llm_answer",
    ]

    # One data list per layer
    layer_data = [[] for _ in range(layers + 1)]

    for text_ind, text in enumerate(tqdm(texts)):
        hidden_states, llm_answer, likelihood_answer = get_embeddings(text, model, tokenizer)

        real_answer = answer[text_ind]
        print("answer:", real_answer)

        correctness, gen_correctness = topo.evaluate_model(real_answer, llm_answer, likelihood_answer)

        for layer in range(layers + 1):

            embeddings = torch.cat(
                [step[layer] for step in hidden_states],
                dim=1
            )

            tda_features = compute_emb_tda_feat(embeddings)

            # Convert features to a list
            if isinstance(tda_features, dict):
                row_features = list(tda_features.values())
            elif isinstance(tda_features, (list, tuple)):
                row_features = list(tda_features)
            elif hasattr(tda_features, "tolist"):
                row_features = tda_features.tolist()
            else:
                row_features = [tda_features]

            # Exactly one row for this text at this layer
            row = row_features + [correctness, gen_correctness, llm_answer]

            layer_data[layer].append(row)

    # Create one DataFrame per layer
    dfs = [pd.DataFrame(layer_data[layer], columns=columns) for layer in range(layers + 1)]
    return dfs


def emb_top_feat(model, dataset, select=0, create = False, prompting=""):
    # get labels for model
    model_name, model_id, model_short, layers = topo.which_model(model)
    print("layers: ", layers)

    # labels for dataset
    data_name, data_csv = topo.data_label(dataset)
    questions = pd.read_csv(data_csv)

    prompt_name = "prompt" + prompting
 
    # either create or load data
    if create:
        feats_sen = questions[prompt_name]
        answer = []
        if data_name == "hellaswag":
            answer = questions["label"]
        elif data_name == "mcqa":
            answer = questions["alternative"]
        else:
            answer = questions["answer"]
        feats_tda = process_emb(feats_sen, answer, model_id, layers)
        for layer in range(0, layers + 1):
            tda_path = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpret/{model_short}/{model_short}_{data_name}_tda_{layer}{prompting}.csv")
            
            feats_tda[layer].to_csv(tda_path, index=False)
            print(f"Added questions from {data_name} for {model_short}! (layer {layer})")
   
    else:
        tda_path = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpret/{model_short}/{model_short}_{data_name}_tda_{select}.csv")
        feats_tda = pd.read_csv(tda_path)

    return feats_tda