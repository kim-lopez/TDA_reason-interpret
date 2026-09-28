import pandas as pd
from tqdm import tqdm
import numpy as np
from ripser import ripser
import torch
import os
import analysis_functions as topo

def get_embeddings(text, model, tokenizer, layer):
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
        [step[layer] for step in outputs.hidden_states],
        dim=1
    )
    
    print(f"Total number of layers captured (embeddings + blocks): {len(all_hidden_states)}")
    print(f"Shape of hidden states at layer {layer}: {embeddings.shape}")

    return embeddings, llm_answer


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


def process_emb(texts, answer, model_id, layer):
    model, tokenizer = topo.load_model(model_id)
    data = []

    for text_ind, text in enumerate(tqdm(texts)):
        embeddings, llm_answer = get_embeddings(text, model, tokenizer, layer)
        real_answer = answer[text_ind]
        print("answer: ", real_answer)

        tda_features = compute_emb_tda_feat(embeddings)
        correctness = topo.evaluate_model(real_answer, llm_answer)
        
        # Format TDA features properly into a list
        if isinstance(tda_features, dict):
            row_features = list(tda_features.values())
        elif isinstance(tda_features, (list, tuple)):
            row_features = list(tda_features)
        elif hasattr(tda_features, "tolist"):
            row_features = tda_features.tolist()
        else:
            row_features = [tda_features]

        # Combine into EXACTLY ONE row per loop iteration
        row = row_features + [correctness]
        data.append(row)

    columns = [
        "Num_0dim", "Max_0dim", "Max_0dim_Minus_Second", "Mean_0dim", "betti_curve_0", "persistence_entropy_0",
        "Num_1dim", "Max_1dim", "Max_1dim_Minus_Second", "Mean_1dim", "betti_curve_1", "persistence_entropy_1",
        "diagrams", "correctness",
    ]

    return pd.DataFrame(data, columns=columns)


def emb_top_feat(model, dataset, layer, create = False):
    # get labels for model
    model_name, model_id, model_short  = topo.which_model(model)

    # labels for dataset
    data_name, data_csv = topo.data_label(dataset)
    questions = pd.read_csv(data_csv)

    tda_path = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpret/{model_short}/{model_short}_{data_name}_tda_{layer}.csv")
 
    # either create or load data
    if create:
        feats_sen = questions["prompt"]
        answer = []
        if data_name == "hellaswag":
            answer = questions["label"]
        elif data_name == "mcqa":
            answer = questions["alternative"]
        else:
            answer = questions["answer"]
        feats_tda = process_emb(feats_sen, answer, model_id, layer)
        feats_tda.to_csv(tda_path, index=False)
        print(f"Added questions from {data_name} for {model_short}! (layer {layer})")
   
    else:
        feats_tda = pd.read_csv(tda_path)

    return feats_tda