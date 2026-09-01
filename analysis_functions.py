## == IMPORTS == ##
# for attention extraction + analysis
import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel, AutoModelForSequenceClassification
import numpy as np
import pandas as pd
from ripser import ripser
from tqdm import tqdm
import math
import networkx as nx
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from prettytable import PrettyTable
import matplotlib.pyplot as plt
import re
from persim import wasserstein_matching

# to evaluate model
from lm_eval.tasks import TaskManager

## == LOAD IN MODELS == ##
def load_model(model_id = "meta-llama/Llama-3.1-8B-Instruct", device = "cuda"):
    """
    Load model and tokenizer, handling different architectures.
    """
    from transformers import AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        # For causal LM models (Llama, Qwen, GPT, etc.)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            output_attentions=True,
            attn_implementation="eager",
            dtype=torch.float16 if device == "cuda" else torch.float32,
            device_map="auto" if device == "cuda" else None
        )
    except Exception:
        try:
            # for encoder-only models (BERT, RoBERTa, etc.)
            model = AutoModel.from_pretrained(
                model_id, 
                output_attentions=True,
                dtype=torch.float16 if device == "cuda" else torch.float32
            )
        except Exception:
            # for seq2seq models (BART, T5, etc.)
            model = AutoModelForSequenceClassification.from_pretrained(
                model_id,
                output_attentions=True,
                dtype=torch.float16 if device == "cuda" else torch.float32
            )
    
    model.to(device)
    model.eval()
    
    return model, tokenizer

# get information from dataset
def data_label(dataset):
    """Extracts information from dataset strings"""
    name = dataset[0]
    csv = dataset[1]
    return name, csv

# get information from model
def which_model(model):
    """Extracts information from model strings"""
    name = model[0]
    hf_id = model[1]
    short = model[2]
    return name, hf_id, short

# check if answer is correct
def evaluate_model(real_answer, llm_answer):
    correct_index = int(real_answer)
    match = re.search(r"Answer:\s*([A-D])|([A-D])\s*$", llm_answer.strip(), re.IGNORECASE)
    
    if match:
        extracted = (match.group(1) or match.group(2)).upper()
        predicted_index = ord(extracted) - ord('A')
    else:
        cleaned = llm_answer.strip()
        if cleaned in ["0", "1", "2", "3"]:
            predicted_index = int(cleaned)
        else:
            # raise ValueError(f"Expected A/B/C/D or 0/1/2/3, got: {llm_answer!r}")
            return False

    correctness = (predicted_index == correct_index)
    
    return correctness


## == TDA HELPERS == ##
## function to assist with finding highest h0/h1 TDA feature
def find_highest_finite_value_comprehension(data):
    finite_values = [
        float(x) for x in data
        if np.isfinite(x) and x > 0
    ]
    return max(finite_values) if finite_values else 0.0
    
# def find_highest_finite_value_comprehension(data):
#     """Finds the highest value in a list, ignoring inf values, using list comprehension."""
#     finite_values = [x for x in data if np.isfinite(x)]
#     return max(finite_values) if finite_values else 0.0

## function to assist with finding second highest h0/h1 TDA feature
def get_second_value_ignoring_inf(data):
    finite_values = sorted(
        [
            float(x) for x in data
            if np.isfinite(x) and x > 0
        ],
        reverse=True
    )
    if len(finite_values) >= 2:
        return finite_values[1]
    return 0.0


## == EXTRACT ATTENTION MAP == #
# extract attention from model
# def get_attention(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(model.device)
#     with torch.no_grad():
#         outputs = model.generate(
#             **inputs,
#             max_new_tokens=200,
#             pad_token_id=tokenizer.eos_token_id,
#             return_dict_in_generate=True,
#             output_attentions=True
#         )
#     answer = tokenizer.decode(outputs.sequences[0], skip_special_tokens=True)
    
#     # get average attention across heads
#     attention_matrices = torch.stack(outputs.attentions[0]).mean(dim=(1,3)).squeeze(0).cpu().numpy()
#     avg_attention = np.mean(attention_matrices, axis=0)
#     return avg_attention, answer

def get_attention(text, model, tokenizer):

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=512
    ).to(model.device)

    prompt_length = inputs["input_ids"].shape[1]

    captured_attention = None

    # ---------------------------------------------
    # Hook the last transformer layer
    # ---------------------------------------------

    attention_layer = model.model.layers[-1].self_attn

    def attention_hook(module, module_inputs, module_outputs):

        nonlocal captured_attention

        # Llama attention output normally contains:
        #
        #   0 = attention output
        #   1 = attention weights
        #
        if (
            isinstance(module_outputs, tuple)
            and len(module_outputs) > 1
            and module_outputs[1] is not None
        ):

            attention = module_outputs[1]

            # We only want the FIRST forward pass,
            # which contains the complete original prompt.
            if attention.shape[-2] == prompt_length:

                captured_attention = attention.detach()

    handle = attention_layer.register_forward_hook(
        attention_hook
    )

    # ---------------------------------------------
    # Generate answer
    # ---------------------------------------------

    with torch.no_grad():

        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            pad_token_id=tokenizer.eos_token_id,
            return_dict_in_generate=True,
        )

    # Remove hook
    handle.remove()

    # ---------------------------------------------
    # Decode answer
    # ---------------------------------------------

    answer = tokenizer.decode(
        outputs.sequences[0],
        skip_special_tokens=True
    )

    # ---------------------------------------------
    # Validate attention
    # ---------------------------------------------

    if captured_attention is None:
        raise RuntimeError(
            "Could not capture original prompt attention."
        )

    # [batch, heads, prompt, prompt]
    print(
        "Raw prompt attention shape:",
        captured_attention.shape
    )

    # Average attention heads
    attention_matrix = captured_attention.mean(
        dim=1
    )[0]

    # [prompt, prompt]
    attention_matrix = (
        attention_matrix
        .float()
        .cpu()
        .numpy()
    )

    print(
        "Final prompt attention shape:",
        attention_matrix.shape
    )

    return attention_matrix, answer


def build_graph(attention_matrix, threshold=0.0):
    """
    Construct an attention-derived dissimilarity matrix.

    Tokens are the points.
    Attention determines pairwise similarity.
    Strong attention -> small distance.
    Weak attention -> large distance.

    The threshold argument is retained for compatibility but is
    not used to remove edges before persistent homology.
    """

    attention = np.asarray(attention_matrix, dtype=np.float64)

    attention = np.nan_to_num(
        attention,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # Make attention undirected.
    attention = (attention + attention.T) / 2.0

    # Normalize.
    max_attention = np.max(attention)

    if max_attention > 0:
        attention = attention / max_attention

    # Attention similarity -> distance.
    distance_matrix = 1.0 - attention

    distance_matrix = np.clip(
        distance_matrix,
        0.0,
        1.0
    )

    np.fill_diagonal(distance_matrix, 0.0)

    return distance_matrix

    
def attention_to_distance(attention_matrix):
    """
    Convert attention similarity matrix into a distance matrix.

    Strong attention --> small distance
    Weak attention   --> large distance
    """

    A = np.asarray(attention_matrix, dtype=np.float64)

    A = np.nan_to_num(A, nan=0.0, posinf=0.0, neginf=0.0)

    # Symmetrize attention
    A = (A + A.T) / 2.0

    # Attention similarity -> distance
    D = 1.0 - A

    # Distance from a token to itself = 0
    np.fill_diagonal(D, 0.0)

    return D



# aquire TDA features from model
# def compute_tda_features(graph):
#     adjacency_matrix = nx.to_numpy_array(graph)
#     diagrams = ripser(adjacency_matrix, distance_matrix=True, maxdim=1)['dgms']

#     h0 = diagrams[0]
#     h1 = diagrams[1] if len(diagrams) > 1 else np.array([])

#     num_h0 = np.sum(np.isfinite(h0[:, 1]))
#     highest_h0 = find_highest_finite_value_comprehension(h0[:, 1] - h0[:, 0]) if num_h0 > 0 else 0
#     Second_highest_h0 = get_second_value_ignoring_inf(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else 0
#     highest_minus_second_h0 = highest_h0 - Second_highest_h0 if num_h0 > 1 else 0

#     # Replace inf values with 0
#     h0[np.isinf(h0)] = 0
#     mean_h0 = np.mean(h0) if num_h0 > 0 else 0
#     # print("mean h0: ", mean_h0)


#     num_h1 = np.count_nonzero(np.round(h1))
#     highest_h1 = find_highest_finite_value_comprehension(h1[:, 1] - h1[:, 0]) if num_h1 > 0 else 0
#     second_highest_h1 = get_second_value_ignoring_inf(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else 0
#     highest_minus_second_h1 = highest_h1 - second_highest_h1 if num_h1 > 1 else 0

#     # Replace inf values with 0
#     h1[np.isinf(h1)] = 0
#     mean_h1 = np.mean(h1) if num_h1 > 0 else 0
#     # print("mean h1: ", mean_h1)

#     # h0_persistences = np.sort(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else np.array([0])

#     # h1_persistences = np.sort(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else np.array([0])
#     h0_persistences = (
#                         np.sort(h0[:, 1] - h0[:, 0])
#                         if num_h0 > 0
#                         else np.array([])
#                       )
    
#     h1_persistences = (
#                         np.sort(h1[:, 1] - h1[:, 0])
#                         if num_h1 > 0
#                         else np.array([])
#                     )


#     # Additional TDA features for linguistic correlation
#     sum_persistence_0 = np.sum(h0_persistences) if len(h0_persistences) > 0 else 0
#     sum_persistence_1 = np.sum(h1_persistences) if len(h1_persistences) > 0 else 0
#     persistence_entropy_0 = -np.sum(h0_persistences * np.log(h0_persistences + 1e-10)) if len(h0_persistences) > 0 else 0
#     persistence_entropy_1 = -np.sum(h1_persistences * np.log(h1_persistences + 1e-10)) if len(h1_persistences) > 0 else 0
#     betti_curve_0 = len(h0_persistences)
#     betti_curve_1 = len(h1_persistences)


#     return [num_h0, highest_h0, highest_minus_second_h0, mean_h0, betti_curve_0, persistence_entropy_0,
#             num_h1, highest_h1, highest_minus_second_h1, mean_h1, betti_curve_1, persistence_entropy_1]

def _persistence_entropy(persistences):

    persistences = np.asarray(
        persistences,
        dtype=np.float64
    )

    persistences = persistences[
        persistences > 0
    ]

    if len(persistences) == 0:
        return 0.0

    total = np.sum(persistences)

    if total <= 0:
        return 0.0

    probabilities = persistences / total

    return float(
        -np.sum(
            probabilities *
            np.log(probabilities)
        )
    )
    
def compute_betti_curve(diagram, num_bins=50):
    """
    Compute the Betti curve beta_k(t) from a persistence diagram.

    beta_k(t) = number of persistence intervals alive at filtration t.
    """

    if diagram is None or len(diagram) == 0:
        return np.zeros(num_bins, dtype=np.float64)

    diagram = np.asarray(diagram, dtype=np.float64)

    births = diagram[:, 0]
    deaths = diagram[:, 1]

    # Keep finite persistence intervals
    mask = (
        np.isfinite(births) &
        np.isfinite(deaths) &
        (deaths > births)
    )

    births = births[mask]
    deaths = deaths[mask]

    if len(births) == 0:
        return np.zeros(num_bins, dtype=np.float64)

    # Common filtration grid
    min_value = np.min(births)
    max_value = np.max(deaths)

    if max_value <= min_value:
        return np.zeros(num_bins, dtype=np.float64)

    grid = np.linspace(
        min_value,
        max_value,
        num_bins
    )

    betti_curve = np.zeros(
        num_bins,
        dtype=np.float64
    )

    for i, t in enumerate(grid):
        betti_curve[i] = np.sum(
            (births <= t) &
            (t < deaths)
        )

    return betti_curve


def summarize_betti_curve(betti_curve):
    """
    Convert the Betti curve into one scalar.

    Uses the mean Betti number across the filtration.
    """

    betti_curve = np.asarray(
        betti_curve,
        dtype=np.float64
    )

    if betti_curve.size == 0:
        return 0.0

    return float(np.mean(betti_curve))

    
def compute_tda_features(distance_matrix):
    """
    Run persistent homology directly on the
    attention-derived distance matrix.
    """

    distance_matrix = np.asarray(
        distance_matrix,
        dtype=np.float64
    )

    # Numerical cleanup
    distance_matrix = np.nan_to_num(
        distance_matrix,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # Ensure symmetric distance matrix
    distance_matrix = (
        distance_matrix + distance_matrix.T
    ) / 2.0

    np.fill_diagonal(
        distance_matrix,
        0.0
    )

    # =====================================================
    # ATTENTION DISTANCE MATRIX --> RIPSER
    # =====================================================

    diagrams = ripser(
        distance_matrix,
        distance_matrix=True,
        maxdim=1
    )["dgms"]

    h0 = diagrams[0]

    h1 = (
        diagrams[1]
        if len(diagrams) > 1
        else np.empty((0, 2))
    )

    # =====================================================
    # H0
    # =====================================================

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

    betti_curve_0_array = compute_betti_curve(
        h0,
        num_bins=50
    )

    betti_curve_0 = summarize_betti_curve(
        betti_curve_0_array
    )

    persistence_entropy_0 = _persistence_entropy(
        finite_h0
    )

    # =====================================================
    # H1
    # =====================================================

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

    betti_curve_1_array = compute_betti_curve(
        h1,
        num_bins=50
    )

    betti_curve_1 = summarize_betti_curve(
        betti_curve_1_array
    )

    persistence_entropy_1 = _persistence_entropy(
        finite_h1
    )

    return [num_h0, max_h0, max_minus_second_h0, mean_h0, betti_curve_0, persistence_entropy_0,
            num_h1, max_h1, max_minus_second_h1, mean_h1, betti_curve_1, persistence_entropy_1,
            diagrams]


# # analyzes text from model
# def process_texts(texts, text_cat, model_id):
#     model, tokenizer = load_model(model_id)
#     data = []
#     index = 0
#     for text in tqdm(texts):
#         attention_matrix, answer = get_attention(text, model, tokenizer)
#         graph = build_graph(attention_matrix)
#         tda_features = compute_tda_features(graph)
#         correctness = evaluate_model(text_cat, index, answer)
        
#         if isinstance(tda_features, dict):
#             row = list(tda_features.values()) + [correctness]
#         elif isinstance(tda_features, (list, tuple)):
#             row = list(tda_features) + [correctness]
#         else:
#             # If tda_features is a numpy arraySSS
#             row = tda_features.tolist() + [correctness]

#         data.append(row)
        
#         index += 1

#     columns = ["Num_0dim", "Max_0dim", "Max_0dim_Minus_Second", "Mean_0dim", "betti_curve_0", "persistence_entropy_0",
#                "Num_1dim", "Max_1dim", "Max_1dim_Minus_Second", "Mean_1dim", "betti_curve_1", "persistence_entropy_1",
#                "correctness"]
#     return pd.DataFrame(data, columns=columns)

def process_texts(texts, answer, model_id):
    model, tokenizer = load_model(model_id)
    data = []

    for index, text in enumerate(tqdm(texts)):
        attention_matrix, llm_answer = get_attention(text, model, tokenizer)
        real_answer = answer[index]
        print("answer: ", real_answer)

        # DEBUG CHECK: Ensure attention matrix is non-zero
        if (
            attention_matrix is None
            or np.all(attention_matrix == 0)
            or np.isnan(attention_matrix).all()
        ):
            print(f"[Warning] Empty attention matrix at index {index}")

        # print("Attention shape:", attention_matrix.shape)
        # print("Attention min:", attention_matrix.min())
        # print("Attention max:", attention_matrix.max())
        # print("Attention mean:", attention_matrix.mean())

        graph = attention_to_distance(attention_matrix)
        
        # print("Distance matrix:", graph.shape)
        # print("Distance min:", graph.min())
        # print("Distance max:", graph.max())

        tda_features = compute_tda_features(graph)
        correctness = evaluate_model(real_answer, llm_answer)
        
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


## == EXTRACT TDA FEATURES FROM MODEL == ##
# extract top feats from models
def get_top_feat(model, dataset, create = False):
    # get labels for model
    model_name, model_id, model_short  = which_model(model)

    # labels for dataset
    data_name, data_csv = data_label(dataset)
    questions = pd.read_csv(data_csv)
    
    tda_path = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpret/{model_short}/{model_short}_{data_name}_tda.csv")

    # either create or load data
    if create:
        feats_sen = questions["prompt"]
        answer = []
        if data_name == "hellaswag":
            answer = questions["label"]
        else:
            answer = questions["answer"]
        feats_tda = process_texts(feats_sen, answer, model_id)
        feats_tda.to_csv(tda_path, index=False)
        print(f"Added questions from {data_name} for {model_short}!")
   
    else:
        feats_tda = pd.read_csv(tda_path)

    return feats_tda

# sample from topo features
def rand_sample(model, dataset, random_state=8):
    feats_tda = get_top_feat(model, dataset)
    
    # isolate correct/incorrect answers
    correct_feats = feats_tda[feats_tda["correctness"] == 1]
    incorrect_feats = feats_tda[feats_tda["correctness"] == 0]

    correct_feats = correct_feats.sample(n=100, random_state)
    incorrect_feats = incorrect_feats.sample(n=100, random_state)

    return correct_feats, incorrect_feats

# analyze the h0 and h1 features
def analyze_feats(model, dataset):
    correct_feats, incorrect_feats = rand_sample(model, dataset)


    avg_correct_0dim = [correct_feats["Num_0dim"].mean(), correct_feats["Max_0dim"].mean(),
                        correct_feats["Max_0dim_Minus_Second"].mean(), correct_feats["Mean_0dim"].mean(),
                        correct_feats["betti_curve_0"].mean(), correct_feats["persistence_entropy_0"].mean()]
    std_correct_0dim = [correct_feats["Num_0dim"].sem(), correct_feats["Max_0dim"].sem(),
                        correct_feats["Max_0dim_Minus_Second"].sem(), correct_feats["Mean_0dim"].sem(),
                        correct_feats["betti_curve_0"].sem(), correct_feats["persistence_entropy_0"].sem()]
    
    avg_correct_1dim = [correct_feats["Num_1dim"].mean(), correct_feats["Max_1dim"].mean(),
                        correct_feats["Max_1dim_Minus_Second"].mean(), correct_feats["Mean_1dim"].mean(),
                        correct_feats["betti_curve_1"].mean(), correct_feats["persistence_entropy_1"].mean()]
    std_correct_1dim = [correct_feats["Num_1dim"].sem(), correct_feats["Max_1dim"].sem(),
                        correct_feats["Max_1dim_Minus_Second"].sem(), correct_feats["Mean_1dim"].sem(),
                        correct_feats["betti_curve_1"].sem(), correct_feats["persistence_entropy_1"].sem()]
    
    avg_incorrect_0dim = [incorrect_feats["Num_0dim"].mean(), incorrect_feats["Max_0dim"].mean(),
                        incorrect_feats["Max_0dim_Minus_Second"].mean(), incorrect_feats["Mean_0dim"].mean(),
                        incorrect_feats["betti_curve_0"].mean(), incorrect_feats["persistence_entropy_0"].mean()]
    std_incorrect_0dim = [incorrect_feats["Num_0dim"].sem(), incorrect_feats["Max_0dim"].sem(),
                        incorrect_feats["Max_0dim_Minus_Second"].sem(), incorrect_feats["Mean_0dim"].sem(),
                        incorrect_feats["betti_curve_0"].sem(), incorrect_feats["persistence_entropy_0"].sem()]
    
    avg_incorrect_1dim = [incorrect_feats["Num_1dim"].mean(), incorrect_feats["Max_1dim"].mean(),
                        incorrect_feats["Max_1dim_Minus_Second"].mean(), incorrect_feats["Mean_1dim"].mean(),
                        incorrect_feats["betti_curve_1"].mean(), incorrect_feats["persistence_entropy_1"].mean()]
    std_incorrect_1dim = [incorrect_feats["Num_1dim"].sem(), incorrect_feats["Max_1dim"].sem(),
                        incorrect_feats["Max_1dim_Minus_Second"].sem(), incorrect_feats["Mean_1dim"].sem(),
                        incorrect_feats["betti_curve_1"].sem(), incorrect_feats["persistence_entropy_1"].sem()]
    
    avg_diff_0dim = [corr0 - incorr0 for corr0, incorr0 in zip(avg_correct_0dim, avg_incorrect_0dim)]
    avg_diff_1dim = [corr1 - incorr1 for corr1, incorr1 in zip(avg_correct_1dim, avg_incorrect_1dim)]
    
    # display info
    table = PrettyTable()
    feature_names = ["num_feat", "max_feat", "max_feat_minus_second", "mean_feat", "betti_curve", "persistence_entropy"]
    filter_names = ["max_feat", "max_feat_minus_second", "mean_feat"]
    table.field_names = ["label"] + feature_names
    table.add_row(["correct_0dim"] + avg_correct_0dim)
    table.add_row(["correct_0dim_stdv"] + std_correct_0dim)
    
    table.add_row(["correct_1dim"] + avg_correct_1dim)
    table.add_row(["correct_1dim_std"] + std_correct_1dim)
    
    table.add_row(["incorrect_0dim"] + avg_incorrect_0dim)
    table.add_row(["incorrect_0dim_std"] + std_incorrect_0dim)
    
    table.add_row(["incorrect_1dim"] + avg_incorrect_1dim)
    table.add_row(["incorrect_1dim"] + std_incorrect_1dim)
    
    table.add_row(["diff_0dim"] + avg_diff_0dim)
    table.add_row(["diff_1dim"] + avg_diff_1dim)

    print(table)

    # Plot
    x = np.arange(len(filter_names))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # -------------------------
    # 0-dimensional features
    # -------------------------
    axes[0].bar(
        x - width / 2,
        avg_correct_0dim[1:4],
        width,
        yerr=std_correct_0dim[1:4],
        capsize=4,
        label="Correct",
        color="steelblue",
        alpha=0.85
    )

    axes[0].bar(
        x + width / 2,
        avg_incorrect_0dim[1:4],
        width,
        yerr=std_incorrect_0dim[1:4],
        capsize=4,
        label="Incorrect",
        color="tomato",
        alpha=0.85
    )

    axes[0].set_title("0D TDA Features")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(filter_names, rotation=35, ha="right")
    axes[0].set_ylabel("Mean Feature Value")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.25)

    # -------------------------
    # 1-dimensional features
    # -------------------------
    x = np.arange(len(feature_names))
    
    axes[1].bar(
        x - width / 2,
        avg_correct_1dim,
        width,
        yerr=std_correct_1dim,
        capsize=4,
        label="Correct",
        color="steelblue",
        alpha=0.85
    )

    axes[1].bar(
        x + width / 2,
        avg_incorrect_1dim,
        width,
        yerr=std_incorrect_1dim,
        capsize=4,
        label="Incorrect",
        color="tomato",
        alpha=0.85
    )

    axes[1].set_title("1D TDA Features")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(feature_names, rotation=35, ha="right")
    axes[1].set_ylabel("Mean Feature Value")
    axes[1].legend()
    axes[1].grid(axis="y", alpha=0.25)

    plt.tight_layout()
    plt.show()

    return 0

# plot barcode graph
def plot_barcode(model, dataset):
    correct_feats, incorrect_feats = rand_sample(model, dataset)
    
    diagram_corr_list = correct_feats["diagrams"].to_list()
    diagram_incorr_list = incorrect_feats["diagrams"].to_list()

    diagram_corr = np.mean(np.stack(diagram_corr_list, axis=0), axis=0)
    diagram_incorr = np.mean(np.stack(diagram_incorr_list, axis=0), axis=0)
    
    fig, ax = plt.subplots(figsize=(12, 6))

    colors = ["tab:blue", "tab:orange"]
    x1 = 0
    x2 = 0

    for dim, dgm in enumerate(diagram_corr):
        for birth, death in dgm:
            if np.isinf(death):
                # Choose a finite endpoint for visualization
                death = max(
                    np.max(dgm[np.isfinite(dgm[:, 1]), 1]),
                    birth + 1
                )

            ax.plot(
                [birth, death],
                [x1, x1],
                color=colors[0],
                linewidth=4
            )

            x1 += 1
            
    for dim, dgm in enumerate(diagram_incorr):
        for birth, death in dgm:
            if np.isinf(death):
                # Choose a finite endpoint for visualization
                death = max(
                    np.max(dgm[np.isfinite(dgm[:, 1]), 1]),
                    birth + 1
                )

            ax.plot(
                [birth, death],
                [x2, x2],
                color=colors[1],
                linewidth=4
            )

            x2 += 1

    ax.set_xlabel("Topological feature")
    ax.set_ylabel("Filtration value")
    ax.set_title("Persistent Homology Barcode")

    plt.tight_layout()
    plt.show()


## == old functions for extracting attention + embeddings == ##
# def get_attention(text, model, tokenizer):
#     inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512).to(model.device)
#     with torch.no_grad():
#         outputs = model(**inputs, output_attentions=True)
#     attention_matrices = torch.stack(outputs.attentions).mean(dim=0).squeeze(0).cpu().numpy()
#     return np.mean(attention_matrices, axis=0)  # Averaging across heads

# def get_embeddings(text, model, tokenizer, last_layer = False):
#     # get the intial tokenization (layer 0 embeddings)
#     inputs = tokenizer(text=text, truncation=True, return_tensors="pt").to(model.device)

#     # get hidden state information
#     with torch.no_grad():
#         outputs = model(**inputs, output_hidden_states=True)
#     index = 0
#     if last_layer:
#         index = -1
#     embeddings = outputs.hidden_states[index]

#     return embeddings

# def build_graph_attention(attention_matrix, threshold=0.1):
#     graph = nx.Graph()
#     num_nodes = attention_matrix.shape[0]

#     for i in range(num_nodes):
#         for j in range(i + 1, num_nodes):
#             if attention_matrix[i, j] > threshold:
#                 graph.add_edge(i, j, weight=attention_matrix[i, j])

#     if graph.number_of_nodes() == 0:
#         print("transposed matrix")
#         for i in range(num_nodes):
#             for j in range(i + 1, num_nodes):
#                 if attention_matrix[j, i] > threshold:
#                     graph.add_edge(i, j, weight=attention_matrix[i, j])
    
#     adjacency_matrix = nx.to_numpy_array(graph)
#     return adjacency_matrix

# def build_graph_embeddings(embeddings):
#     # ensure embeddings are numpy array
#     embeddings = np.asarray(embeddings)

#     # standardize embeddings
#     if embeddings.ndim > 2:
#         embeddings = embeddings.reshape(-1, embeddings.shape[-1])

#     # check if empty
#     if embeddings.shape[1] == 0:
#         raise ValueError(f"embeddings have shape: {embeddings.shape}")

#     # remove values if they are NaN
#     if np.isnan(embeddings).any():
#         imputer = SimpleImputer(strategy='mean')
#         embeddings = imputer.fit_transform(embeddings)
    
#     # fit + transform PCA with 2 components
#     n_components = min(2, embeddings.shape[1])
#     graph_embeddings = PCA(n_components=n_components).fit_transform(embeddings)

#     return graph_embeddings

# def compute_tda_features(graph):
#     diagrams = ripser(graph, maxdim=1)['dgms']

#     h0 = diagrams[0]
#     h1 = diagrams[1] if len(diagrams) > 1 else np.array([])

#     num_h0 = np.count_nonzero(np.round(h0))
#     highest_h0 = find_highest_finite_value_comprehension(h0[:, 1] - h0[:, 0]) if num_h0 > 0 else 0
#     Second_highest_h0 = get_second_value_ignoring_inf(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else 0
#     highest_minus_second_h0 = highest_h0 - Second_highest_h0 if num_h0 > 1 else 0

#     # Replace inf values with 0
#     h0[np.isinf(h0)] = 0
#     mean_h0 = np.mean(h0) if num_h0 > 0 else 0

#     num_h1 = np.count_nonzero(np.round(h1))
#     highest_h1 = find_highest_finite_value_comprehension(h1[:, 1] - h1[:, 0]) if num_h1 > 0 else 0
#     second_highest_h1 = get_second_value_ignoring_inf(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else 0
    
#     # set none values to 0 to avoid error
#     highest_h1 = highest_h1 if highest_h1 is not None else 0
#     second_highest_h1 = second_highest_h1 if second_highest_h1 is not None else 0
#     highest_minus_second_h1 = highest_h1 - second_highest_h1 if num_h1 > 1 else 0

#     # Replace inf values with 0
#     h1[np.isinf(h1)] = 0
#     mean_h1 = np.mean(h1) if num_h1 > 0 else 0
#     # print("mean h1: ", mean_h1)

#     h0_persistences = np.sort(h0[:, 1] - h0[:, 0]) if num_h0 > 1 else np.array([0])

#     h1_persistences = np.sort(h1[:, 1] - h1[:, 0]) if num_h1 > 1 else np.array([0])

#     # Additional TDA features for linguistic correlation
#     sum_persistence_0 = np.sum(h0_persistences) if len(h0_persistences) > 0 else 0
#     sum_persistence_1 = np.sum(h1_persistences) if len(h1_persistences) > 0 else 0
#     persistence_entropy_0 = -np.sum(h0_persistences * np.log(h0_persistences + 1e-10)) if len(h0_persistences) > 0 else 0
#     persistence_entropy_1 = -np.sum(h1_persistences * np.log(h1_persistences + 1e-10)) if len(h1_persistences) > 0 else 0
#     betti_curve_0 = len(h0_persistences)
#     betti_curve_1 = len(h1_persistences)


#     return [num_h0, highest_h0, highest_minus_second_h0, mean_h0, betti_curve_0, persistence_entropy_0,
#             num_h1, highest_h1, highest_minus_second_h1, mean_h1, betti_curve_1, persistence_entropy_1]
    
# def process_texts(texts, lat_rep, model_id, last_layer = False):
#     """
#     input: - texts: array of strings
#            - lat_rep: string of which latent representation to use for PH analysis,
#              valid arguments are "hs" for hidden states or "a" for attention
#            - model_id: which model to use
#     """
#     model, tokenizer = load_model(model_id)
    
#     data = []
#     for text in tqdm(texts):
#         if lat_rep == "hs":
#             hidden_states = get_embeddings(text, model, tokenizer, last_layer).cpu().detach().numpy()
#             graph = build_graph_embeddings(hidden_states)
#         elif lat_rep == "a":
#             attention_matrix = get_attention(text, model, tokenizer)
#             graph = build_graph_attention(attention_matrix)
#         else:
#             print("could not make a graph")
#             return
#         tda_features = compute_tda_features(graph)
#         data.append(tda_features)

#     columns = ["Num_0dim", "Max_0dim", "Max_0dim_Minus_Second", "Mean_0dim", "betti_curve_0", "persistence_entropy_0",
#                "Num_1dim", "Max_1dim", "Max_1dim_Minus_Second", "Mean_1dim", "betti_curve_1", "persistence_entropy_1"]
#     return pd.DataFrame(data, columns=columns)
