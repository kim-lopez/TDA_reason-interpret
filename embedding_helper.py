import analysis_functions as topo

def get_embeddings(text, model, tokenizer, index):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)

    llm_answer = tokenizer.decode(llm_tokens, skip_special_tokens=True,
                                  clean_up_tokenization_spaces=False)
    
    print(f"Total number of layers captured (embeddings + blocks): {len(all_hidden_states)}")
    print(f"Shape of hidden states at layer 15: {all_hidden_states[15].shape}")

    # outputs.hidden_states is a tuple containing:                                                                                                                                            
    # index 0: output of the embedding layer                                                                                                                                                
    # index > 1 output of each respective transformer decoder layer                                                                                                  
    all_hidden_states = outputs.hidden_states

    return all_hidden_states[index], llm_answer

def compute_emb_tda_feat(embeddings):
    """
    Run persistent homology directly on the
    activations!
    """
    embeddings = np.asarray(
        embeddings,
        dtype=np.float64
    )

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


def process_emb(texts, answer, model_id, index):
    model, tokenizer = topo.load_model(model_id)
    data = []

    for index, text in enumerate(tqdm(texts)):
        embeddings, llm_answer = get_embeddings(text, model, tokenizer, index)
        real_answer = answer[index]
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

def emb_top_feat(model, dataset, index, create = False):
    # get labels for model
    model_name, model_id, model_short  = topo.which_model(model)

    # labels for dataset
    data_name, data_csv = topo.data_label(dataset)
    questions = pd.read_csv(data_csv)
    
    tda_path = os.path.expanduser(f"~/TDA_RI/TDA_reason-interpret/{model_short}/{model_short}_{data_name}_tda_{index}.csv")

    # either create or load data
    if create:
        feats_sen = questions["prompt"]
        answer = []
        if data_name == "hellaswag":
            answer = questions["label"]
        else:
            answer = questions["answer"]
        feats_tda = process_emb(feats_sen, answer, model_id, index)
        feats_tda.to_csv(tda_path, index=False)
        print(f"Added questions from {data_name} for {model_short}! (layer {index})")
   
    else:
        feats_tda = pd.read_csv(tda_path)

    return feats_tda