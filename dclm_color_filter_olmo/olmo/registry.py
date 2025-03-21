download_path = "/n/holyscratch01/sham_lab/dclm/color_filter"

DATA_DICT = {
    # "c4": f"{download_path}/full_data/c4",
    "prior_data": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/dclm-filtered_1-to-5",
    "conditional_data": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/memmap/core-task-trainsets-v3",
    "books-val": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/data/validation/",
}

# Pretrained model weights
MODEL_DICT = {
    "prior_8b_full": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/models/prior_8b/4725806_1/latest-unsharded",
    "prior_8b_small": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/models/prior_8b/4725783_1/latest-unsharded",
    "prior_8b_reduced": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/models/prior_8b/4769090_1/latest-unsharded",
    "conditional_hellaswag_8b_full": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/models/conditional_8b_hellaswag/8b/5146029_1/latest-unsharded",
    "conditional_hellaswag_8b_small": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/models/conditional_8b_hellaswag/8b-small/5145957_1/latest-unsharded",
    "conditional_hellaswag_8b_reduced": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/models/conditional_8b_hellaswag/8b-reduced/5145544_1/latest-unsharded",

    "conditional_books": f"{download_path}/models/conditional_books",
    "conditional_all": f"/n/netscratch/sham_lab/Everyone/dclm/color_filter/models/conditional_fineweb-edu10B/3525191_1/latest-unsharded",
    "random_1b": f"{download_path}/models/random_1b",
    "books_tau=64_1b": f"{download_path}/models/books_tau=64_1b",
    "all_tau=64_1b": f"{download_path}/models/all_tau=64_1b",
}

# Datasets of scores from auxiliary models on c4
SCORE_DICT = {
    "pretrain-1-seq": f"{download_path}/scores/pretrain-1-seq",
    "pretrain-2-seq": f"{download_path}/scores/pretrain-2-seq",
    "pretrain-3-seq": f"{download_path}/scores/pretrain-3-seq",
    "pretrain-4-seq": f"{download_path}/scores/pretrain-4-seq",
    "pretrain-5-seq": f"{download_path}/scores/pretrain-5-seq",
    "pretrain-6-seq": f"{download_path}/scores/pretrain-6-seq",
    "pretrain-7-seq": f"{download_path}/scores/pretrain-7-seq",
    "books-1-seq": f"{download_path}/scores/books-1-seq",
    "books-2-seq": f"{download_path}/scores/books-2-seq",
    "books-3-seq": f"{download_path}/scores/books-3-seq",
    "books-4-seq": f"{download_path}/scores/books-4-seq",
    "books-5-seq": f"{download_path}/scores/books-5-seq",
    "books-6-seq": f"{download_path}/scores/books-6-seq",
    "books-7-seq": f"{download_path}/scores/books-7-seq",
    "all-1-seq": f"{download_path}/scores/all-1-seq",
    "all-2-seq": f"{download_path}/scores/all-2-seq",
    "all-3-seq": f"{download_path}/scores/all-3-seq",
    "all-4-seq": f"{download_path}/scores/all-4-seq",
    "all-5-seq": f"{download_path}/scores/all-5-seq",
    "all-6-seq": f"{download_path}/scores/all-6-seq",
    "all-7-seq": f"{download_path}/scores/all-7-seq",
}

# Index files with the selected indices for each dataset
INDEX_DICT = {
    "c4-down-tau=7": f"{download_path}/indices/c4-down-tau=7/selected_indices.npy",
    "c4-down-tau=16": f"{download_path}/indices/c4-down-tau=16/selected_indices.npy",
    "c4-down-tau=32": f"{download_path}/indices/c4-down-tau=32/selected_indices.npy",
    "c4-down-tau=64": f"{download_path}/indices/c4-down-tau=64/selected_indices.npy",
    "c4-books-tau=7": f"{download_path}/indices/c4-books-tau=7/selected_indices.npy",
    "c4-books-tau=16": f"{download_path}/indices/c4-books-tau=16/selected_indices.npy",
    "c4-books-tau=32": f"{download_path}/indices/c4-books-tau=32/selected_indices.npy",
    "c4-books-tau=64": f"{download_path}/indices/c4-books-tau=64/selected_indices.npy",
}