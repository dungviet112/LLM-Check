import argparse

from six.moves import cPickle as pkl

from common_utils import *

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="open-llama-7b", help="model name or path for the evaluator model")
parser.add_argument("--n_samples", type=int, default=100)
parser.add_argument("--dataset", choices=["fava", "fava_annot", "selfcheck", "rag_truth", "trivia_qa"], default="trivia_qa")
parser.add_argument(
    "--use_toklens", action="store_true", help="remove prompt prefix before computing hidden and eigen scores"
)
parser.add_argument(
    "--mt", choices=["logit", "hidden", "attns"], action="append", help="choose method types for detection scores"
)


args = parser.parse_args()

if __name__ == "__main__":
    n_samples = args.n_samples
    model_name_or_path = get_full_model_name(args.model.lower())[1]
    mt_list = args.mt

    print(
        f"Model: {args.model.lower()}, Method types: {mt_list}, Dataset: {args.dataset}, Use toklens: {args.use_toklens}"
    )

    # load dataset specific utils
    get_data, get_scores_dict = load_dataset_utils(args)

    sample_data, _ = get_data(n_samples=n_samples)

    # get scores for sample data
    scores, sample_indiv_scores, generated_embeddings, sample_labels = get_scores_dict(model_name_or_path, sample_data, mt_list, args)

    # save the scores to /data
    with open(f"data/full_context/scores_{args.dataset}_{args.model.lower()}_layer32_{n_samples}samp.pkl", "wb") as f:
        pkl.dump([scores, sample_indiv_scores, generated_embeddings, sample_labels], f)
