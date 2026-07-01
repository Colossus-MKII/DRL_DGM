"""CLI."""
import os
import sys
import argparse
import pickle
import datetime
import pandas as pd
import numpy as np
import wandb

from timeit import default_timer as timer

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DRL_DIR = os.path.join(ROOT_DIR, "DRL")
if DRL_DIR not in sys.path:
    sys.path.insert(0, DRL_DIR)

from synthetizers.CTGAN.ctgan import CTGAN
from evaluation.eval_for_testing import constraints_sat_check
from utils import set_seed, read_csv, all_div_gt_n, _load_json

# wandb.log({'accuracy': train_acc, 'loss': train_loss})
# wandb.config.dropout = 0.2
# wandb.alert(title="Low accuracy", text=f"Accuracy {acc} is below threshold {thresh}")
# https://docs.wandb.ai/guides/data-and-model-versioning/dataset-versioning?_gl=1*1la1mgf*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MjguMTMuMC4w
# https://wandb.ai/dpaiton/splitting-tabular-data/reports/Tabular-Data-Versioning-and-Deduplication-with-Weights-Biases--VmlldzoxNDIzOTA1?_gl=1*1p4t0h4*_ga*MTMwNzYxOTUyOC4xNjU1MzA5NTE0*_ga_JH1SJHJQXJ*MTY3OTY3MTkyNC4xOC4xLjE2Nzk2NzI0MTUuMjYuMC4w
# https://docs.wandb.ai/guides/data-vis/tables-quickstart
DATETIME = datetime.datetime.now()


def _parse_args():
    parser = argparse.ArgumentParser(description='CTGAN Command Line Interface')
    parser.add_argument("--seed", default=7, type=int)
    parser.add_argument("--use_only_target_original_dtype", action='store_true')
    parser.add_argument("--pac", default=10, type=int)
    parser.add_argument("--wandb_project", default="ctgan", type=str)
    parser.add_argument("--wandb_mode", default="online", type=str, choices=['online', 'disabled', 'offline'])
    parser.add_argument('-e', '--epochs', default=300, type=int,
                        help='Number of training epochs')
    parser.add_argument('-n', '--num-samples', type=int,
                        help='Number of rows to sample. Defaults to the training data size')
    parser.add_argument("--save_every_n_epochs", default=5, type=int)
    parser.add_argument('--generator_lr', type=float, default=2e-4,
                        help='Learning rate for the generator.')
    parser.add_argument('--discriminator_lr', type=float, default=2e-4,
                        help='Learning rate for the discriminator.')

    parser.add_argument('--generator_decay', type=float, default=1e-6,
                        help='Weight decay for the generator.')
    parser.add_argument('--discriminator_decay', type=float, default=0,
                        help='Weight decay for the discriminator.')
    parser.add_argument('--optimiser', type=str, default="adam", choices=['adam','rmsprop','sgd'], help='')

    parser.add_argument('--embedding_dim', type=int, default=128,
                        help='Dimension of input z to the generator.')
    parser.add_argument('--generator_dim', type=str, default='256,256',
                        help='Dimension of each generator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument('--discriminator_dim', type=str, default='256,256',
                        help='Dimension of each discriminator layer. '
                        'Comma separated integers with no whitespaces.')
    parser.add_argument("--label_ordering", default='random', choices=['random', 'corr', 'kde', 'wasserstein', 'jsd', 'causal'])

    parser.add_argument('--batch_size', type=int, default=500,
                        help='Batch size. Must be an even number.')
    parser.add_argument('--save', default=None, type=str,
                        help='A filename to save the trained synthesizer.')
    parser.add_argument('--load', default=None, type=str,
                        help='A filename to load a trained synthesizer.')

    parser.add_argument('--sample_condition_column', default=None, type=str,
                        help='Select a discrete column name.')
    parser.add_argument('--sample_condition_column_value', default=None, type=str,
                        help='Specify the value of the selected discrete column.')
    parser.add_argument("use_case", type=str, choices=["url","wids","botnet","lcld","heloc","news","faults",'kc','cc'])
    parser.add_argument("--version", type=str, default='unconstrained', choices=['unconstrained','constrained', "postprocessing"],
                        help='Version of training. Correct values are unconstrained, constrained and postprocessing')
    parser.add_argument('--skip_evaluation', action='store_true')
    parser.add_argument('--runtime_evaluation_only', action='store_true')

    return parser.parse_args()


def _dataset_paths(use_case, tiny=False):
    data_dir = os.path.join("data", use_case, "tiny") if tiny else os.path.join("data", use_case)
    return {
        "train": os.path.join(data_dir, "train_data.csv"),
        "test": os.path.join(data_dir, "test_data.csv"),
        "val": os.path.join(data_dir, "val_data.csv"),
    }


def _prepare_heloc_dataset(paths):
    try:
        from datasets import load_dataset
        from sklearn.model_selection import train_test_split
    except ImportError as exc:
        raise RuntimeError(
            "HELOC split files are missing and this environment cannot generate them. "
            "Install the missing dependency with `pip install datasets scikit-learn`, "
            "then rerun main_ctgan.py."
        ) from exc

    print("HELOC split files are missing; preparing them from Hugging Face dataset mstz/heloc.", flush=True)
    os.makedirs(os.path.dirname(paths["train"]), exist_ok=True)

    try:
        dataset = load_dataset("mstz/heloc")["train"]
    except Exception as exc:
        raise RuntimeError(
            "Failed to download the HELOC dataset from Hugging Face dataset `mstz/heloc`. "
            "Check network access, or run `other_helper_scripts/prepare_heloc_dataset.py` "
            "after installing `datasets`."
        ) from exc

    df = pd.DataFrame(dataset)
    df.drop_duplicates(inplace=True)

    train_ratio = 0.8
    validation_ratio = 0.10
    test_ratio = 0.10
    ratio_remaining = 1 - test_ratio
    ratio_val_adjusted = validation_ratio / ratio_remaining

    x_train, x_test = train_test_split(df, test_size=1 - train_ratio, random_state=1)
    x_val, x_test = train_test_split(x_test, test_size=ratio_val_adjusted, random_state=1)

    x_train.to_csv(paths["train"], index=False)
    x_test.to_csv(paths["test"], index=False)
    x_val.to_csv(paths["val"], index=False)
    print("HELOC split files written:", paths, flush=True)


def _ensure_dataset_files(use_case, paths):
    missing = [path for path in paths.values() if not os.path.exists(path)]
    if not missing:
        return

    if use_case == "heloc":
        _prepare_heloc_dataset(paths)
        missing = [path for path in paths.values() if not os.path.exists(path)]
        if not missing:
            return

    raise FileNotFoundError(
        f"Missing dataset split files for use_case={use_case}: {missing}. "
        "Expected train/test/val CSV files named train_data.csv, test_data.csv, and val_data.csv."
    )


def _load_dataset_splits(args, dataset_info):
    if args.use_case == "botnet":
        dataset_paths = _dataset_paths(args.use_case, tiny=True)
    else:
        dataset_paths = _dataset_paths(args.use_case)

    _ensure_dataset_files(args.use_case, dataset_paths)
    print(f"Loading dataset splits from {dataset_paths}", flush=True)

    X_train, (cat_cols, cat_idx), (roundable_idx, round_digits) = read_csv(
        dataset_paths["train"],
        args.use_case,
        dataset_info["manual_inspection_categorical_cols_idx"],
    )
    X_test = pd.read_csv(dataset_paths["test"])
    X_val = pd.read_csv(dataset_paths["val"])

    print(
        f"Loaded dataset splits: train={X_train.shape}, val={X_val.shape}, test={X_test.shape}",
        flush=True,
    )

    return X_train, X_test, X_val, (cat_cols, cat_idx), (roundable_idx, round_digits), dataset_paths


def main():
    """CLI."""
    print(f"Starting main_ctgan.py from {os.path.abspath(__file__)}", flush=True)
    args = _parse_args()
    set_seed(args.seed)
    exp_id = f"{args.version}_{args.label_ordering}_{args.seed}_{args.epochs}_{args.batch_size}_{args.discriminator_lr}_{args.generator_lr}_{DATETIME:%d-%m-%y--%H-%M-%S}"
    path = f"outputs/CTGAN_out/{args.use_case}/{args.version}/{exp_id}"
    args.exp_path = path
    os.makedirs(path, exist_ok=True)


    # set args.pac:
    if args.pac != 1:
        if args.batch_size % args.pac != 0:
            original_pac = args.pac
            args.pac = all_div_gt_n(original_pac, args.batch_size)
            print(f'Changed pac {original_pac} to {args.pac}')

    args.constraints_file = f'./data/{args.use_case}/{args.use_case}_constraints.txt'
    ######################################################################
    dataset_info = _load_json("datasets_info.json")[args.use_case]
    print(dataset_info)
    ######################################################################
    X_train, X_test, X_val, (cat_cols, cat_idx), (roundable_idx, round_digits), dataset_paths = _load_dataset_splits(args, dataset_info)
    columns = X_train.columns.values.tolist()
    args.train_data_cols = columns
    args.dtypes = X_train.dtypes

    if cat_cols == None:
        cat_cols = []
        cat_idx = []

    ######################################################################
    print("Initialising wandb and CTGAN training.", flush=True)
    wandb_run = wandb.init(project=args.wandb_project, id=exp_id, reinit=True, mode=args.wandb_mode)
    for k, v in args._get_kwargs():
        wandb_run.config[k] = v
    ######################################################################

    if args.load:
        model = CTGAN.load(args.load)
    else:
        generator_dim = [int(x) for x in args.generator_dim.split(',')]
        discriminator_dim = [int(x) for x in args.discriminator_dim.split(',')]
        model = CTGAN(X_test,
            embedding_dim=args.embedding_dim, generator_dim=generator_dim,
            discriminator_dim=discriminator_dim, generator_lr=args.generator_lr,
            generator_decay=args.generator_decay, discriminator_lr=args.discriminator_lr,
            discriminator_decay=args.discriminator_decay, batch_size=args.batch_size,
            epochs=args.epochs, path=path, bin_cols_idx=cat_idx, version=args.version, pac=args.pac,
                      feats_in_constraints=dataset_info["feats_in_constraints"])

    model.set_random_state(args.seed)
    print(f"Starting CTGAN fit for {args.epochs} epochs.", flush=True)
    model.fit(args, X_train, cat_cols)
    print("CTGAN fit finished.", flush=True)

    # args.save = f'{path}/final_ctgan_model.pt'
    if args.save is not None:
        model.save(args.save)

    if args.sample_condition_column is not None:
        assert args.sample_condition_column_value is not None

    if args.use_case == "botnet" or args.use_case == "lcld":
        dataset_paths = _dataset_paths(args.use_case, tiny=True)
        _ensure_dataset_files(args.use_case, dataset_paths)
        X_train = pd.read_csv(dataset_paths["train"])
        X_test = pd.read_csv(dataset_paths["test"])
        X_val = pd.read_csv(dataset_paths["val"])
    args.sampling_sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        
    model.set_random_state(args.seed)
    num_sampling_rounds = 5

    if args.runtime_evaluation_only:
        size = 1000
        runs = []
        for i in range(num_sampling_rounds):
            start = timer()
            sampled_data, unconstrained_output = model.sample(size, args.sample_condition_column, args.sample_condition_column_value)
            end = timer()
            runtime = end - start
            runs.append(runtime)
        runtime_df = pd.DataFrame(list(zip([np.mean(runs)],[np.std(runs)])), columns=["Mean", "Std"])
        wandb.log({'Runtime/Sampling': runtime_df})
    
    else:

        gen_data = [[], [], []]
        unconstrained_gen_data = [[], [], []]
        constrained_unrounded_gen_data = [[], [], []]
        sizes = [X_train.shape[0], X_val.shape[0], X_test.shape[0]]
        for r in range(num_sampling_rounds): 
            for i in range (len(sizes)):
                sampled_data, unconstrained_output = model.sample(sizes[i], args.sample_condition_column, args.sample_condition_column_value)
                unconstrained_gen_data[i].append(unconstrained_output)
                constrained_unrounded_output = sampled_data
                constrained_unrounded_output = pd.DataFrame(constrained_unrounded_output, columns=columns)
                constrained_unrounded_output = constrained_unrounded_output.astype(float)
                target_col = columns[-1]
                constrained_unrounded_output[target_col] = constrained_unrounded_output[target_col].astype(X_train.dtypes[-1])
                constrained_unrounded_gen_data[i].append(constrained_unrounded_output)

                # sampled_data = pd.DataFrame(sampled_data, columns=columns)
                # sampled_data.iloc[:, roundable_idx] = sampled_data.iloc[:, roundable_idx].round(round_digits)  # NOTE: this shouldn't be after the constraints have been applied! (fixed by removing constr correction from sample fc, and adding it below here)
                # sampled_data = sampled_data.astype(X_train.dtypes)

                gen_data[i].append(constrained_unrounded_output)


        generated_data = {"train":gen_data[0], "val":gen_data[1], "test":gen_data[2]}
        unconstrained_generated_data = {"train":unconstrained_gen_data[0], "val":unconstrained_gen_data[1], "test":unconstrained_gen_data[2]}
        constrained_unrounded_generated_data = {"train":constrained_unrounded_gen_data[0], "val":constrained_unrounded_gen_data[1], "test":constrained_unrounded_gen_data[2]}

        with open(f'{path}/generated_data.pkl', 'wb') as f:
            pickle.dump(generated_data, f)
        with open(f'{path}/unconstrained_generated_data.pkl', 'wb') as f:
            pickle.dump(unconstrained_generated_data, f)
        with open(f'{path}/constrained_unrounded_generated_data.pkl', 'wb') as f:
            pickle.dump(constrained_unrounded_generated_data, f)

        real_data = {"train": X_train, "val": X_val, "test": X_test}



        if not args.skip_evaluation: 

            generated_label = "unconstrained"
            comparison_data = None
            if args.version == "constrained":
                generated_label = "constrained"
                comparison_data = {"unconstrained": unconstrained_generated_data}
            elif args.version == "postprocessing":
                generated_label = "postprocessed"
                comparison_data = {"unconstrained": unconstrained_generated_data}

            constraints_sat_check(
                args,
                real_data,
                generated_data,
                log_wandb=True,
                generated_label=generated_label,
                comparison_data=comparison_data,
            )
            print("Skipping synthetic quality and utility evaluation: this repository "
                  "snapshot is missing evaluation.eval and gather_results.reeval_final.")

            # wandb.finish()
            # ######################################################################
            # args.real_data_partition = 'test'
            # args.model_type = 'ctgan'
            #
            # if 'hyerparam' in args.wandb_project or 'hyper' in args.wandb_project:
            #     args.wandb_project = f"DRL_evaluation_{args.model_type}_{args.use_case}_hyperparam_search"
            # else:
            #     args.wandb_project = f"DRL_evaluation_{args.model_type}_{args.use_case}"
            #
            # wandb_run = wandb.init(project=args.wandb_project, id=exp_id)
            # for k, v in args._get_kwargs():
            #     wandb_run.config[k] = v
            # ######################################################################
            # args.round_before_cons = False
            # args.round_after_cons = False
            # args.postprocessing = False
            # if args.version != 'unconstrained':
            #     args.version = args.label_ordering
            #
            # generated_data, unrounded_generated_data = prepare_gen_data(args, unconstrained_generated_data, roundable_idx, round_digits, columns, X_train)
            #
            # constraints_sat_check(args, real_data, unrounded_generated_data, log_wandb=True)
            # sdv_eval_synthetic_data(args, args.use_case, real_data, generated_data, columns,
            #                         problem_type=dataset_info["problem_type"],
            #                         target_utility=dataset_info["target_col"], target_detection="", log_wandb=True,
            #                         wandb_run=wandb_run)
            # print('Using evaluators with the following specs', dataset_info["problem_type"], dataset_info["target_size"],
            #     dataset_info["target_col"])
            # eval_synthetic_data(args, args.use_case, real_data, generated_data, columns,
            #                     problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"],
            #                     target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True,
            #                     wandb_run=wandb_run, unrounded_generated_data_for_cons_sat=unrounded_generated_data)


            # if args.seed < 3:
            #     constraints_sat_check(args, real_data, generated_data, log_wandb=True)
            #     sdv_eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"], target_detection="", log_wandb=True, wandb_run=wandb_run)
            #     eval_synthetic_data(args, args.use_case, real_data, generated_data, columns, problem_type=dataset_info["problem_type"], target_utility=dataset_info["target_col"], target_utility_size=dataset_info["target_size"], target_detection="", log_wandb=True, wandb_run=wandb_run)

    wandb.finish()

if __name__ == '__main__':

    main()
