import configargparse
import argparse


def _add_common_args(arg_parser):
    arg_parser.add('--config', type=str)

    # Input
    arg_parser.add('--types_path', type=str, help="Path to type specifications")

    # Preprocessing
    arg_parser.add('--tokenizer_path', type=str, help="Path to tokenizer")
    arg_parser.add('--max_span_size', type=int, default=10, help="Maximum size of spans")
    arg_parser.add('--lowercase', action='store_true', default=False,
                            help="If true, input is lowercased during preprocessing")
    arg_parser.add('--sampling_processes', type=int, default=4,
                            help="Number of sampling processes. 0 = no multiprocessing for sampling")
    arg_parser.add('--sampling_limit', type=int, default=100, help="Maximum number of sample batches in queue")

    # Logging
    arg_parser.add('--label', type=str, help="Label of run. Used as the directory name of logs/models")
    arg_parser.add('--log_path', type=str, help="Path do directory where training/evaluation logs are stored")
    arg_parser.add('--store_predictions', action='store_true', default=False,
                            help="If true, store predictions on disc (in log directory)")
    arg_parser.add('--store_examples', action='store_true', default=False,
                            help="If true, store evaluation examples on disc (in log directory)")
    arg_parser.add('--example_count', type=int, default=None,
                            help="Count of evaluation example to store (if store_examples == True)")
    arg_parser.add('--debug', action='store_true', default=False, help="Debugging mode on/off")

    # Model / Training / Evaluation
    arg_parser.add('--model_path', type=str, help="Path to directory that contains model checkpoints")
    arg_parser.add('--model_type', type=str, default="SynFue", help="Type of model")
    arg_parser.add('--cpu', action='store_true', default=False,
                            help="If true, train/evaluate on CPU even if a CUDA device is available")
    arg_parser.add('--eval_batch_size', type=int, default=1, help="Evaluation batch size")
    arg_parser.add('--max_pairs', type=int, default=1000,
                            help="Maximum term pairs to process during training/evaluation")
    arg_parser.add('--rel_filter_threshold', type=float, default=0.4, help="Filter threshold for relations")
    arg_parser.add('--size_embedding', type=int, default=25, help="Dimensionality of size embedding")
    arg_parser.add('--prop_drop', type=float, default=0.1, help="Probability of dropout used in SpERT")
    arg_parser.add('--freeze_transformer', action='store_true', default=False, help="Freeze BERT weights")
    arg_parser.add('--no_overlapping', action='store_true', default=False,
                            help="If true, do not evaluate on overlapping terms "
                                 "and relations with overlapping terms")

    # SGCN
    arg_parser.add('--bert_dim', type=int, default=768)
    arg_parser.add('--dep_dim', type=int, default=100)
    arg_parser.add('--bert_dropout', type=float, default=0.5)
    arg_parser.add('--output_dropout', type=float, default=0.5)
    arg_parser.add('--dep_num', type=int, default=42)  # 40 + 1(None) + 1(self-loop)
    arg_parser.add('--num_layer', type=int, default=3)

    # POS
    # arg_parser.add('--use_pos', type=bool, default=True)
    arg_parser.add('--pos_num', type=int, default=45)
    arg_parser.add('--pos_dim', type=int, default=100)
    arg_parser.add('--w_size', type=int, default=1, help='the window size of local attention')

    # Misc
    arg_parser.add('--seed', type=int, default=58986, help="Seed")
    arg_parser.add('--cache_path', type=str, default=None,
                            help="Path to cache transformer models (for HuggingFace transformers library)")
    
    arg_parser.add_argument('--alpha', type=float, default=1.0, help="alpha")
    arg_parser.add_argument('--beta', type=float, default=0.4, help="beta")
    arg_parser.add_argument('--sigma', type=float, default=1.0, help="sigma")

def train_argparser():
    arg_parser = argparse.ArgumentParser()

    # Input
    arg_parser.add('--train_path', type=str, help="Path to train dataset")
    arg_parser.add('--valid_path', type=str, help="Path to validation dataset")

    # Logging
    arg_parser.add('--save_path', type=str, help="Path to directory where model checkpoints are stored")
    arg_parser.add('--init_eval', action='store_true', default=False,
                            help="If true, evaluate validation set before training")
    arg_parser.add('--save_optimizer', action='store_true', default=False,
                            help="Save optimizer alongside model")
    arg_parser.add('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    arg_parser.add('--final_eval', action='store_true', default=False,
                            help="Evaluate the model only after training, not at every epoch")

    # Model / Training
    arg_parser.add('--train_batch_size', type=int, default=2, help="Training batch size")
    arg_parser.add('--epochs', type=int, default=20, help="Number of epochs")
    arg_parser.add('--neg_term_count', type=int, default=100,
                            help="Number of negative term samples per document (sentence)")
    arg_parser.add('--neg_relation_count', type=int, default=100,
                            help="Number of negative relation samples per document (sentence)")
    arg_parser.add('--lr', type=float, default=5e-5, help="Learning rate")
    arg_parser.add('--lr_warmup', type=float, default=0.1,
                            help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    arg_parser.add('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    arg_parser.add('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")

    _add_common_args(arg_parser)

    return arg_parser


def eval_argparser():
    arg_parser = configargparse.ArgParser(default_config_files=['/content/configs/eval.conf'])

    # Input
    # arg_parser.add('--dataset_path', type=str, help="Path to dataset")

    _add_common_args(arg_parser)

    return arg_parser
