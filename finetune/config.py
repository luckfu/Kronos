import os

class Config:
    """
    Configuration class for the entire project.
    """

    def __init__(self):
        # =================================================================
        # Data & Feature Parameters
        # =================================================================
        # TODO: Update this path to your Qlib data directory.
        self.qlib_data_path = "~/.qlib/qlib_data/cn_data"
        self.instrument = 'csi800'

        # Overall time range for data loading from Qlib.
        self.dataset_begin_time = "2015-01-01"
        self.dataset_end_time = '2026-12-31'

        # Sliding window parameters for creating samples.
        self.lookback_window = 90  # Number of past time steps for input.
        self.predict_window = 10  # Number of future time steps for prediction.
        self.max_context = 512  # Maximum context length for the model.

        # Features to be used from the raw data.
        self.feature_list = ['open', 'high', 'low', 'close', 'volume', 'amount']
        # Time-based features to be generated.
        self.time_feature_list = ['minute', 'hour', 'weekday', 'day', 'month']

        # Optional static asset metadata used to condition the predictor.
        # The metadata CSV should contain: symbol, sector, size_bucket.
        # Leave empty to run the original model without conditioning.
        self.asset_metadata_path = os.getenv("KRONOS_METADATA_PATH", "./data/a_share/asset_metadata.csv")
        self.use_context_features = True
        self.use_sector_features = False
        self.use_size_features = True
        self.use_size_percentile = os.getenv(
            "KRONOS_USE_SIZE_PERCENTILE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.num_sectors = 0
        self.num_size_buckets = 10
        self.size_mlp_hidden_dim = int(os.getenv("KRONOS_SIZE_MLP_HIDDEN_DIM", "64"))

        # =================================================================
        # Dataset Splitting & Paths
        # =================================================================
        # Note: The validation/test set starts earlier than the training/validation set ends
        # to account for the `lookback_window`.
        self.train_time_range = ["2015-01-01", "2025-12-31"]
        self.val_time_range = ["2026-01-01", "2026-12-31"]
        self.test_time_range = ["2027-01-01", "2027-12-31"]
        self.backtest_time_range = ["2026-01-01", "2026-12-31"]

        # TODO: Directory to save the processed, pickled datasets.
        self.dataset_path = os.getenv("KRONOS_DATASET_PATH", "./data/a_share/processed_datasets")

        # =================================================================
        # Training Hyperparameters
        # =================================================================
        self.clip = 5.0  # Clipping value for normalized data to prevent outliers.

        self.epochs = int(os.getenv("KRONOS_EPOCHS", "50"))
        self.early_stopping_patience = int(
            os.getenv("KRONOS_EARLY_STOPPING_PATIENCE", "5")
        )
        self.log_interval = int(os.getenv("KRONOS_LOG_INTERVAL", "100"))
        self.batch_size = int(os.getenv("KRONOS_BATCH_SIZE", "4"))
        self.num_workers = int(os.getenv("KRONOS_NUM_WORKERS", "0"))
        # Coverage training uses unique windows within each segment and advances
        # through one fixed permutation before any window is reused. Zero means
        # a literal full-dataset epoch, which takes about 26 hours on this MPS host.
        self.n_train_iter = int(os.getenv("KRONOS_TRAIN_SAMPLES_PER_SEGMENT", "20000"))
        self.n_val_iter = int(os.getenv("KRONOS_VALIDATION_SAMPLES", "2000"))
        self.coverage_passes = int(os.getenv("KRONOS_COVERAGE_PASSES", "1"))
        self.require_full_coverage = os.getenv(
            "KRONOS_REQUIRE_FULL_COVERAGE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.resume_training = os.getenv(
            "KRONOS_RESUME_TRAINING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.reset_size_embedding = os.getenv(
            "KRONOS_RESET_SIZE_EMBEDDING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.balance_size_buckets = os.getenv(
            "KRONOS_BALANCE_SIZE_BUCKETS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}

        # Learning rates for different model components.
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = 1e-5
        self.condition_learning_rate = 1e-3
        self.trainable_transformer_layers = 2
        self.context_layer = 10

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = 0.1

        # Miscellaneous
        self.seed = 100  # Global random seed for reproducibility.

        # =================================================================
        # Experiment Logging & Saving
        # =================================================================
        self.use_comet = False # Set to True only when Comet ML is configured
        self.comet_config = {
            # It is highly recommended to load secrets from environment variables
            # for security purposes. Example: os.getenv("COMET_API_KEY")
            "api_key": "YOUR_COMET_API_KEY",
            "project_name": "Kronos-Finetune-Demo",
            "workspace": "your_comet_workspace" # TODO: Change to your Comet ML workspace name
        }
        self.comet_tag = 'finetune_demo'
        self.comet_name = 'finetune_demo'

        # Base directory for saving model checkpoints and results.
        # Using a general 'outputs' directory is a common practice.
        self.save_path = os.getenv("KRONOS_SAVE_PATH", "./outputs/models")
        self.tokenizer_save_folder_name = 'finetune_tokenizer_demo'
        self.predictor_save_folder_name = os.getenv(
            "KRONOS_PREDICTOR_SAVE_FOLDER", "a_share_size_kronos_base"
        )
        self.backtest_save_folder_name = 'finetune_backtest_demo'

        # Path for backtesting results.
        self.backtest_result_path = "./outputs/backtest_results"

        # =================================================================
        # Model & Checkpoint Paths
        # =================================================================
        # TODO: Update these paths to your pretrained model locations.
        # These can be local paths or Hugging Face Hub model identifiers.
        self.pretrained_tokenizer_path = "NeoQuasar/Kronos-Tokenizer-base"
        self.pretrained_predictor_path = os.getenv(
            "KRONOS_PREDICTOR_PATH", "./Kronos-base"
        )

        # Paths to the fine-tuned models, derived from the save_path.
        # These will be generated automatically during training.
        self.finetuned_tokenizer_path = f"{self.save_path}/{self.tokenizer_save_folder_name}/checkpoints/best_model"
        self.finetuned_predictor_path = f"{self.save_path}/{self.predictor_save_folder_name}/checkpoints/best_model"

        # =================================================================
        # Backtesting Parameters
        # =================================================================
        self.backtest_n_symbol_hold = 50  # Number of symbols to hold in the portfolio.
        self.backtest_n_symbol_drop = 5  # Number of symbols to drop from the pool.
        self.backtest_hold_thresh = 5  # Minimum holding period for a stock.
        self.inference_T = 0.6
        self.inference_top_p = 0.9
        self.inference_top_k = 0
        self.inference_sample_count = 5
        self.backtest_batch_size = 1000
        self.backtest_benchmark = self._set_benchmark(self.instrument)

    def _set_benchmark(self, instrument):
        dt_benchmark = {
            'csi800': "SH000906",
            'csi1000': "SH000852",
            'csi300': "SH000300",
        }
        if instrument in dt_benchmark:
            return dt_benchmark[instrument]
        else:
            raise ValueError(f"Benchmark not defined for instrument: {instrument}")
