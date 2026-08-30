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

        # Sliding window parameters for creating samples.  Keep the production
        # defaults unchanged; long-context experiments (for example 120+10)
        # opt in through environment variables in their run script.
        self.lookback_window = int(os.getenv("KRONOS_LOOKBACK_WINDOW", "90"))
        self.predict_window = int(os.getenv("KRONOS_PREDICT_WINDOW", "10"))
        if self.lookback_window < 1 or self.predict_window < 1:
            raise ValueError("KRONOS_LOOKBACK_WINDOW and KRONOS_PREDICT_WINDOW must be positive")
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
        self.use_sector_features = os.getenv(
            "KRONOS_USE_SECTOR_FEATURES", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.use_size_features = os.getenv(
            "KRONOS_USE_SIZE_FEATURES", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.use_size_percentile = os.getenv(
            "KRONOS_USE_SIZE_PERCENTILE", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.disable_condition_inputs = os.getenv(
            "KRONOS_DISABLE_CONDITION_INPUTS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.num_sectors = int(os.getenv("KRONOS_NUM_SECTORS", "86"))
        self.num_size_buckets = int(os.getenv("KRONOS_NUM_SIZE_BUCKETS", "10"))
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
        default_train_path = os.path.join(self.dataset_path, "train_data.pkl")
        default_val_path = os.path.join(self.dataset_path, "val_data.pkl")
        self.train_data_paths = tuple(
            path for path in os.getenv(
                "KRONOS_TRAIN_DATA_PATHS", default_train_path
            ).split(os.pathsep) if path
        )
        self.val_data_paths = tuple(
            path for path in os.getenv(
                "KRONOS_VAL_DATA_PATHS", default_val_path
            ).split(os.pathsep) if path
        )
        self.train_signal_start = os.getenv("KRONOS_TRAIN_SIGNAL_START", "")
        self.train_signal_end = os.getenv("KRONOS_TRAIN_SIGNAL_END", "")
        self.val_signal_start = os.getenv("KRONOS_VAL_SIGNAL_START", "")
        self.val_signal_end = os.getenv("KRONOS_VAL_SIGNAL_END", "")
        self.dataset_manifest_sha256 = os.getenv(
            "KRONOS_DATA_MANIFEST_SHA256", ""
        )
        self.fixed_validation_manifest_path = os.getenv(
            "KRONOS_FIXED_VALIDATION_MANIFEST_PATH", ""
        )
        self.fixed_validation_manifest_sha256 = os.getenv(
            "KRONOS_FIXED_VALIDATION_MANIFEST_SHA256", ""
        )
        self.exclude_fixed_validation_from_training = os.getenv(
            "KRONOS_EXCLUDE_FIXED_VALIDATION_FROM_TRAINING", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.validation_quick_samples = int(
            os.getenv("KRONOS_VALIDATION_QUICK_SAMPLES", "3000")
        )
        self.validation_full_only = os.getenv(
            "KRONOS_VALIDATION_FULL_ONLY", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.validation_large_samples = int(
            os.getenv("KRONOS_VALIDATION_LARGE_SAMPLES", "12000")
        )
        self.validation_large_interval_segments = int(
            os.getenv("KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS", "10")
        )
        if not self.validation_full_only and self.validation_quick_samples <= 0:
            raise ValueError("KRONOS_VALIDATION_QUICK_SAMPLES must be positive")
        if (
            not self.validation_full_only
            and self.validation_large_samples < self.validation_quick_samples
        ):
            raise ValueError(
                "KRONOS_VALIDATION_LARGE_SAMPLES must be at least the quick sample count"
            )
        if self.validation_large_interval_segments <= 0:
            raise ValueError(
                "KRONOS_VALIDATION_LARGE_INTERVAL_SEGMENTS must be positive"
            )
        self.history_replay_ratio = float(
            os.getenv("KRONOS_HISTORY_REPLAY_RATIO", "0")
        )
        self.replay_signal_start = os.getenv("KRONOS_REPLAY_SIGNAL_START", "")
        self.replay_signal_end = os.getenv("KRONOS_REPLAY_SIGNAL_END", "")

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
        self.use_amp = os.getenv(
            "KRONOS_USE_AMP", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.amp_dtype = os.getenv("KRONOS_AMP_DTYPE", "float16").strip().lower()
        if self.amp_dtype not in {"float16", "fp16", "bfloat16", "bf16"}:
            raise ValueError(
                "KRONOS_AMP_DTYPE must be float16/fp16 or bfloat16/bf16"
            )
        # Coverage training uses unique windows within each segment and advances
        # through one fixed permutation before any window is reused. Zero means
        # a literal full-dataset epoch, which takes about 26 hours on this MPS host.
        self.n_train_iter = int(os.getenv("KRONOS_TRAIN_SAMPLES_PER_SEGMENT", "20000"))
        self.n_val_iter = int(os.getenv("KRONOS_VALIDATION_SAMPLES", "2000"))
        self.coverage_passes = int(os.getenv("KRONOS_COVERAGE_PASSES", "1"))
        # Limit only the number of completed segments in this process. This is
        # intentionally separate from epochs so resumed jobs keep one global
        # coverage plan and one unchanged OneCycle schedule.
        self.max_segments_per_run = int(
            os.getenv("KRONOS_MAX_SEGMENTS_PER_RUN", "0")
        )
        self.require_full_coverage = os.getenv(
            "KRONOS_REQUIRE_FULL_COVERAGE", "1"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.resume_training = os.getenv(
            "KRONOS_RESUME_TRAINING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.bootstrap_completed_segments = int(
            os.getenv("KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS", "0")
        )
        if self.bootstrap_completed_segments < 0:
            raise ValueError("KRONOS_BOOTSTRAP_COMPLETED_SEGMENTS must be non-negative")
        bootstrap_best_val_loss = os.getenv(
            "KRONOS_BOOTSTRAP_BEST_VAL_LOSS", ""
        ).strip()
        self.bootstrap_best_val_loss = (
            float(bootstrap_best_val_loss)
            if bootstrap_best_val_loss
            else float("inf")
        )
        self.reset_size_embedding = os.getenv(
            "KRONOS_RESET_SIZE_EMBEDDING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.reset_sector_embedding = os.getenv(
            "KRONOS_RESET_SECTOR_EMBEDDING", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.balance_size_buckets = os.getenv(
            "KRONOS_BALANCE_SIZE_BUCKETS", "0"
        ).strip().lower() in {"1", "true", "yes", "on"}
        self.predictor_loss_mode = os.getenv(
            "KRONOS_PREDICTOR_LOSS_MODE", "full_sequence"
        ).strip().lower()
        if self.predictor_loss_mode not in {"full_sequence", "forecast"}:
            raise ValueError(
                "KRONOS_PREDICTOR_LOSS_MODE must be full_sequence or forecast"
            )
        self.history_loss_weight = float(
            os.getenv("KRONOS_HISTORY_LOSS_WEIGHT", "0")
        )
        if self.history_loss_weight < 0:
            raise ValueError("KRONOS_HISTORY_LOSS_WEIGHT must be non-negative")
        self.best_selection_metric = os.getenv(
            "KRONOS_BEST_SELECTION_METRIC", "objective"
        ).strip().lower()
        if self.best_selection_metric not in {
            "objective", "full_sequence", "forecast", "history",
            "validation_large_objective"
        }:
            raise ValueError(
                "KRONOS_BEST_SELECTION_METRIC must be objective, full_sequence, "
                "forecast, history, or validation_large_objective"
            )
        forecast_weights = os.getenv("KRONOS_FORECAST_HORIZON_WEIGHTS", "").strip()
        if forecast_weights:
            self.forecast_horizon_weights = tuple(
                float(value.strip()) for value in forecast_weights.split(",")
            )
            if len(self.forecast_horizon_weights) != self.predict_window:
                raise ValueError(
                    "KRONOS_FORECAST_HORIZON_WEIGHTS must provide one value per "
                    "forecast day"
                )
            if any(value <= 0 for value in self.forecast_horizon_weights):
                raise ValueError("Forecast horizon weights must be positive")
        else:
            self.forecast_horizon_weights = tuple(1.0 for _ in range(self.predict_window))

        # Learning rates for different model components.
        self.tokenizer_learning_rate = 2e-4
        self.predictor_learning_rate = float(
            os.getenv("KRONOS_PREDICTOR_LEARNING_RATE", "1e-5")
        )
        self.condition_learning_rate = float(
            os.getenv("KRONOS_CONDITION_LEARNING_RATE", "1e-4")
        )
        self.scheduler_min_learning_rate = float(
            os.getenv("KRONOS_SCHEDULER_MIN_LR", "1e-6")
        )
        self.predictor_min_learning_rate = float(
            os.getenv(
                "KRONOS_PREDICTOR_MIN_LR",
                str(self.scheduler_min_learning_rate),
            )
        )
        self.condition_min_learning_rate = float(
            os.getenv(
                "KRONOS_CONDITION_MIN_LR",
                str(self.scheduler_min_learning_rate),
            )
        )
        self.scheduler_type = os.getenv(
            "KRONOS_SCHEDULER", "warmup_cosine"
        ).strip().lower()
        if self.scheduler_type not in {
            "warmup_cosine", "two_speed", "uniform_cosine", "fixed", "one_cycle"
        }:
            raise ValueError(
                "v1-beta optimized training requires "
                "KRONOS_SCHEDULER=warmup_cosine, two_speed, uniform_cosine, fixed, or one_cycle"
            )
        self.scheduler_warmup_ratio = float(
            os.getenv("KRONOS_SCHEDULER_WARMUP_RATIO", "0.02")
        )
        if not 0 < self.scheduler_warmup_ratio < 1:
            raise ValueError("KRONOS_SCHEDULER_WARMUP_RATIO must be in (0, 1)")
        self.predictor_warmup_start_learning_rate = float(
            os.getenv("KRONOS_PREDICTOR_WARMUP_START_LR", "1e-6")
        )
        self.condition_warmup_start_learning_rate = float(
            os.getenv("KRONOS_CONDITION_WARMUP_START_LR", "1e-5")
        )
        if not 0 < self.predictor_warmup_start_learning_rate <= self.predictor_learning_rate:
            raise ValueError("Predictor warmup start LR must be in (0, predictor LR]")
        if not 0 < self.condition_warmup_start_learning_rate <= self.condition_learning_rate:
            raise ValueError("Condition warmup start LR must be in (0, condition LR]")
        self.condition_fast_decay_ratio = float(
            os.getenv("KRONOS_CONDITION_FAST_DECAY_RATIO", "0.075")
        )
        self.condition_fast_decay_learning_rate = float(
            os.getenv("KRONOS_CONDITION_FAST_DECAY_LR", "1e-5")
        )
        if (
            self.scheduler_type == "two_speed"
            and not self.scheduler_warmup_ratio < self.condition_fast_decay_ratio < 1
        ):
            raise ValueError(
                "Condition fast-decay ratio must be greater than warmup ratio and less than 1"
            )
        if self.scheduler_type == "two_speed" and not (
            self.condition_min_learning_rate
            <= self.condition_fast_decay_learning_rate
            <= self.condition_learning_rate
        ):
            raise ValueError(
                "Condition fast-decay LR must be between condition min and peak LR"
            )
        if not 0 < self.predictor_min_learning_rate <= self.predictor_learning_rate:
            raise ValueError("Predictor min LR must be in (0, predictor LR]")
        if not 0 < self.condition_min_learning_rate <= self.condition_learning_rate:
            raise ValueError("Condition min LR must be in (0, condition LR]")
        if self.scheduler_type == "uniform_cosine":
            uniform_values = (
                (
                    "peak LR",
                    self.predictor_learning_rate,
                    self.condition_learning_rate,
                ),
                (
                    "warmup start LR",
                    self.predictor_warmup_start_learning_rate,
                    self.condition_warmup_start_learning_rate,
                ),
                (
                    "minimum LR",
                    self.predictor_min_learning_rate,
                    self.condition_min_learning_rate,
                ),
            )
            for label, predictor_value, condition_value in uniform_values:
                if predictor_value != condition_value:
                    raise ValueError(
                        f"uniform_cosine requires identical predictor and condition {label}"
                    )
        self.gradient_clip_norm = float(
            os.getenv("KRONOS_GRAD_CLIP_NORM", "3.0")
        )
        if self.gradient_clip_norm <= 0:
            raise ValueError("KRONOS_GRAD_CLIP_NORM must be positive")
        self.condition_monitor_interval_steps = int(
            os.getenv("KRONOS_CONDITION_MONITOR_INTERVAL_STEPS", "100")
        )
        self.condition_ablation_interval_segments = int(
            os.getenv("KRONOS_CONDITION_ABLATION_INTERVAL_SEGMENTS", "10")
        )
        if self.condition_monitor_interval_steps < 0:
            raise ValueError("Condition monitor interval must be non-negative")
        if self.condition_ablation_interval_segments < 0:
            raise ValueError("Condition ablation interval must be non-negative")
        self.trainable_transformer_layers = int(
            os.getenv("KRONOS_TRAINABLE_TRANSFORMER_LAYERS", "2")
        )
        self.context_layer = int(os.getenv("KRONOS_CONTEXT_LAYER", "10"))

        # Gradient accumulation to simulate a larger batch size.
        self.accumulation_steps = 1

        # AdamW optimizer parameters.
        self.adam_beta1 = 0.9
        self.adam_beta2 = 0.95
        self.adam_weight_decay = float(
            os.getenv("KRONOS_ADAM_WEIGHT_DECAY", "0.1")
        )

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
        self.pretrained_tokenizer_path = os.getenv(
            "KRONOS_TOKENIZER_PATH", "NeoQuasar/Kronos-Tokenizer-base"
        )
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
