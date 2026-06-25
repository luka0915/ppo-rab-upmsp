class Config:
    # ----- random seeds -----
    TRAIN_SEED = 3
    TEST_SEED = 3

    # ----- PPO hyperparameters (paper, Section 5.2) -----
    NUM_EPISODES = 1000        # N_E
    GAMMA = 0.99               # discount factor
    GAE_LAMBDA = 0.98          # lambda

    ACTOR_LR = 5e-5            # actor learning rate
    CRITIC_LR = 2e-4           # critic learning rate

    BATCH_SIZE = 2048          # N_B (PPO update is triggered when |D| >= N_B)
    NUM_MINI_BATCHES = 64      # N_B is split into 64 mini-batches
    UPDATE_EPOCHS = 16         # N_U (number of update epochs per iteration)

    EPSILON = 0.15             # PPO clipping parameter
    TEMPERATURE = 0.1          # tau (softmax temperature, Eq. 22)
    ETA = 0.015                # adaptive-batching tolerance (Algorithm 1)

    # Training (Algorithm 2, line 10) and test/scheduling (Algorithm 4) both use
    # critic-based adaptive batching (Algorithm 1).

    # standard PPO gradient-clipping stabilizer (not specified in the paper;
    # set to None to disable)
    MAX_GRAD_NORM = 0.5

    # ----- scheduling (test) -----
    TIME_LIMIT = 300           # 5-minute time limit for PPO-RAB

    # Parallelize the per-step rollout over above-average action candidates.
    # Default OFF: PPO-RAB then runs single-process, matching the (single-process)
    # baselines (SA / RLGA) for a fair comparison under the same time
    # limit. Set True only for fast development runs. (The original submission
    # effectively ran with this ON using cpu_count()-1 workers.)
    USE_MULTIPROCESSING = True
    NUM_WORKERS = None         # None -> cpu_count() - 1 at runtime

    # ----- data paths (relative to the project root) -----
    # Training uses a SINGLE instance file (paper: one D1 instance is used to
    # train the agent). Testing uses a FOLDER: every Example_*.xlsx inside it is
    # evaluated with the trained, size-independent model.
    TRAIN_DATA_PATH = 'datasets/D1/Example_1.xlsx'
    # TRAIN_DATA_PATH = ''
    # A single folder/file OR a list of dataset folders. Each folder is evaluated
    # into its own results subfolder (named after the folder, e.g. D2/D3/D4).
    # Set to None to fall back to dataset.py TEST_* data.
    TEST_DATA_PATH = ['datasets/D2/', 'datasets/D3/', 'datasets/D4/']

    # ----- I/O -----
    MODEL_DIR = 'models/PPO_RAB/'      # where training writes best/checkpoint/final models
    RESULTS_DIR = 'results/PPO_RAB/'
    SAVE_INTERVAL = 100        # save a checkpoint every 100 episodes
    PLOT_INTERVAL = 100        # refresh training plots/metrics every N episodes

    # Trained checkpoint to evaluate at test time (PPO/test.py). Point this at any
    # .pt produced during training: best_model.pt, final_model.pt, or a specific
    # checkpoint_<N>.pt (e.g. the one with the lowest validation MWT).
    TEST_MODEL_PATH = 'models/PPO_RAB/best_model.pt'
