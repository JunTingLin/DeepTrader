"""
Sliding Window Training for DeepTrader

This script implements rolling window training where the model continuously learns
from new data by shifting the training/validation/test windows forward.

Supports two modes:
- "expanding": Training window grows (train_idx stays at 0, train_idx_end increases)
- "fixed": Training window size stays constant (all indices shift forward)

Each cycle:
1. Train for N epochs (more for first cycle, fewer for fine-tuning)
2. Evaluate on validation set, save best model
3. Test on test set (typically 21 days = 1 month)
4. Shift windows forward by trade_len days
5. Fine-tune from previous best model (not from scratch)
6. Repeat until data is exhausted

Usage:
    python run_2.py -c hyper_sliding.json
    python run_2.py -c hyper_sliding.json --initial_epochs 500 --finetune_epochs 100
"""

import argparse
import json
import os
import copy
from datetime import datetime
import logging
from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = func_args.data_prefix
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H%M%S')

    # Use sliding window specific output directory
    outputs_base_path = getattr(func_args, 'outputs_base_path', './outputs')
    PREFIX = os.path.join(outputs_base_path + '_sliding', start_time)
    print(f"[DEEPTRADER_PREFIX] {PREFIX}")
    img_dir = os.path.join(PREFIX, 'img_file')
    save_dir = os.path.join(PREFIX, 'log_file')
    model_save_dir = os.path.join(PREFIX, 'model_file')
    npy_save_dir = os.path.join(PREFIX, 'npy_file')
    json_save_dir = os.path.join(PREFIX, 'json_file')

    for d in [save_dir, img_dir, model_save_dir, npy_save_dir, json_save_dir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    hyper = copy.deepcopy(func_args.__dict__)
    print(hyper)
    hyper['device'] = 'cuda' if hyper['device'] == torch.device('cuda') else 'cpu'
    json_str = json.dumps(hyper, indent=4)

    with open(os.path.join(save_dir, 'hyper.json'), 'w') as json_file:
        json_file.write(json_str)

    writer = SummaryWriter(save_dir)
    writer.add_text('hyper_setting', str(hyper))

    logger = logging.getLogger()
    logger.setLevel('INFO')
    BASIC_FORMAT = "%(asctime)s:%(levelname)s:%(message)s"
    DATE_FORMAT = '%Y-%m-%d %H%M%S'
    formatter = logging.Formatter(BASIC_FORMAT, DATE_FORMAT)
    chlr = logging.StreamHandler()
    chlr.setFormatter(formatter)
    chlr.setLevel('WARNING')
    fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    # Initialize news embeddings variable
    news_embeddings = None
    news_embedding_bool = getattr(func_args, 'news_embedding_bool', False)

    # Load data based on market
    if func_args.market == 'DJIA':
        logger.info('Using DJIA data')
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load(data_prefix + 'ror.npy')
        market_history = np.load(data_prefix + 'market_data.npy')
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        train_idx, train_idx_end, val_idx, val_idx_end, test_idx, test_idx_end = resolve_split_indices(func_args)
        allow_short = getattr(func_args, 'allow_short', True)

        # Load news embeddings (Path B) if enabled
        if news_embedding_bool:
            sentiment_data_path = getattr(func_args, 'sentiment_data_path', 'src/data/DJIA/sentiment/')
            news_embeddings_file = os.path.join(sentiment_data_path, 'cls_embeddings.npy')
            if os.path.exists(news_embeddings_file):
                news_embeddings = np.load(news_embeddings_file)
                logger.info(f'Loaded news embeddings: {news_embeddings.shape}')
            else:
                logger.warning(f'News embeddings file not found: {news_embeddings_file}')
                logger.warning('Disabling news_embedding_bool')
                news_embedding_bool = False

    elif func_args.market == 'TWII':
        logger.info('Using TWII data')
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load(data_prefix + 'ror.npy')
        market_history = np.load(data_prefix + 'market_data.npy')
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        train_idx, train_idx_end, val_idx, val_idx_end, test_idx, test_idx_end = resolve_split_indices(func_args)
        allow_short = getattr(func_args, 'allow_short', True)

        # Load news embeddings (Path B) if enabled
        if news_embedding_bool:
            sentiment_data_path = getattr(func_args, 'sentiment_data_path', 'src/data/TWII/sentiment/')
            news_embeddings_file = os.path.join(sentiment_data_path, 'cls_embeddings.npy')
            if os.path.exists(news_embeddings_file):
                news_embeddings = np.load(news_embeddings_file)
                logger.info(f'Loaded news embeddings: {news_embeddings.shape}')
            else:
                logger.warning(f'News embeddings file not found: {news_embeddings_file}')
                logger.warning('Disabling news_embedding_bool')
                news_embedding_bool = False

    elif func_args.market in ['HSI', 'CSI100']:
        logger.info(f'Using {func_args.market} data')
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load(data_prefix + 'ror.npy')
        market_history = np.load(data_prefix + 'market_data.npy')
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        train_idx, train_idx_end, val_idx, val_idx_end, test_idx, test_idx_end = resolve_split_indices(func_args)
        allow_short = getattr(func_args, 'allow_short', True)
        # News embeddings not supported for HSI/CSI100 yet
        news_embedding_bool = False

    else:
        raise ValueError(f"Unknown market: {func_args.market}")

    # Filter assets by indices if specified
    asset_indices = getattr(func_args, 'asset_indices', None)
    if asset_indices is not None:
        asset_indices = np.array(asset_indices)
        logger.info(f'Filtering assets by indices: {asset_indices}')
        logger.info(f'Original data shape: stocks_data={stocks_data.shape}, ror={rate_of_return.shape}')

        stocks_data = stocks_data[asset_indices]
        rate_of_return = rate_of_return[asset_indices]
        A = A[asset_indices][:, asset_indices]
        if news_embeddings is not None:
            news_embeddings = news_embeddings[asset_indices]

        func_args.num_assets = len(asset_indices)
        logger.info(f'Filtered data shape: stocks_data={stocks_data.shape}, ror={rate_of_return.shape}')
        logger.info(f'Updated num_assets: {func_args.num_assets}')

    # Sliding window configuration
    sliding_mode = getattr(func_args, 'sliding_mode', 'expanding')  # 'expanding' or 'fixed'
    initial_epochs = getattr(func_args, 'initial_epochs', 200)
    finetune_epochs = getattr(func_args, 'finetune_epochs', 30)
    finetune_lr_decay = getattr(func_args, 'finetune_lr_decay', 0.5)
    base_lr = func_args.lr

    logger.warning("=" * 70)
    logger.warning("SLIDING WINDOW TRAINING MODE")
    logger.warning("=" * 70)
    logger.warning(f"Market: {func_args.market}")
    logger.warning(f"Data shape: stocks={stocks_data.shape}, ror={rate_of_return.shape}")
    logger.warning(f"Sliding mode: {sliding_mode}")
    logger.warning(f"Initial epochs (Cycle 0): {initial_epochs}")
    logger.warning(f"Fine-tune epochs (Cycle 1+): {finetune_epochs}")
    logger.warning(f"Fine-tune LR decay: {finetune_lr_decay}x")
    logger.warning(f"Window shift per cycle: {func_args.trade_len} days")
    logger.warning(
        f"Initial indices: train=[{train_idx}, {train_idx_end}), "
        f"val=[{val_idx}, {val_idx_end}), test=[{test_idx}, {test_idx_end})"
    )
    if news_embedding_bool:
        logger.warning(f"News embedding: ENABLED (shape={news_embeddings.shape})")
        logger.warning(f"  Fusion method: {getattr(func_args, 'fusion_method', 'concat')}")
        logger.warning(f"  Aggregation: {getattr(func_args, 'news_aggregation', 'mean')}")
    else:
        logger.warning("News embedding: DISABLED")
    logger.warning("=" * 70)

    env = PortfolioEnv(
        assets_data=stocks_data,
        market_data=market_history,
        rtns_data=rate_of_return,
        in_features=func_args.in_features,
        train_idx=train_idx,
        train_idx_end=train_idx_end,
        val_idx=val_idx,
        val_idx_end=val_idx_end,
        test_idx=test_idx,
        test_idx_end=test_idx_end,
        batch_size=func_args.batch_size,
        window_len=func_args.window_len,
        trade_len=func_args.trade_len,
        max_steps=func_args.max_steps,
        norm_type=func_args.norm_type,
        allow_short=allow_short,
        logger=logger,
        news_embeddings=news_embeddings,
        news_embedding_bool=news_embedding_bool
    )

    supports = [A]
    actor = RLActor(supports, func_args).to(func_args.device)
    agent = RLAgent(env, actor, func_args)

    # Check for pre-trained Cycle 0 model
    pretrained_cycle0_path = getattr(func_args, 'pretrained_cycle0_path', None)
    pretrained_model_file = None

    if pretrained_cycle0_path:
        import glob
        # Find the model file
        if os.path.isfile(pretrained_cycle0_path):
            pretrained_model_file = pretrained_cycle0_path
        else:
            model_dir = os.path.join(pretrained_cycle0_path, 'model_file')
            if os.path.isdir(model_dir):
                pkl_files = glob.glob(os.path.join(model_dir, 'best_cr-*.pkl'))
                pth_files = glob.glob(os.path.join(model_dir, 'best_cr*.pth'))
                if pkl_files:
                    pkl_files.sort(key=lambda x: int(x.split('-')[-1].replace('.pkl', '')))
                    pretrained_model_file = pkl_files[-1]
                elif pth_files:
                    pretrained_model_file = pth_files[0]

        if pretrained_model_file and os.path.exists(pretrained_model_file):
            logger.warning("=" * 70)
            logger.warning("PRE-TRAINED CYCLE 0 MODEL DETECTED")
            logger.warning(f"  Path: {pretrained_model_file}")
            logger.warning("  Cycle 0 training will be SKIPPED")
            logger.warning("=" * 70)
        else:
            logger.warning(f"WARNING: pretrained_cycle0_path specified but model not found")
            pretrained_model_file = None

    # Sliding window training loop
    cycle = 0
    best_global_CR = -float('inf')
    global_epoch = 0

    # Store results for all cycles
    all_cycle_results = []

    # Store optimizer state for fine-tuning
    best_optimizer_state = None

    while True:
        # Check if using pretrained model for Cycle 0
        use_pretrained = (cycle == 0 and pretrained_model_file is not None)

        # Determine epochs for this cycle
        if use_pretrained:
            epochs_this_cycle = 0  # Skip training
            current_lr = base_lr
        elif cycle == 0:
            epochs_this_cycle = initial_epochs
            current_lr = base_lr
        else:
            epochs_this_cycle = finetune_epochs
            current_lr = base_lr * finetune_lr_decay

        logger.warning("=" * 70)
        logger.warning(f"CYCLE {cycle} STARTING")
        if use_pretrained:
            logger.warning(f"  Mode: Using pre-trained model (SKIPPING TRAINING)")
            logger.warning(f"  Pre-trained: {pretrained_model_file}")
        else:
            logger.warning(f"  Mode: {'From scratch' if cycle == 0 else 'Fine-tuning'}")
            logger.warning(f"  Epochs: {epochs_this_cycle}")
        logger.warning(f"  Learning rate: {current_lr}")
        logger.warning(f"  train_idx: [{env.src.train_idx}, {env.src.train_idx_end})")
        logger.warning(f"  val_idx: [{env.src.val_idx}, {env.src.val_idx_end})")
        logger.warning(f"  test_idx: [{env.src.test_idx}, {env.src.test_idx_end})")
        logger.warning("=" * 70)

        # For fine-tuning cycles, update learning rate
        if cycle > 0:
            for param_group in agent.optimizer.param_groups:
                param_group['lr'] = current_lr

        best_cycle_CR = -float('inf')
        best_cycle_epoch = -1
        best_cycle_metrics = None  # Store all validation metrics for the best model

        # Handle pre-trained model for Cycle 0
        if use_pretrained:
            # Load pre-trained model (use weights_only=False for compatibility with old .pkl files)
            loaded = torch.load(pretrained_model_file, weights_only=False, map_location=func_args.device)
            # Handle both state_dict and full model formats
            if hasattr(loaded, 'state_dict'):
                actor.load_state_dict(loaded.state_dict())
            else:
                actor.load_state_dict(loaded)

            # Run validation to get metrics
            agent_wealth_val, rho_record, param1_record, param2_record, portfolio_records = agent.evaluation()
            metrics_val = calculate_metrics(agent_wealth_val, func_args.trade_mode)
            best_cycle_CR = float(metrics_val['CR'])
            best_cycle_epoch = -1  # Indicates pre-trained
            best_cycle_metrics = copy.deepcopy(metrics_val)  # Store all metrics

            logger.warning(f"Cycle 0 (pretrained) Validation: CR={best_cycle_CR:.3f}, ARR={100*float(metrics_val['ARR']):.2f}%")

            # Save model and results
            torch.save(actor.state_dict(), os.path.join(model_save_dir, 'best_cr_cycle0.pth'))
            torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'best_optimizer_cycle0.pth'))
            best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())
            np.save(os.path.join(npy_save_dir, 'agent_wealth_val_cycle0.npy'), agent_wealth_val)

            # Save validation results JSON
            distribution_type = getattr(func_args, 'msu_distribution_type', 'normal').lower()
            param1_name = 'alpha_record' if distribution_type == 'beta' else 'mu_record'
            param2_name = 'beta_record' if distribution_type == 'beta' else 'sigma_record'

            val_results = {
                'cycle': 0, 'epoch': -1, 'source': 'pretrained',
                'pretrained_path': pretrained_model_file,
                'agent_wealth': agent_wealth_val.tolist(),
                'rho_record': [convert_to_native_type(r) for r in rho_record],
                param1_name: [convert_to_native_type(r) if r is not None else None for r in param1_record],
                param2_name: [convert_to_native_type(r) if r is not None else None for r in param2_record],
                'distribution_type': distribution_type,
                'performance_metrics': {
                    'ARR': convert_to_native_type(metrics_val['ARR']),
                    'MDD': convert_to_native_type(metrics_val['MDD']),
                    'AVOL': convert_to_native_type(metrics_val['AVOL']),
                    'ASR': convert_to_native_type(metrics_val['ASR']),
                    'DDR': convert_to_native_type(metrics_val['DDR']),
                    'CR': convert_to_native_type(metrics_val['CR'])
                },
                'window_info': {
                    'train_idx': env.src.train_idx, 'train_idx_end': env.src.train_idx_end,
                    'val_idx': env.src.val_idx, 'val_idx_end': env.src.val_idx_end,
                    'test_idx': env.src.test_idx, 'test_idx_end': env.src.test_idx_end
                }
            }
            with open(os.path.join(json_save_dir, 'val_results_cycle0.json'), 'w', encoding='utf-8') as f:
                json.dump(val_results, f, indent=2, ensure_ascii=False)

        # Train for epochs_this_cycle epochs in this cycle (skipped if epochs_this_cycle == 0)
        mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))

        for epoch in range(epochs_this_cycle):
            epoch_return = 0
            epoch_loss = 0
            epoch_loss_asu = 0
            epoch_loss_msu = 0

            for j in tqdm(range(mini_batch_num), desc=f"Cycle {cycle} Epoch {epoch}"):
                episode_return, avg_rho, avg_mdd, episode_loss, episode_loss_asu, episode_loss_msu = agent.train_episode()
                epoch_return += episode_return
                epoch_loss += episode_loss
                epoch_loss_asu += episode_loss_asu
                epoch_loss_msu += episode_loss_msu

            avg_train_return = epoch_return / mini_batch_num
            avg_epoch_loss = epoch_loss / mini_batch_num
            avg_epoch_loss_asu = epoch_loss_asu / mini_batch_num
            avg_epoch_loss_msu = epoch_loss_msu / mini_batch_num

            logger.warning('[Cycle %d][Epoch %d] avg return %.4f, avg rho %.4f, avg mdd %.4f, loss %.4f (ASU: %.4f, MSU: %.4f)' %
                           (cycle, epoch, avg_train_return, avg_rho, avg_mdd, avg_epoch_loss, avg_epoch_loss_asu, avg_epoch_loss_msu))

            # TensorBoard logging
            writer.add_scalar(f'Cycle_{cycle}/Train/Loss', avg_epoch_loss, global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Train/Loss_ASU', avg_epoch_loss_asu, global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Train/Loss_MSU', avg_epoch_loss_msu, global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Train/Return', avg_train_return, global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Train/Rho', avg_rho, global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Train/MDD', avg_mdd, global_step=epoch)

            # Global tracking
            writer.add_scalar('Global/Train/Loss', avg_epoch_loss, global_step=global_epoch)
            writer.add_scalar('Global/Train/Return', avg_train_return, global_step=global_epoch)

            # Validation
            agent_wealth_val, rho_record, param1_record, param2_record, portfolio_records = agent.evaluation()
            metrics = calculate_metrics(agent_wealth_val, func_args.trade_mode)

            writer.add_scalar(f'Cycle_{cycle}/Val/ARR', metrics['ARR'], global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Val/MDD', metrics['MDD'], global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Val/AVOL', metrics['AVOL'], global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Val/ASR', metrics['ASR'], global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Val/SoR', metrics['DDR'], global_step=epoch)
            writer.add_scalar(f'Cycle_{cycle}/Val/CR', metrics['CR'], global_step=epoch)
            writer.add_scalar('Global/Val/CR', metrics['CR'], global_step=global_epoch)
            writer.flush()

            # Save best model for this cycle
            cr_value = float(metrics['CR'])
            start_checkpoint_epoch = getattr(func_args, 'start_checkpoint_epoch', 0)

            # For cycle 0, only save checkpoint after start_checkpoint_epoch to avoid early unstable models
            if cycle == 0:
                can_save = (epoch >= start_checkpoint_epoch) and (cr_value > best_cycle_CR)
            else:
                # For fine-tune cycles (only ~10 epochs), no start epoch restriction
                can_save = cr_value > best_cycle_CR

            if can_save:
                best_cycle_CR = cr_value
                best_cycle_epoch = epoch
                best_cycle_metrics = copy.deepcopy(metrics)  # Store all validation metrics

                # Save model
                torch.save(actor.state_dict(), os.path.join(model_save_dir, f'best_cr_cycle{cycle}.pth'))
                # Save optimizer state for fine-tuning
                torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, f'best_optimizer_cycle{cycle}.pth'))
                best_optimizer_state = copy.deepcopy(agent.optimizer.state_dict())

                np.save(os.path.join(npy_save_dir, f'agent_wealth_val_cycle{cycle}.npy'), agent_wealth_val)

                # Save detailed results
                distribution_type = getattr(func_args, 'msu_distribution_type', 'normal').lower()
                if distribution_type == 'beta':
                    param1_name, param2_name = 'alpha_record', 'beta_record'
                else:
                    param1_name, param2_name = 'mu_record', 'sigma_record'

                val_results = {
                    'cycle': cycle,
                    'epoch': epoch,
                    'learning_rate': current_lr,
                    'agent_wealth': agent_wealth_val.tolist(),
                    'rho_record': [convert_to_native_type(r) for r in rho_record],
                    param1_name: [convert_to_native_type(r) if r is not None else None for r in param1_record],
                    param2_name: [convert_to_native_type(r) if r is not None else None for r in param2_record],
                    'distribution_type': distribution_type,
                    'performance_metrics': {
                        'ARR': convert_to_native_type(metrics['ARR']),
                        'MDD': convert_to_native_type(metrics['MDD']),
                        'AVOL': convert_to_native_type(metrics['AVOL']),
                        'ASR': convert_to_native_type(metrics['ASR']),
                        'DDR': convert_to_native_type(metrics['DDR']),
                        'CR': convert_to_native_type(metrics['CR'])
                    },
                    'window_info': {
                        'train_idx': env.src.train_idx,
                        'train_idx_end': env.src.train_idx_end,
                        'val_idx': env.src.val_idx,
                        'val_idx_end': env.src.val_idx_end,
                        'test_idx': env.src.test_idx,
                        'test_idx_end': env.src.test_idx_end
                    }
                }

                with open(os.path.join(json_save_dir, f'val_results_cycle{cycle}.json'), 'w', encoding='utf-8') as f:
                    json.dump(val_results, f, indent=2, ensure_ascii=False)

            logger.warning('[Cycle %d][Epoch %d] Val: wealth=%.4f, ARR=%.3f%%, ASR=%.3f, MDD=%.2f%%, CR=%.3f, DDR=%.3f'
                           % (cycle, epoch, agent_wealth_val[-1, -1], 100 * metrics['ARR'], metrics['ASR'],
                              100 * metrics['MDD'], metrics['CR'], metrics['DDR']))

            global_epoch += 1

        # Cycle finished - load best model and run test
        logger.warning("=" * 50)
        logger.warning(f"CYCLE {cycle} COMPLETED - Running Test")
        logger.warning(f"Best validation CR in cycle: {best_cycle_CR:.3f} (epoch {best_cycle_epoch})")
        logger.warning("=" * 50)

        # Load best model for testing
        best_model_path = os.path.join(model_save_dir, f'best_cr_cycle{cycle}.pth')
        if os.path.exists(best_model_path):
            actor.load_state_dict(torch.load(best_model_path, weights_only=True))

        agent_wealth_test, rho_test, param1_test, param2_test, portfolio_test = agent.test()
        np.save(os.path.join(npy_save_dir, f'agent_wealth_test_cycle{cycle}.npy'), agent_wealth_test)

        metrics_test = calculate_metrics(agent_wealth_test, func_args.trade_mode)

        # Handle single-step metrics (ASR/AVOL may be inf/nan with only 1 trading step)
        def safe_metric(value):
            v = convert_to_native_type(value)
            if v is None or (isinstance(v, float) and (np.isnan(v) or np.isinf(v))):
                return None
            return v

        # Only log finite values
        if np.isfinite(metrics_test['ARR']):
            writer.add_scalar(f'Cycle_{cycle}/Test/ARR', metrics_test['ARR'])
        if np.isfinite(metrics_test['MDD']):
            writer.add_scalar(f'Cycle_{cycle}/Test/MDD', metrics_test['MDD'])
        if np.isfinite(metrics_test['CR']):
            writer.add_scalar(f'Cycle_{cycle}/Test/CR', metrics_test['CR'])
            writer.add_scalar('Global/Test/CR', metrics_test['CR'], global_step=cycle)

        # Calculate simple return for single-step case
        test_return = agent_wealth_test[-1, -1] - 1.0 if agent_wealth_test.shape[-1] > 1 else 0.0

        logger.warning(f"CYCLE {cycle} TEST: wealth={agent_wealth_test[-1, -1]:.4f}, return={100*test_return:.2f}%, "
                       f"MDD={100*metrics_test['MDD']:.2f}%")

        # Save test results (format matching test.py output)
        distribution_type = getattr(func_args, 'msu_distribution_type', 'normal').lower()
        if distribution_type == 'beta':
            param1_name, param2_name = 'alpha_record', 'beta_record'
        else:
            param1_name, param2_name = 'mu_record', 'sigma_record'

        test_results = {
            'cycle': cycle,
            'best_val_epoch': best_cycle_epoch,
            'best_val_CR': convert_to_native_type(best_cycle_CR),
            'agent_wealth': agent_wealth_test.tolist(),
            'rho_record': [convert_to_native_type(r) for r in rho_test],
            param1_name: [convert_to_native_type(r) if r is not None else None for r in param1_test],
            param2_name: [convert_to_native_type(r) if r is not None else None for r in param2_test],
            'distribution_type': distribution_type,
            'portfolio_records': convert_portfolio_records_to_json(
                portfolio_test,
                start_idx=env.src.test_idx,
                window_len=func_args.window_len,
                trade_len=func_args.trade_len
            ),
            'performance_metrics': {
                'ARR': safe_metric(metrics_test['ARR']),
                'MDD': safe_metric(metrics_test['MDD']),
                'AVOL': safe_metric(metrics_test['AVOL']),
                'ASR': safe_metric(metrics_test['ASR']),
                'DDR': safe_metric(metrics_test['DDR']),
                'CR': safe_metric(metrics_test['CR'])
            },
            'summary': {
                'total_steps': len(portfolio_test),
                'agent_wealth_shape': list(agent_wealth_test.shape),
                'final_wealth': convert_to_native_type(agent_wealth_test[0, -1]),
                'total_return': convert_to_native_type(test_return)
            },
            'window_info': {
                'train_idx': env.src.train_idx,
                'train_idx_end': env.src.train_idx_end,
                'val_idx': env.src.val_idx,
                'val_idx_end': env.src.val_idx_end,
                'test_idx': env.src.test_idx,
                'test_idx_end': env.src.test_idx_end
            }
        }

        with open(os.path.join(json_save_dir, f'test_results_cycle{cycle}.json'), 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)

        all_cycle_results.append({
            'cycle': cycle,
            # Validation metrics (from best model in this cycle)
            'val_CR': convert_to_native_type(best_cycle_metrics['CR']) if best_cycle_metrics else convert_to_native_type(best_cycle_CR),
            'val_ARR': convert_to_native_type(best_cycle_metrics['ARR']) if best_cycle_metrics else None,
            'val_MDD': convert_to_native_type(best_cycle_metrics['MDD']) if best_cycle_metrics else None,
            'val_ASR': convert_to_native_type(best_cycle_metrics['ASR']) if best_cycle_metrics else None,
            'val_AVOL': convert_to_native_type(best_cycle_metrics['AVOL']) if best_cycle_metrics else None,
            'val_DDR': convert_to_native_type(best_cycle_metrics['DDR']) if best_cycle_metrics else None,
            # Test metrics
            'test_CR': convert_to_native_type(metrics_test['CR']),
            'test_ARR': convert_to_native_type(metrics_test['ARR']),
            'test_MDD': convert_to_native_type(metrics_test['MDD']),
            'test_ASR': convert_to_native_type(metrics_test['ASR']),
            'test_AVOL': convert_to_native_type(metrics_test['AVOL']),
            'test_DDR': convert_to_native_type(metrics_test['DDR']),
            'test_wealth': convert_to_native_type(agent_wealth_test[-1, -1])
        })

        # Update global best
        if best_cycle_CR > best_global_CR:
            best_global_CR = best_cycle_CR
            torch.save(actor.state_dict(), os.path.join(model_save_dir, 'best_global.pth'))

        # Check if we can continue to next cycle
        if env.src.test_idx_end + func_args.trade_len < stocks_data.shape[1]:
            # Roll update based on sliding mode
            if sliding_mode == 'expanding':
                # Expanding window: only shift end indices, keep train_idx at 0
                env.src.train_idx_end += func_args.trade_len
                env.src.val_idx += func_args.trade_len
                env.src.val_idx_end += func_args.trade_len
                env.src.test_idx += func_args.trade_len
                env.src.test_idx_end += func_args.trade_len
                # Update order_set for new training range
                env.src.train_set_len = env.src.train_idx_end - env.src.train_idx
                lower_bound = max(env.src.train_idx, 5 * (env.src.window_len + 1) - 1)
                env.src.order_set = np.arange(lower_bound, env.src.train_idx_end - 6 * func_args.trade_len)
                env.src.tmp_order = np.array([])
            else:
                # Fixed window: use default roll_update
                env.roll_update()

            # Continue with the same actor (fine-tuning)
            # Load the best model and optimizer state from this cycle
            actor.load_state_dict(torch.load(best_model_path, weights_only=True))
            agent = RLAgent(env, actor, func_args)

            # Restore optimizer state for continuity
            if best_optimizer_state is not None:
                agent.optimizer.load_state_dict(best_optimizer_state)

            cycle += 1
        else:
            logger.warning("=" * 70)
            logger.warning("REACHED END OF DATA - TRAINING COMPLETE")
            logger.warning("=" * 70)
            break

    # Final summary
    logger.warning("=" * 70)
    logger.warning("SLIDING WINDOW TRAINING SUMMARY")
    logger.warning("=" * 70)
    logger.warning(f"Total cycles: {cycle + 1}")
    logger.warning(f"Total epochs: {global_epoch}")
    logger.warning(f"Sliding mode: {sliding_mode}")
    logger.warning(f"Global best validation CR: {best_global_CR:.3f}")
    logger.warning("")

    # Build cumulative wealth trajectory from all cycles
    # Each cycle's test wealth starts at 1.0 and ends at final_wealth
    # We need to chain them together: W0 * W1 * W2 * ...
    cumulative_wealth_list = [1.0]
    current_wealth = 1.0
    logger.warning("Per-cycle results:")
    for r in all_cycle_results:
        cycle_return = r['test_wealth'] - 1.0
        current_wealth *= r['test_wealth']
        cumulative_wealth_list.append(current_wealth)
        logger.warning(f"  Cycle {r['cycle']}: Val CR={r['val_CR']:.3f}, "
                       f"Test Return={100*cycle_return:.2f}%, "
                       f"Cumulative Wealth={current_wealth:.4f}")

    cumulative_wealth = current_wealth

    # Calculate cumulative metrics using the full trajectory
    cumulative_wealth_array = np.array(cumulative_wealth_list).reshape(1, -1)
    if len(cumulative_wealth_list) > 2:
        cumulative_metrics = calculate_metrics(cumulative_wealth_array, func_args.trade_mode)
        logger.warning("")
        logger.warning(f"Cumulative Metrics (across all {cycle + 1} cycles):")
        logger.warning(f"  Final Wealth: {cumulative_wealth:.4f}")
        logger.warning(f"  Total Return: {(cumulative_wealth - 1) * 100:.2f}%")
        logger.warning(f"  ARR: {100*float(cumulative_metrics['ARR']):.2f}%")
        logger.warning(f"  ASR: {float(cumulative_metrics['ASR']):.3f}")
        logger.warning(f"  MDD: {100*float(cumulative_metrics['MDD']):.2f}%")
        logger.warning(f"  CR: {float(cumulative_metrics['CR']):.3f}")
    else:
        cumulative_metrics = {'ARR': None, 'ASR': None, 'MDD': None, 'CR': None, 'AVOL': None, 'DDR': None}
        logger.warning("")
        logger.warning(f"Cumulative test wealth: {cumulative_wealth:.4f}")
        logger.warning(f"Cumulative return: {(cumulative_wealth - 1) * 100:.2f}%")
        logger.warning("(Not enough cycles to calculate meaningful ASR/CR)")

    logger.warning("=" * 70)

    # Compute averages and standard deviations for validation and test metrics
    def compute_avg_std(values):
        """Compute average and std, handling infinity and None values."""
        # Filter out None and infinity values for avg/std computation
        finite_values = [v for v in values if v is not None and np.isfinite(v)]
        if len(finite_values) == 0:
            return None, None
        avg = float(np.mean(finite_values))
        std = float(np.std(finite_values)) if len(finite_values) > 1 else 0.0
        return avg, std

    val_metrics_keys = ['CR', 'ARR', 'MDD', 'ASR', 'AVOL', 'DDR']
    test_metrics_keys = ['CR', 'ARR', 'MDD', 'ASR', 'AVOL', 'DDR']

    val_metrics_avg = {}
    val_metrics_std = {}
    test_metrics_avg = {}
    test_metrics_std = {}

    for key in val_metrics_keys:
        values = [r.get(f'val_{key}') for r in all_cycle_results]
        avg, std = compute_avg_std(values)
        val_metrics_avg[key] = avg
        val_metrics_std[key] = std

    for key in test_metrics_keys:
        values = [r.get(f'test_{key}') for r in all_cycle_results]
        avg, std = compute_avg_std(values)
        test_metrics_avg[key] = avg
        test_metrics_std[key] = std

    # Log the averages
    logger.warning("")
    logger.warning("Validation Metrics (across all cycles):")
    logger.warning(f"  CR:   avg={val_metrics_avg['CR']:.3f}, std={val_metrics_std['CR']:.3f}" if val_metrics_avg['CR'] else "  CR:   N/A")
    logger.warning(f"  ARR:  avg={100*val_metrics_avg['ARR']:.2f}%, std={100*val_metrics_std['ARR']:.2f}%" if val_metrics_avg['ARR'] else "  ARR:  N/A")
    logger.warning(f"  ASR:  avg={val_metrics_avg['ASR']:.3f}, std={val_metrics_std['ASR']:.3f}" if val_metrics_avg['ASR'] else "  ASR:  N/A")
    logger.warning("")
    logger.warning("Test Metrics (across all cycles):")
    logger.warning(f"  CR:   avg={test_metrics_avg['CR']:.3f}, std={test_metrics_std['CR']:.3f}" if test_metrics_avg['CR'] else "  CR:   N/A (may contain infinity)")
    logger.warning(f"  ARR:  avg={100*test_metrics_avg['ARR']:.2f}%, std={100*test_metrics_std['ARR']:.2f}%" if test_metrics_avg['ARR'] else "  ARR:  N/A")
    logger.warning(f"  MDD:  avg={100*test_metrics_avg['MDD']:.2f}%, std={100*test_metrics_std['MDD']:.2f}%" if test_metrics_avg['MDD'] else "  MDD:  N/A")
    logger.warning("=" * 70)

    # Save overall summary
    summary = {
        'total_cycles': cycle + 1,
        'total_epochs': global_epoch,
        'sliding_mode': sliding_mode,
        'initial_epochs': initial_epochs,
        'finetune_epochs': finetune_epochs,
        'finetune_lr_decay': finetune_lr_decay,
        'global_best_val_CR': convert_to_native_type(best_global_CR),
        'cumulative_test_wealth': convert_to_native_type(cumulative_wealth),
        'cumulative_return': convert_to_native_type(cumulative_wealth - 1),
        'cumulative_wealth_trajectory': [convert_to_native_type(w) for w in cumulative_wealth_list],
        # Cumulative metrics (chaining all test cycles together)
        'cumulative_metrics': {
            'ARR': convert_to_native_type(cumulative_metrics['ARR']) if cumulative_metrics['ARR'] is not None else None,
            'ASR': convert_to_native_type(cumulative_metrics['ASR']) if cumulative_metrics['ASR'] is not None else None,
            'MDD': convert_to_native_type(cumulative_metrics['MDD']) if cumulative_metrics['MDD'] is not None else None,
            'CR': convert_to_native_type(cumulative_metrics['CR']) if cumulative_metrics['CR'] is not None else None,
            'AVOL': convert_to_native_type(cumulative_metrics['AVOL']) if cumulative_metrics.get('AVOL') is not None else None,
            'DDR': convert_to_native_type(cumulative_metrics['DDR']) if cumulative_metrics.get('DDR') is not None else None
        },
        # Validation metrics average and std across all cycles
        'val_metrics_avg': val_metrics_avg,
        'val_metrics_std': val_metrics_std,
        # Test metrics average and std across all cycles (may contain infinity in individual cycles)
        'test_metrics_avg': test_metrics_avg,
        'test_metrics_std': test_metrics_std,
        'cycle_results': all_cycle_results,
        'config': {
            'market': func_args.market,
            'trade_len': func_args.trade_len,
            'window_len': func_args.window_len,
            'base_lr': base_lr
        }
    }

    with open(os.path.join(json_save_dir, 'training_summary.json'), 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='DeepTrader Sliding Window Training')
    parser.add_argument('-c', '--config', type=str, help='Path to config JSON file')
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--initial_epochs', type=int, help='Epochs for first cycle (from scratch)')
    parser.add_argument('--finetune_epochs', type=int, help='Epochs for subsequent cycles (fine-tuning)')
    parser.add_argument('--finetune_lr_decay', type=float, help='LR decay factor for fine-tuning (default: 0.5)')
    parser.add_argument('--sliding_mode', type=str, choices=['expanding', 'fixed'],
                        help='Sliding window mode: expanding (train grows) or fixed (window shifts)')
    parser.add_argument('--pretrained_cycle0_path', type=str,
                        help='Path to pre-trained experiment folder (e.g., ./src/outputs/0223/185007) to skip Cycle 0 training')
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false', default=None)
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false', default=None)
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false', default=None)
    parser.add_argument('--no_tfinasu', dest='transformer_asu_bool', action='store_false', default=None)
    parser.add_argument('--no_tfinmsu', dest='transformer_msu_bool', action='store_false', default=None)

    opts = parser.parse_args()

    if opts.config is not None:
        with open(opts.config) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open('./hyper_sliding.json') as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)

    run(args)
