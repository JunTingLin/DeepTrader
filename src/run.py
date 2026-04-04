import argparse
import json
import os
import copy
import time
import re
from datetime import datetime
import logging
from tqdm import *

from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv


def load_pretrained_msu_encoder(actor, checkpoint_path, logger):
    """
    Load pretrained encoder (TE_1D) from Stage 1 PMSU checkpoint to Stage 2 MSU.

    This enables two-stage training:
    - Stage 1: Pretrain MSU encoder with masked reconstruction (self-supervised)
    - Stage 2: Fine-tune with RL (use pretrained encoder weights)

    Args:
        actor: RLActor instance containing the MSU module
        checkpoint_path: Path to PMSU checkpoint (best_model.pth from Stage 1)
        logger: Logger instance for logging
    """
    from model.PMSU import PMSU

    # Check if MSU is enabled
    if not actor.args.msu_bool:
        logger.warning('MSU is disabled (msu_bool=False), skipping pretrained encoder loading.')
        return

    logger.warning('='*80)
    logger.warning('Loading Pretrained MSU Encoder (Stage 1 → Stage 2)')
    logger.warning('='*80)
    logger.warning(f'Checkpoint path: {checkpoint_path}')

    # Load PMSU checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=actor.args.device, weights_only=False)
    market_features = checkpoint.get('market_features', 27)

    logger.warning(f'Stage 1 training info:')
    logger.warning(f'  Epoch:           {checkpoint["epoch"]}')
    logger.warning(f'  Best val loss:   {checkpoint.get("best_val_loss", "N/A"):.6f}')
    logger.warning(f'  Market features: {market_features}')
    logger.warning(f'  Mask ratio:      {checkpoint.get("mask_ratio", "N/A")}')

    # Initialize PMSU temporarily to extract encoder
    pmsu = PMSU(in_features=market_features)
    pmsu.load_state_dict(checkpoint['model_state_dict'])

    # Extract encoder state dict (only TE_1D, not decoder)
    encoder_state_dict = pmsu.get_encoder_state_dict()

    # Load into MSU's TE_1D
    missing_keys, unexpected_keys = actor.msu.TE_1D.load_state_dict(
        encoder_state_dict['TE_1D'], strict=True
    )

    if len(missing_keys) == 0 and len(unexpected_keys) == 0:
        logger.warning('✅ Successfully loaded pretrained encoder to MSU.TE_1D')
    else:
        logger.warning(f'⚠️  Loading completed with issues:')
        logger.warning(f'   Missing keys: {missing_keys}')
        logger.warning(f'   Unexpected keys: {unexpected_keys}')

    logger.warning('='*80)
    logger.warning('Pretrained encoder loaded. Starting Stage 2 RL training...')
    logger.warning('='*80)


def load_and_freeze_asu(actor, checkpoint_path, logger):
    """
    Load pretrained ASU model and freeze its parameters.

    This enables MSU-only RL training:
    - ASU is frozen (no gradient updates)
    - Only MSU parameters are trained
    - Stock selection remains fixed, only rho allocation is learned

    Args:
        actor: RLActor instance containing the ASU module
        checkpoint_path: Path to pretrained ASU model (e.g., best_cr-xxx.pkl)
        logger: Logger instance for logging
    """
    import glob

    logger.warning('='*80)
    logger.warning('Loading and Freezing ASU (MSU-only RL Training Mode)')
    logger.warning('='*80)

    # Handle directory path - find the best model file
    if os.path.isdir(checkpoint_path):
        model_dir = os.path.join(checkpoint_path, 'model_file')
        if os.path.isdir(model_dir):
            # Find the best_cr-*.pkl file with highest epoch
            pkl_files = sorted(glob.glob(os.path.join(model_dir, 'best_cr-*.pkl')))
            if pkl_files:
                checkpoint_path = pkl_files[-1]  # Get the latest best model
                logger.warning(f'Found best model: {checkpoint_path}')
            else:
                raise FileNotFoundError(f'No best_cr-*.pkl files found in {model_dir}')
        else:
            raise FileNotFoundError(f'model_file directory not found in {checkpoint_path}')

    logger.warning(f'Loading pretrained ASU from: {checkpoint_path}')

    # Load the pretrained model
    pretrained_actor = torch.load(checkpoint_path, map_location=actor.args.device, weights_only=False)

    # Copy ASU weights from pretrained model to current actor
    actor.asu.load_state_dict(pretrained_actor.asu.state_dict())
    logger.warning('✅ Successfully loaded pretrained ASU weights')

    # Freeze ASU parameters
    frozen_params = 0
    for param in actor.asu.parameters():
        param.requires_grad = False
        frozen_params += param.numel()

    # Lock ASU in eval mode to prevent BatchNorm from updating running statistics
    # This uses the ASU class's built-in freeze_eval_mode() method which is pickleable
    actor.asu.freeze_eval_mode()

    logger.warning(f'✅ Frozen {frozen_params:,} ASU parameters')
    logger.warning(f'✅ ASU locked in eval mode (BatchNorm stats frozen)')

    # Count trainable parameters (should only be MSU)
    trainable_params = sum(p.numel() for p in actor.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in actor.parameters())

    logger.warning(f'   Total parameters:     {total_params:,}')
    logger.warning(f'   Trainable parameters: {trainable_params:,} (MSU only)')
    logger.warning(f'   Frozen parameters:    {total_params - trainable_params:,} (ASU)')
    logger.warning('='*80)
    logger.warning('ASU frozen. Only MSU will be trained via RL.')
    logger.warning('='*80)


def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = func_args.data_prefix
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H%M%S')

    outputs_base_path = getattr(func_args, 'outputs_base_path', './outputs')
    PREFIX = os.path.join(outputs_base_path, start_time)
    print(f"[DEEPTRADER_PREFIX] {PREFIX}")  # Clear output PREFIX for script extraction
    img_dir = os.path.join(PREFIX, 'img_file')
    save_dir = os.path.join(PREFIX, 'log_file')
    model_save_dir = os.path.join(PREFIX, 'model_file')
    npy_save_dir = os.path.join(PREFIX, 'npy_file')
    json_save_dir = os.path.join(PREFIX, 'json_file')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.isdir(npy_save_dir):
        os.mkdir(npy_save_dir)
    if not os.path.isdir(json_save_dir):
        os.makedirs(json_save_dir)

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

    if func_args.market == 'DJIA':
        logger.info('using DJIA data')
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load( data_prefix + 'ror.npy')
        market_history = np.load(data_prefix + 'market_data.npy')
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        train_idx = func_args.train_idx
        train_idx_end = func_args.train_idx_end
        val_idx = func_args.val_idx
        test_idx = func_args.test_idx
        test_idx_end = func_args.test_idx_end
        allow_short = getattr(func_args, 'allow_short', True)

        # Load news embeddings (Path B) if enabled
        if news_embedding_bool:
            sentiment_data_path = getattr(func_args, 'sentiment_data_path', 'src/data/DJIA/sentiment/')
            news_embedding_dim = getattr(func_args, 'news_embedding_dim', 768)
            news_embedding_zeroed = getattr(func_args, 'news_embedding_zeroed', False)
            # Build filename based on dimension and zeroed flag
            if news_embedding_zeroed:
                if news_embedding_dim != 768:
                    news_embeddings_file = os.path.join(sentiment_data_path, f'cls_embeddings_zeroed_dim{news_embedding_dim}.npy')
                else:
                    news_embeddings_file = os.path.join(sentiment_data_path, 'cls_embeddings_zeroed.npy')
            else:
                if news_embedding_dim != 768:
                    news_embeddings_file = os.path.join(sentiment_data_path, f'cls_embeddings_dim{news_embedding_dim}.npy')
                else:
                    news_embeddings_file = os.path.join(sentiment_data_path, 'cls_embeddings.npy')
            if os.path.exists(news_embeddings_file):
                news_embeddings = np.load(news_embeddings_file)
                logger.info(f'Loaded news embeddings: {news_embeddings.shape} (zeroed={news_embedding_zeroed})')
            else:
                logger.warning(f'News embeddings file not found: {news_embeddings_file}')
                logger.warning('Disabling news_embedding_bool')
                news_embedding_bool = False
    elif func_args.market == 'TWII':
        logger.info('using TWII data')
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load( data_prefix + 'ror.npy')
        market_history = np.load(data_prefix + 'market_data.npy')
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        train_idx = func_args.train_idx
        train_idx_end = func_args.train_idx_end
        val_idx = func_args.val_idx
        test_idx = func_args.test_idx
        test_idx_end = func_args.test_idx_end
        allow_short = getattr(func_args, 'allow_short', True)

        # Load news embeddings (Path B) if enabled
        news_embedding_bool = getattr(func_args, 'news_embedding_bool', False)
        if news_embedding_bool:
            sentiment_data_path = getattr(func_args, 'sentiment_data_path', 'src/data/TWII/sentiment/')
            news_embedding_dim = getattr(func_args, 'news_embedding_dim', 768)
            news_embedding_zeroed = getattr(func_args, 'news_embedding_zeroed', False)
            # Build filename based on dimension and zeroed flag
            if news_embedding_zeroed:
                if news_embedding_dim != 768:
                    news_embeddings_file = os.path.join(sentiment_data_path, f'cls_embeddings_zeroed_dim{news_embedding_dim}.npy')
                else:
                    news_embeddings_file = os.path.join(sentiment_data_path, 'cls_embeddings_zeroed.npy')
            else:
                if news_embedding_dim != 768:
                    news_embeddings_file = os.path.join(sentiment_data_path, f'cls_embeddings_dim{news_embedding_dim}.npy')
                else:
                    news_embeddings_file = os.path.join(sentiment_data_path, 'cls_embeddings.npy')
            if os.path.exists(news_embeddings_file):
                news_embeddings = np.load(news_embeddings_file)
                logger.info(f'Loaded news embeddings: {news_embeddings.shape} (zeroed={news_embedding_zeroed})')
            else:
                logger.warning(f'News embeddings file not found: {news_embeddings_file}')
                logger.warning('Disabling news_embedding_bool')
                news_embedding_bool = False
                news_embeddings = None
        else:
            news_embeddings = None
    elif func_args.market == 'HSI':
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load(data_prefix + 'ror.npy')
        market_history = np.load(data_prefix + 'market_data.npy')
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        test_idx = 4211
        allow_short = getattr(func_args, 'allow_short', True)
    elif func_args.market == 'CSI100':
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load(data_prefix + 'ror.npy')
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        test_idx = 1944
        market_history = None
        allow_short = getattr(func_args, 'allow_short', False)

    # Filter assets by indices if specified
    asset_indices = getattr(func_args, 'asset_indices', None)
    if asset_indices is not None:
        asset_indices = np.array(asset_indices)
        logger.info(f'Filtering assets by indices: {asset_indices}')
        logger.info(f'Original data shape: stocks_data={stocks_data.shape}, ror={rate_of_return.shape}')

        # Filter stocks_data: (num_assets, num_days, num_features) -> filtered
        stocks_data = stocks_data[asset_indices]
        # Filter rate_of_return: (num_assets, num_days) -> filtered
        rate_of_return = rate_of_return[asset_indices]
        # Filter correlation matrix A: (num_assets, num_assets) -> filtered
        A = A[asset_indices][:, asset_indices]
        # Filter news_embeddings if present: (num_assets, num_days, embedding_dim) -> filtered
        if news_embeddings is not None:
            news_embeddings = news_embeddings[asset_indices]

        # Update num_assets in func_args
        func_args.num_assets = len(asset_indices)
        logger.info(f'Filtered data shape: stocks_data={stocks_data.shape}, ror={rate_of_return.shape}')
        logger.info(f'Updated num_assets: {func_args.num_assets}')

    env = PortfolioEnv(
        assets_data=stocks_data,
        market_data=market_history,
        rtns_data=rate_of_return,
        in_features=func_args.in_features,
        train_idx=train_idx,
        train_idx_end=train_idx_end,
        val_idx=val_idx,
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

    # Resume from checkpoint or create new actor
    resume_epoch = 0
    if hasattr(func_args, 'resume') and func_args.resume is not None:
        logger.warning('=' * 80)
        logger.warning(f'RESUMING from checkpoint: {func_args.resume}')
        actor = torch.load(func_args.resume, map_location=func_args.device, weights_only=False)

        # Auto-detect epoch from filename
        match = re.search(r'(\d+)', os.path.basename(func_args.resume))
        if match:
            resume_epoch = int(match.group(1))

        logger.warning(f'Resuming from epoch {resume_epoch}')
        logger.warning('=' * 80)
    else:
        actor = RLActor(supports, func_args).to(func_args.device)

        # Load pretrained MSU encoder if specified (Stage 1 → Stage 2 transfer learning)
        if hasattr(func_args, 'pretrained_msu_path') and func_args.pretrained_msu_path is not None:
            load_pretrained_msu_encoder(actor, func_args.pretrained_msu_path, logger)
        else:
            logger.warning('Training MSU from scratch (no pretrained encoder specified)')

    # Load and freeze ASU if specified (MSU-only RL training mode)
    freeze_asu = getattr(func_args, 'freeze_asu', False)
    if freeze_asu:
        pretrained_asu_path = getattr(func_args, 'pretrained_asu_path', None)
        if pretrained_asu_path is None:
            raise ValueError('--freeze_asu requires --pretrained_asu_path to be specified')
        if not func_args.msu_bool:
            raise ValueError('--freeze_asu requires --msu_bool to be True (MSU must be enabled)')
        load_and_freeze_asu(actor, pretrained_asu_path, logger)

    agent = RLAgent(env, actor, func_args, freeze_asu=freeze_asu)

    mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
    try:
        max_cr = -999
        start_checkpoint_epoch = func_args.start_checkpoint_epoch
        for epoch in range(resume_epoch, func_args.epochs):
            epoch_return = 0
            epoch_loss = 0
            epoch_loss_asu = 0
            epoch_loss_msu = 0
            for j in tqdm(range(mini_batch_num)):
                episode_return, avg_rho, avg_mdd, episode_loss, episode_loss_asu, episode_loss_msu = agent.train_episode()
                epoch_return += episode_return
                epoch_loss += episode_loss
                epoch_loss_asu += episode_loss_asu
                epoch_loss_msu += episode_loss_msu
            avg_train_return = epoch_return / mini_batch_num
            avg_epoch_loss = epoch_loss / mini_batch_num
            avg_epoch_loss_asu = epoch_loss_asu / mini_batch_num
            avg_epoch_loss_msu = epoch_loss_msu / mini_batch_num
            logger.warning('[%s]round %d, avg train return %.4f, avg rho %.4f, avg mdd %.4f, avg loss %.4f (ASU: %.4f, MSU: %.4f)' %
                            (start_time, epoch, avg_train_return, avg_rho, avg_mdd, avg_epoch_loss, avg_epoch_loss_asu, avg_epoch_loss_msu))
            writer.add_scalar('Train/Loss', avg_epoch_loss, global_step=epoch)
            writer.add_scalar('Train/Loss_ASU', avg_epoch_loss_asu, global_step=epoch)
            writer.add_scalar('Train/Loss_MSU', avg_epoch_loss_msu, global_step=epoch)
            writer.add_scalar('Train/Return', avg_train_return, global_step=epoch)
            writer.add_scalar('Train/Rho', avg_rho, global_step=epoch)
            writer.add_scalar('Train/MDD', avg_mdd, global_step=epoch)

            agent_wealth, rho_record, param1_record, param2_record, portfolio_records, gate_record = agent.evaluation()
            logger.warning('agent_wealth: %s' % agent_wealth)
            logger.warning('agent_wealth shape: %s', agent_wealth.shape)
            metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
            writer.add_scalar('Val/ARR', metrics['ARR'], global_step=epoch)
            writer.add_scalar('Val/MDD', metrics['MDD'], global_step=epoch)
            writer.add_scalar('Val/AVOL', metrics['AVOL'], global_step=epoch)
            writer.add_scalar('Val/ASR', metrics['ASR'], global_step=epoch)
            writer.add_scalar('Val/SoR', metrics['DDR'], global_step=epoch)
            writer.add_scalar('Val/CR', metrics['CR'], global_step=epoch)

            # Log gate statistics if available
            if gate_record and gate_record[0] is not None:
                gate_array = np.array([g for g in gate_record if g is not None])
                writer.add_scalar('Val/Gate_Mean', gate_array.mean(), global_step=epoch)
                writer.add_scalar('Val/Gate_Std', gate_array.std(), global_step=epoch)

            writer.flush() # flush the writer

            # Log validation metrics every epoch
            logger.info('Epoch %d - Validation: Final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, '
                        'AVOL: %.3f, MDD: %.2f%%, CR: %.3f, DDR: %.3f, Avg Rho: %.3f'
                        % (epoch, agent_wealth[0, -1], metrics['ARR'] * 100, metrics['ASR'],
                           metrics['AVOL'], metrics['MDD'] * 100, metrics['CR'], metrics['DDR'],
                           np.mean(rho_record)))

            # Save periodic checkpoint every N epochs (for later analysis)
            checkpoint_interval = getattr(func_args, 'checkpoint_interval', 100)
            if checkpoint_interval > 0 and epoch > 0 and epoch % checkpoint_interval == 0:
                periodic_checkpoint_path = os.path.join(model_save_dir, f'epoch-{epoch}.pkl')
                torch.save(actor, periodic_checkpoint_path)
                logger.info(f'Saved periodic checkpoint: epoch-{epoch}.pkl')

            if epoch >= start_checkpoint_epoch and metrics['CR'] > max_cr:
                print('New Best CR Policy!!!!')
                max_cr = metrics['CR']
                torch.save(actor, os.path.join(model_save_dir, 'best_cr-'+str(epoch)+'.pkl'))
                np.save(os.path.join(npy_save_dir, 'agent_wealth_val.npy'), agent_wealth)
                
                # Determine parameter names based on distribution type
                distribution_type = getattr(func_args, 'msu_distribution_type', 'normal').lower()
                if distribution_type == 'beta':
                    param1_name, param2_name = 'alpha_record', 'beta_record'
                else:  # normal
                    param1_name, param2_name = 'mu_record', 'sigma_record'

                # Process gate_record
                gate_record_json = None
                gate_summary = None
                if gate_record and gate_record[0] is not None:
                    gate_record_json = [g.tolist() if g is not None else None for g in gate_record]
                    gate_array = np.array([g for g in gate_record if g is not None])
                    gate_summary = {
                        'overall_mean': float(gate_array.mean()),
                        'per_stock_mean': gate_array.mean(axis=0).tolist(),
                        'per_step_mean': gate_array.mean(axis=1).tolist(),
                        'shape': list(gate_array.shape)
                    }

                val_results = {
                    'agent_wealth': agent_wealth.tolist(),
                    'rho_record': [convert_to_native_type(r) for r in rho_record],
                    param1_name: [convert_to_native_type(r) if r is not None else None for r in param1_record],
                    param2_name: [convert_to_native_type(r) if r is not None else None for r in param2_record],
                    'distribution_type': distribution_type,
                    'gate_record': gate_record_json,
                    'gate_summary': gate_summary,
                    'portfolio_records': convert_portfolio_records_to_json(
                        portfolio_records, 
                        start_idx=val_idx, 
                        window_len=func_args.window_len, 
                        trade_len=func_args.trade_len
                    ),
                    'performance_metrics': {
                        'ARR': convert_to_native_type(metrics['ARR']),
                        'MDD': convert_to_native_type(metrics['MDD']),
                        'AVOL': convert_to_native_type(metrics['AVOL']),
                        'ASR': convert_to_native_type(metrics['ASR']),
                        'DDR': convert_to_native_type(metrics['DDR']),
                        'CR': convert_to_native_type(metrics['CR'])
                    },
                    'summary': {
                        'total_steps': len(portfolio_records),
                        'agent_wealth_shape': list(agent_wealth.shape),
                        'final_wealth': convert_to_native_type(agent_wealth[0, -1]),
                        'total_return': convert_to_native_type(agent_wealth[0, -1] - 1.0)
                    },
                    'epoch': epoch
                }
                
                val_json_file = os.path.join(json_save_dir, 'val_results.json')
                with open(val_json_file, 'w', encoding='utf-8') as f:
                    json.dump(val_results, f, indent=2, ensure_ascii=False)
                
            logger.warning('after training %d round, max wealth: %.4f, min wealth: %.4f,'
                            ' avg wealth: %.4f, final wealth: %.4f, ARR: %.3f%%, ASR: %.3f, AVol" %.3f,'
                            'MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                            % (
                                epoch, max(agent_wealth[0]), min(agent_wealth[0]), np.mean(agent_wealth),
                                agent_wealth[-1, -1], 100 * metrics['ARR'], metrics['ASR'], metrics['AVOL'],
                                100 * metrics['MDD'], metrics['CR'], metrics['DDR']
                            ))
    except KeyboardInterrupt:
        torch.save(actor, os.path.join(model_save_dir, 'final_model.pkl'))
        torch.save(agent.optimizer.state_dict(), os.path.join(model_save_dir, 'final_optimizer.pkl'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false', default=None)
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false', default=None)
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false', default=None)
    parser.add_argument('--no_tfinasu', dest='transformer_asu_bool', action='store_false', default=None)
    parser.add_argument('--no_tfinmsu', dest='transformer_msu_bool', action='store_false', default=None)
    parser.add_argument('--pretrained_msu_path', type=str, default=None,
                        help='Path to pretrained MSU encoder checkpoint from Stage 1 (e.g., checkpoints/msu_stage1_masked/.../best_model.pth)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training (e.g., ./src/outputs/0313/015755/model_file/epoch-1400.pkl)')
    parser.add_argument('--freeze_asu', type=lambda x: x.lower() == 'true', default=None,
                        help='Freeze ASU parameters and only train MSU (requires --pretrained_asu_path). Use --freeze_asu true or set in JSON.')
    parser.add_argument('--pretrained_asu_path', type=str, default=None,
                        help='Path to pretrained ASU model for freeze_asu mode (e.g., src/outputs/0308/210009/model_file/best_cr-xxx.pkl)')

    opts = parser.parse_args()

    if opts.config is not None:
        with open(opts.config) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        with open('./hyper.json') as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.update(opts)

    run(args)
