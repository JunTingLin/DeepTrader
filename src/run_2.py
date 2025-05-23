import argparse
import json
import os
import copy
import time
from datetime import datetime
import logging
from tqdm import *

from torch.utils.tensorboard import SummaryWriter

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv

def run(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = './data/' + func_args.market + '/'
    matrix_path = data_prefix + func_args.relation_file

    start_time = datetime.now().strftime('%m%d/%H%M%S')

    PREFIX = os.path.join('outputs_2', start_time)
    img_dir = os.path.join(PREFIX, 'img_file')
    save_dir = os.path.join(PREFIX, 'log_file')
    model_save_dir = os.path.join(PREFIX, 'model_file')
    npy_save_dir = os.path.join(PREFIX, 'npy_file')

    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(img_dir):
        os.makedirs(img_dir)
    if not os.path.isdir(model_save_dir):
        os.mkdir(model_save_dir)
    if not os.path.isdir(npy_save_dir):
        os.mkdir(npy_save_dir)

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
    chlr.setLevel('INFO')
    fhlr = logging.FileHandler(os.path.join(save_dir, 'logger.log'))
    fhlr.setFormatter(formatter)
    logger.addHandler(chlr)
    logger.addHandler(fhlr)

    if func_args.market == 'DJIA':
        stocks_data = np.load(os.path.join(data_prefix, 'stocks_data.npy'))
        rate_of_return = np.load(os.path.join(data_prefix, 'ror.npy'))
        market_history = np.load(os.path.join(data_prefix, 'market_data.npy'))
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        train_idx = func_args.train_idx
        train_idx_end = func_args.train_idx_end
        val_idx = func_args.val_idx
        test_idx = func_args.test_idx
        test_idx_end = func_args.test_idx_end  # 新增測試結束指標
        allow_short = True
    elif func_args.market == 'HSI':
        pass
    elif func_args.market == 'CSI100':
        pass

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
        logger=logger
    )

    supports = [A]
    actor = RLActor(supports, func_args).to(func_args.device)
    agent = RLAgent(env, actor, func_args)

    # 使用外層滾動更新循環
    cycle = 0
    best_global_CR = -float('inf')
    # 這裡使用 func_args.epochs 作為每個 cycle 內訓練的 epoch 數
    epochs_per_cycle = epochs_per_cycle = func_args.epochs
    while True:
        logger.warning("Starting cycle %d: train_idx=%d, val_idx=%d, test_idx=%d, test_idx_end=%d" %
                       (cycle, env.src.train_idx, env.src.val_idx, env.src.test_idx, env.src.test_idx_end))
        best_cycle_CR = -float('inf')
        # 在每個 cycle 內訓練若干 epoch
        for epoch in range(epochs_per_cycle):
            epoch_return = 0
            epoch_loss = 0
            mini_batch_num = int(np.ceil(len(env.src.order_set) / func_args.batch_size))
            for j in tqdm(range(mini_batch_num)):
                episode_return, avg_rho, avg_mdd, episode_loss = agent.train_episode()
                epoch_return += episode_return
                epoch_loss += episode_loss
            avg_train_return = epoch_return / mini_batch_num
            avg_epoch_loss = epoch_loss / mini_batch_num
            logger.warning('[Cycle %d] round %d, avg train return %.4f, avg rho %.4f, avg mdd %.4f, avg loss %.4f' %
                           (cycle, epoch, avg_train_return, avg_rho, avg_mdd, avg_epoch_loss))
            writer.add_scalar('Cycle_{}/Train/Loss'.format(cycle), avg_epoch_loss, global_step=epoch)
            writer.add_scalar('Cycle_{}/Train/Return'.format(cycle), avg_train_return, global_step=epoch)
            writer.add_scalar('Cycle_{}/Train/Rho'.format(cycle), avg_rho, global_step=epoch)
            writer.add_scalar('Cycle_{}/Train/MDD'.format(cycle), avg_mdd, global_step=epoch)

            agent_wealth_val = agent.evaluation()
            metrics = calculate_metrics(agent_wealth_val, func_args.trade_mode)
            writer.add_scalar('Cycle_{}/Val/APR'.format(cycle), metrics['APR'], global_step=epoch)
            writer.add_scalar('Cycle_{}/Val/MDD'.format(cycle), metrics['MDD'], global_step=epoch)
            writer.add_scalar('Cycle_{}/Val/AVOL'.format(cycle), metrics['AVOL'], global_step=epoch)
            writer.add_scalar('Cycle_{}/Val/ASR'.format(cycle), metrics['ASR'], global_step=epoch)
            writer.add_scalar('Cycle_{}/Val/SoR'.format(cycle), metrics['DDR'], global_step=epoch)
            writer.add_scalar('Cycle_{}/Val/CR'.format(cycle), metrics['CR'], global_step=epoch)

            if metrics['CR'] > best_cycle_CR:
                best_cycle_CR = metrics['CR']
                # 保存該cycel周期最佳模型
                torch.save(actor, os.path.join(model_save_dir, f'best_cr_cycle{cycle}_epoch{epoch}.pkl'))
                np.save(os.path.join(npy_save_dir, f'agent_wealth_val_cycle{cycle}.npy'), agent_wealth_val)

            logger.warning('[Cycle %d] After epoch %d, max wealth: %.4f, min wealth: %.4f, avg wealth: %.4f, final wealth: %.4f, APR: %.3f%%, ASR: %.3f, AVol: %.3f, MDD: %.2f%%, CR: %.3f, DDR: %.3f'
                           % (cycle, epoch, max(agent_wealth_val[0]), min(agent_wealth_val[0]), np.mean(agent_wealth_val),
                              agent_wealth_val[-1, -1], 100 * metrics['APR'], metrics['ASR'], metrics['AVOL'],
                              100 * metrics['MDD'], metrics['CR'], metrics['DDR']))

        logger.warning("Cycle %d finished. Best CR in cycle: %.3f" % (cycle, best_cycle_CR))

        agent_wealth_test = agent.test()
        np.save(os.path.join(npy_save_dir, f'agent_wealth_test_cycle{cycle}.npy'), agent_wealth_test)
        metrics_test = calculate_metrics(agent_wealth_test, func_args.trade_mode)
        writer.add_scalar('Cycle_{}/Test/APR'.format(cycle), metrics_test['APR'])
        writer.add_scalar('Cycle_{}/Test/MDD'.format(cycle), metrics_test['MDD'])
        writer.add_scalar('Cycle_{}/Test/CR'.format(cycle), metrics_test['CR'])
        logger.warning("Cycle %d TEST metrics: APR=%.3f, MDD=%.2f%%, CR=%.3f" %
                       (cycle, 100 * metrics_test['APR'], 100 * metrics_test['MDD'], metrics_test['CR']))
        
        # 更新全局最佳模型（可根據需求）
        if best_cycle_CR > best_global_CR:
            best_global_CR = best_cycle_CR

        # 檢查是否還有足夠資料進行下一個 cycle
        if env.src.test_idx_end + func_args.trade_len < stocks_data.shape[1]:
            env.roll_update()
            cycle += 1
        else:
            logger.warning("Reached end of available data. Stopping training.")
            break

    logger.warning("Training finished. Global best CR: %.3f" % best_global_CR)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int, default=-1)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false')

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
