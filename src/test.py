import re
import argparse
import json
import os
import copy
import time
from datetime import datetime
import logging
from tqdm import *

from utils.parse_config import ConfigParser
from utils.functions import *
from agent import *
from environment.portfolio_env import PortfolioEnv


def test(func_args):
    if func_args.seed != -1:
        setup_seed(func_args.seed)

    data_prefix = func_args.data_prefix
    matrix_path = data_prefix + func_args.relation_file


    if func_args.market == 'DJIA':
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
        allow_short = True
    elif func_args.market == 'TWII':
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
        allow_short = True
    elif func_args.market == 'HSI':
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load(data_prefix + 'ror.npy')
        market_history = np.load(data_prefix + 'market_data.npy')
        assert stocks_data.shape[:-1] == rate_of_return.shape, 'file size error'
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        test_idx = 4211
        allow_short = True

    elif func_args.market == 'CSI100':
        stocks_data = np.load(data_prefix + 'stocks_data.npy')
        rate_of_return = np.load(data_prefix + 'ror.npy')
        A = torch.from_numpy(np.load(matrix_path)).float().to(func_args.device)
        test_idx = 1944
        market_history = None
        allow_short = False

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
        logger=None
        )
    
    PREFIX = func_args.prefix
    best_model_path = os.path.join(PREFIX, 'model_file')
    model_sort = sorted(
        [x for x in os.listdir(best_model_path) if "best_cr" in x],
        key=lambda s: int(re.search(r'\d+', s).group())
    )
    model_sort = [x for x in model_sort if "best_cr" in x]
    best_model = model_sort[-1]
    print("best_model_path: ", best_model_path)
    print("best_model: ", best_model)

    actor = torch.load(os.path.join(best_model_path, best_model), weights_only=False)
    actor.eval()
    agent = RLAgent(env, actor, func_args)

    try:
        agent_wealth, rho_record, portfolio_records = agent.test()
        npy_save_dir = os.path.join(PREFIX, 'npy_file')
        np.save(os.path.join(npy_save_dir, 'agent_wealth_test.npy'), agent_wealth)
        
        metrics = calculate_metrics(agent_wealth, func_args.trade_mode)
        
        test_results = {
            'agent_wealth': agent_wealth.tolist(),
            'rho_record': [convert_to_native_type(r) for r in rho_record],
            'portfolio_records': convert_portfolio_records_to_json(
                portfolio_records,
                start_idx=test_idx,
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
            }
        }
        
        test_results_dir = os.path.join(PREFIX, 'json_file')
        os.makedirs(test_results_dir, exist_ok=True)
        json_file = os.path.join(test_results_dir, 'test_results.json')
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(test_results, f, indent=2, ensure_ascii=False)
        
        print("Test completed successfully!")
        print(f"Results saved to: {json_file}")
        print(f"Total steps: {test_results['summary']['total_steps']}")
        print(f"Final wealth: {test_results['summary']['final_wealth']:.4f}")
        print(f"Total return: {test_results['summary']['total_return']:.2%}")

    
    except KeyboardInterrupt:
        pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config', type=str)
    parser.add_argument('--window_len', type=int)
    parser.add_argument('--G', type=int)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--lr', type=float)
    parser.add_argument('--gamma', type=float)
    # parser.add_argument('--no_spatial', dest='spatial_bool', action='store_false')
    parser.add_argument('--no_msu', dest='msu_bool', action='store_false')
    parser.add_argument('--relation_file', type=str)
    parser.add_argument('--addaptiveadj', dest='addaptive_adj_bool', action='store_false')
    parser.add_argument('--no_tfinasu', dest='transformer_asu_bool', action='store_false', default=None)
    parser.add_argument('--no_tfinmsu', dest='transformer_msu_bool', action='store_false', default=None)
    parser.add_argument('--prefix', type=str, help='Experiment output directory prefix')

    opts = parser.parse_args()
    
    if opts.prefix:
        PREFIX = opts.prefix
    else:
        PREFIX = os.path.join("outputs", "0904", "013935")

    if opts.config is not None:
        with open(opts.config) as f:
            options = json.load(f)
            args = ConfigParser(options)
    else:
        hyper_json_path = os.path.join(PREFIX, 'log_file', 'hyper.json')
        with open(hyper_json_path) as f:
            options = json.load(f)
            args = ConfigParser(options)
    args.prefix = PREFIX
    args.update(opts)

    test(args)
