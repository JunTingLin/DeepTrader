import math
import random

import numpy as np
import torch

switch2days = {'D': 1, 'W': 5, 'M': 21}


def convert_to_native_type(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        # If it's a single element array, extract the value
        if obj.size == 1:
            return float(obj.flatten()[0])
        else:
            return obj.tolist()
    elif hasattr(obj, 'item'):
        return obj.item()
    elif isinstance(obj, list) and len(obj) == 1:
        # Handle nested lists
        return convert_to_native_type(obj[0])
    else:
        return obj


def convert_portfolio_records_to_json(portfolio_records, start_idx=None, window_len=None, trade_len=None):
    """Convert portfolio_records to JSON serializable format"""
    json_portfolio_records = []
    
    for step_idx, portfolio_info in enumerate(portfolio_records):
        step_data = {
            'step': step_idx + 1,
            'long_positions': [],
            'short_positions': [],
            'all_scores': portfolio_info['all_scores'][0].tolist()  # 只取第一個 batch
        }
        
        # 添加 input_indices 和 predict_indices 資訊
        if start_idx is not None and window_len is not None and trade_len is not None:
            # 每個step的cursor向前跳躍trade_len天
            cursor = start_idx + step_idx * trade_len
            
            # 觀察窗口：cursor - (window_len + 1) * 5 + 1 : cursor + 1 (含頭不含尾)
            input_start = cursor - (window_len + 1) * 5 + 1
            input_end = cursor + 1  # Python 切片風格，不包含此 index
            
            # 預測窗口：cursor + 1 : cursor + trade_len + 1 (含頭不含尾) 
            predict_start = cursor + 1
            predict_end = cursor + trade_len + 1  # Python 切片風格，不包含此 index
            
            step_data['input_start'] = input_start
            step_data['input_end'] = input_end
            step_data['predict_start'] = predict_start
            step_data['predict_end'] = predict_end
        
        # 只處理第一個 batch 的資料
        batch_idx = 0
        
        # 處理做多倉位
        long_positions = []
        for pos_idx in range(len(portfolio_info['long_indices'][batch_idx])):
            stock_idx = int(portfolio_info['long_indices'][batch_idx][pos_idx])
            weight = float(portfolio_info['long_weights'][batch_idx][pos_idx])
            score = float(portfolio_info['all_scores'][batch_idx][stock_idx])
            long_positions.append({
                'stock_index': stock_idx,
                'weight': weight,
                'score': score
            })
        step_data['long_positions'] = long_positions
        
        # 處理做空倉位
        short_positions = []
        for pos_idx in range(len(portfolio_info['short_indices'][batch_idx])):
            stock_idx = int(portfolio_info['short_indices'][batch_idx][pos_idx])
            weight = float(portfolio_info['short_weights'][batch_idx][pos_idx])
            score = float(portfolio_info['all_scores'][batch_idx][stock_idx])
            short_positions.append({
                'stock_index': stock_idx,
                'weight': weight,
                'score': score
            })
        step_data['short_positions'] = short_positions
        
        json_portfolio_records.append(step_data)
    
    return json_portfolio_records


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def calculate_metrics(agent_wealth, trade_mode, MAR=0.):
    """
    Based on metric descriptions at AlphaStock
    """
    trade_ror = agent_wealth[:, 1:] / agent_wealth[:, :-1] - 1
    if agent_wealth.shape[0] == trade_ror.shape[0] == 1:
        agent_wealth = agent_wealth.flatten()
    trade_periods = trade_ror.shape[-1]
    if trade_mode == 'D':
        Ny = 251
    elif trade_mode == 'W':
        Ny = 50
    elif trade_mode == 'M':
        Ny = 12
    else:
        assert ValueError, 'Please check the trading mode'

    AT = np.mean(trade_ror, axis=-1, keepdims=True)
    VT = np.std(trade_ror, axis=-1, keepdims=True)

    # ARR = 平均報酬 × 年化因子
    ARR = AT * Ny
    AVOL = VT * math.sqrt(Ny)
    ASR = ARR / AVOL
    drawdown = (np.maximum.accumulate(agent_wealth, axis=-1) - agent_wealth) /\
                     np.maximum.accumulate(agent_wealth, axis=-1)
    MDD = np.max(drawdown, axis=-1)
    CR = ARR / MDD

    tmp1 = np.sum(((np.clip(MAR-trade_ror, 0., math.inf))**2), axis=-1) / \
           np.sum(np.clip(MAR-trade_ror, 0., math.inf)>0)
    downside_deviation = np.sqrt(tmp1)
    DDR = ARR / downside_deviation

    metrics = {
        'ARR': ARR,
        'AVOL': AVOL,
        'ASR': ASR,
        'MDD': MDD,
        'CR': CR,
        'DDR': DDR
    }

    return metrics