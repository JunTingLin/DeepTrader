#!/usr/bin/env python3
"""
Test script for the unified plotting script
"""

import sys
import os
sys.path.append(os.path.dirname(__file__))

def test_market_configurations():
    """Test that market configurations are properly set up"""
    print("Testing market configurations...")
    
    try:
        import plot_unified as pu
        
        # Test Taiwan market (default)
        print(f"\nDefault market: {pu.CURRENT_MARKET}")
        print(f"Start date: {pu.START_DATE}")
        print(f"End date: {pu.END_DATE}")
        print(f"Benchmark: {pu.config['benchmark_label']}")
        print(f"Experiment IDs count: {len(pu.EXPERIMENT_IDS)}")
        print(f"Stock symbols count: {len(pu.get_stock_symbols())}")
        
        # Test switching to US market
        pu.set_market('US')
        print(f"\nAfter switching to US:")
        print(f"Current market: {pu.CURRENT_MARKET}")
        print(f"Start date: {pu.START_DATE}")
        print(f"End date: {pu.END_DATE}")
        print(f"Benchmark: {pu.config['benchmark_label']}")
        print(f"Experiment IDs count: {len(pu.EXPERIMENT_IDS)}")
        print(f"Stock symbols count: {len(pu.get_stock_symbols())}")
        
        # Test switching back to Taiwan
        pu.set_market('TW')
        print(f"\nAfter switching back to TW:")
        print(f"Current market: {pu.CURRENT_MARKET}")
        print(f"Start date: {pu.START_DATE}")
        print(f"End date: {pu.END_DATE}")
        print(f"Benchmark: {pu.config['benchmark_label']}")
        print(f"Experiment IDs count: {len(pu.EXPERIMENT_IDS)}")
        print(f"Stock symbols count: {len(pu.get_stock_symbols())}")
        
        # Test invalid market code
        try:
            pu.set_market('INVALID')
            print("ERROR: Should have raised an exception for invalid market")
            return False
        except ValueError as e:
            print(f"\nCorrectly handled invalid market code: {e}")
        
        print("\n✅ All configuration tests passed!")
        return True
        
    except ImportError as e:
        print(f"❌ Failed to import plot_unified: {e}")
        return False
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
        return False

def test_function_signatures():
    """Test that all required functions exist and have proper signatures"""
    print("\nTesting function signatures...")
    
    try:
        import plot_unified as pu
        
        required_functions = [
            'get_stock_symbols',
            'load_agent_wealth', 
            'get_business_day_segments',
            'get_market_data',
            'compute_cumulative_wealth',
            'process_data',
            'plot_results',
            'plot_yearly_results',
            'calculate_periodic_returns_df',
            'calculate_win_rate_df',
            'compute_metrics_df',
            'plot_portfolio_heatmap',
            'set_market',
            'main'
        ]
        
        missing_functions = []
        for func_name in required_functions:
            if not hasattr(pu, func_name):
                missing_functions.append(func_name)
        
        if missing_functions:
            print(f"❌ Missing functions: {missing_functions}")
            return False
        else:
            print(f"✅ All {len(required_functions)} required functions found!")
            return True
            
    except Exception as e:
        print(f"❌ Function signature test failed: {e}")
        return False

def test_market_config_completeness():
    """Test that both market configurations have all required fields"""
    print("\nTesting market configuration completeness...")
    
    try:
        import plot_unified as pu
        
        required_fields = [
            'name', 'start_date', 'end_date', 'market_file', 'stock_symbols',
            'benchmark_column', 'benchmark_label', 'title', 'train_end',
            'val_end', 'test_end', 'experiment_ids', 'plot_ylim'
        ]
        
        for market_code in ['TW', 'US']:
            config = pu.MARKET_CONFIGS[market_code]
            missing_fields = []
            
            for field in required_fields:
                if field not in config:
                    missing_fields.append(field)
            
            if missing_fields:
                print(f"❌ Market {market_code} missing fields: {missing_fields}")
                return False
            else:
                print(f"✅ Market {market_code} configuration complete")
        
        return True
        
    except Exception as e:
        print(f"❌ Configuration completeness test failed: {e}")
        return False

def test_stock_symbols():
    """Test that stock symbols are properly loaded"""
    print("\nTesting stock symbols...")
    
    try:
        import plot_unified as pu
        
        # Test TW symbols
        pu.set_market('TW')
        tw_symbols = pu.get_stock_symbols()
        if len(tw_symbols) != 49:
            print(f"❌ Expected 49 TW symbols, got {len(tw_symbols)}")
            return False
        if not all(symbol.endswith('.TW') for symbol in tw_symbols):
            print("❌ Not all TW symbols end with .TW")
            return False
        print(f"✅ TW symbols: {len(tw_symbols)} symbols, all ending with .TW")
        
        # Test US symbols
        pu.set_market('US')
        us_symbols = pu.get_stock_symbols()
        if len(us_symbols) != 30:
            print(f"❌ Expected 30 US symbols, got {len(us_symbols)}")
            return False
        if any('.' in symbol for symbol in us_symbols):
            print("❌ US symbols should not contain dots")
            return False
        print(f"✅ US symbols: {len(us_symbols)} symbols, no dots")
        
        return True
        
    except Exception as e:
        print(f"❌ Stock symbols test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("=" * 50)
    print("TESTING UNIFIED PLOTTING SCRIPT")
    print("=" * 50)
    
    tests = [
        test_market_configurations,
        test_function_signatures,
        test_market_config_completeness,
        test_stock_symbols
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"TEST SUMMARY: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The unified script looks good.")
        return True
    else:
        print("❌ Some tests failed. Please check the issues above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)