"""
测试 MCTS 训练流程
"""

import logging
import sys

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)

def test_mcts_basic():
    """测试 MCTS 基本功能"""
    try:
        from gomoku.mcts import MCTS, MCTSNode
        from gomoku import GomokuEnv, GomokuNet
        import numpy as np
        
        logging.info("测试 MCTS 基本功能...")
        
        # 初始化环境和网络
        env = GomokuEnv()
        policy_net = GomokuNet()
        
        # 初始化 MCTS
        mcts = MCTS(policy_net, env, num_simulations=10)
        
        # 测试搜索
        state = env.reset()
        policy, value = mcts.search(state)
        
        logging.info(f"MCTS 搜索成功: policy shape={policy.shape}, value={value}")
        logging.info("MCTS 基本功能测试通过！")
        return True
        
    except Exception as e:
        logging.error(f"MCTS 基本功能测试失败: {e}")
        return False

def test_training_flow_init():
    """测试训练流程初始化"""
    try:
        from gomoku.mcts_train_flow import MCTSTrainFlow
        
        logging.info("测试训练流程初始化...")
        
        # 初始化训练流程
        train_flow = MCTSTrainFlow(mode="init", num_simulations=10)
        
        logging.info("训练流程初始化成功！")
        logging.info(f"策略网络参数数量: {sum(p.numel() for p in train_flow.policy_net.parameters())}")
        logging.info(f"价值网络参数数量: {sum(p.numel() for p in train_flow.value_net.parameters())}")
        
        return True
        
    except Exception as e:
        logging.error(f"训练流程初始化测试失败: {e}")
        return False

def test_data_collection():
    """测试数据收集功能"""
    try:
        from gomoku.mcts_train_flow import MCTSTrainFlow
        
        logging.info("测试数据收集功能...")
        
        # 初始化训练流程
        train_flow = MCTSTrainFlow(mode="init", num_simulations=10)
        
        # 收集少量数据
        train_flow.collect_experience(num_games=2)
        
        if train_flow.replay_buffer is not None:
            buffer_size = len(train_flow.replay_buffer)
            logging.info(f"数据收集成功！回放缓冲区大小: {buffer_size}")
        else:
            logging.info("数据收集成功！")
            
        return True
        
    except Exception as e:
        logging.error(f"数据收集测试失败: {e}")
        return False

def main():
    """主测试函数"""
    logging.info("开始测试 MCTS 训练流程...")
    
    tests = [
        test_mcts_basic,
        test_training_flow_init,
        test_data_collection
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    logging.info(f"测试完成: {passed}/{total} 通过")
    
    if passed == total:
        logging.info("所有测试通过！可以开始正式训练。")
        logging.info("运行命令: python gomoku/train.py --mode init --num-iterations 100")
    else:
        logging.warning("部分测试失败，请检查代码。")

if __name__ == "__main__":
    main()
