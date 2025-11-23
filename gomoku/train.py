"""
五子棋 MCTS 训练脚本
使用 torchrl 训练基于 MCTS 的五子棋对战算法
"""

import argparse
import logging
import sys
from typing import Literal

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log', mode='w', encoding='utf-8')
    ]
)

def main():
    """主训练函数"""
    parser = argparse.ArgumentParser(description='五子棋 MCTS 训练脚本')
    parser.add_argument('--mode', type=str, choices=['init', 'continue'], 
                       default='init', help='训练模式: init(从头开始) 或 continue(继续训练)')
    parser.add_argument('--num-simulations', type=int, default=800,
                       help='MCTS 模拟次数')
    parser.add_argument('--num-iterations', type=int, default=1000,
                       help='训练迭代次数')
    parser.add_argument('--eval-interval', type=int, default=50,
                       help='评估间隔')
    parser.add_argument('--save-interval', type=int, default=100,
                       help='模型保存间隔')
    
    args = parser.parse_args()
    
    logging.info("=" * 50)
    logging.info("开始五子棋 MCTS 训练")
    logging.info(f"训练模式: {args.mode}")
    logging.info(f"MCTS 模拟次数: {args.num_simulations}")
    logging.info(f"训练迭代次数: {args.num_iterations}")
    logging.info(f"评估间隔: {args.eval_interval}")
    logging.info(f"模型保存间隔: {args.save_interval}")
    logging.info("=" * 50)
    
    try:
        # 导入训练流程
        from gomoku.mcts_train_flow import MCTSTrainFlow
        
        # 初始化训练流程
        train_flow = MCTSTrainFlow(
            mode=args.mode,
            num_simulations=args.num_simulations
        )
        
        # 开始训练循环
        for k in range(train_flow.init_k, args.num_iterations):
            logging.info(f"\n--- 迭代 {k} ---")
            
            # 收集经验数据
            logging.info("收集经验数据...")
            train_flow.collect_experience(num_games=5)
            logging.info(f"经验回放缓冲区大小: {len(train_flow.replay_buffer)}")
            
            # 训练步骤
            logging.info("执行训练步骤...")
            train_flow.train_step()
            
            # 定期评估
            if k % args.eval_interval == 0:
                logging.info("执行评估步骤...")
                train_flow.eval_step()
            
            # 定期保存模型
            if k % args.save_interval == 0:
                logging.info("保存模型...")
                train_flow.save_model(k)
                
            # 打印进度
            if k % 10 == 0:
                logging.info(f"进度: {k}/{args.num_iterations} ({k/args.num_iterations*100:.1f}%)")
        
        # 训练完成，保存最终模型
        logging.info("训练完成，保存最终模型...")
        train_flow.save_model(args.num_iterations)
        
        logging.info("训练成功完成！")
        
    except Exception as e:
        logging.error(f"训练过程中发生错误: {e}")
        raise

if __name__ == "__main__":
    main()
