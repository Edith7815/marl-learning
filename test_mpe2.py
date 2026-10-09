from mpe2 import simple_spread_v3
import torch

# 1. 验证 PyTorch
print("PyTorch 版本:", torch.__version__)
print("CUDA 可用:", torch.cuda.is_available())

# 2. 创建并重置环境
env = simple_spread_v3.env(render_mode=None)
env.reset(seed=42)

# 3. 让每个智能体随机走 5 步
for step in range(5):
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        if termination or truncation:
            action = None
        else:
            # 从动作空间随机采样
            action = env.action_space(agent).sample()
        env.step(action)
    print(f"第 {step+1} 步完成")

env.close()
print("环境测试完全通过！")