# 使用选择题数据训练需要修改verl的奖励函数相关部分
- chinese_choice_reward.py和english_choice_reward.py 复制 verl/utils/reward_score目录下
- 需要修改verl原来的verl/utils/reward_score/__init__.py和verl/workers/reward_manager/naive.py文件，把选择题的选项传入到reward函数。修改后的文件在verl_change下。