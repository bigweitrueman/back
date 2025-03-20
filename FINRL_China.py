import os
os.chdir('ta-lib') 
os.chdir('../')
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from IPython import display
display.set_matplotlib_formats("svg")

from meta import config
from meta.data_processors.tushare import Tushare, ReturnPlotter
from meta.env_stock_trading.env_stocktrading_China_A_shares import StockTradingEnv
from agents.stablebaselines3_models import DRLAgent
pd.options.display.max_columns = None
    
print("ALL Modules have been imported!")
if not os.path.exists("./datasets" ):
    os.makedirs("./datasets" )
if not os.path.exists("./trained_models"):
    os.makedirs("./trained_models" )
if not os.path.exists("./tensorboard_log"):
    os.makedirs("./tensorboard_log" )
if not os.path.exists("./results" ):
    os.makedirs("./results" )

data_file = './data/dataset.csv'
action_file = './data/action.csv'

ticket_list=['600000.SH', '600009.SH', '600016.SH', '600028.SH', '600030.SH',
       '600031.SH', '600036.SH', '600050.SH', '600104.SH', '600196.SH',
       '600276.SH', '600309.SH', '600519.SH', '600547.SH', '600570.SH']

train_start_date='2015-01-01'
train_stop_date='2019-08-01'
val_start_date='2019-08-01'
val_stop_date='2021-01-03'

token='27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5'

# 检查本地文件是否存在，如果存在则读取
if os.path.exists(data_file):
    ts_processor = Tushare(data_source="tushare", 
                           start_date=train_start_date,
                           end_date=val_stop_date,
                           time_interval="1d",
                           token=token)
    # 自定义一个读取 CSV 文件的方法
    def read_local_csv(self, file_path):
        self.dataframe = pd.read_csv(file_path)
    Tushare.read_local_csv = read_local_csv
    ts_processor.read_local_csv(data_file)
else:
    # 如果文件不存在，下载并保存
    ts_processor = Tushare(data_source="tushare", 
                           start_date=train_start_date,
                           end_date=val_stop_date,
                           time_interval="1d",
                           token=token)
    ts_processor.download_data(ticker_list=ticket_list)
    ts_processor.dataframe.to_csv(data_file, index=False)

# ts_processor = Tushare(data_source="tushare", 
#                            start_date=train_start_date,
#                            end_date=val_stop_date,
#                            time_interval="1d",
#                            token=token)

# ts_processor.dataframe = pd.read_csv(data_file)     #把读取的 CSV 文件内容赋值给 Tushare 实例的 dataframe 属性,而不是直接替换
ts_processor.clean_data()
ts_processor.fillna()
ts_processor.add_technical_indicator(config.INDICATORS)
ts_processor.fillna()
train = ts_processor.data_split(ts_processor.dataframe, train_start_date, train_stop_date) 
print(len(train.tic.unique()))
print(train.tic.unique())
print(train.head)

stock_dimension = len(train.tic.unique())
state_space = stock_dimension*(len(config.INDICATORS)+2)+1
print(f"Stock Dimension: {stock_dimension}, State Space: {state_space}")



save_path = "./trained_models/ddpg_model"

# 检查是否存在已保存的模型
# if os.path.exists(save_path + ".zip"):  # Stable Baselines3 会自动添加.zip扩展
#     print("Loading pre-trained model...")
#     from stable_baselines3 import DDPG
#     trained_ddpg = DDPG.load(save_path)
# else:
print("Training new model...")
#ddpg训练环境
env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000,  # 最大交易量
    "initial_amount": 1000000,  # 初始资金（100万）
    "buy_cost_pct":6.87e-5,  # 买入手续费率
    "sell_cost_pct":1.0687e-3,  # 卖出手续费率
    "reward_scaling": 1e-4,  # 奖励缩放因子
    "state_space": state_space, 
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS,  # 技术指标列表
    "print_verbosity": 1,  # 输出详细程度
    "initial_buy":True,  # 初始买入操作
    "hundred_each_trade":True  # 整手交易
}


e_train_gym = StockTradingEnv(df = train, **env_kwargs)
env_train, _ = e_train_gym.get_sb_env()
print(type(env_train))

agent = DRLAgent(env = env_train)

# DDPG算法参数配置
DDPG_PARAMS = {
    "batch_size": 256,
    "buffer_size": 50000,
    "learning_rate": 0.0005,
    "action_noise": "normal",
}
POLICY_KWARGS = dict(net_arch=dict(pi=[64, 64], qf=[400, 300]))

# 模型初始化与训练
model_ddpg = agent.get_model("ddpg", model_kwargs=DDPG_PARAMS, policy_kwargs=POLICY_KWARGS)
trained_ddpg = agent.train_model(model=model_ddpg, 
                                tb_log_name='ddpg',
                                total_timesteps=10000)
# 保存训练好的模型
trained_ddpg.save(save_path)
print(f"Model saved to {save_path}")



# 后续的交易预测部分保持不变
trade = ts_processor.data_split(ts_processor.dataframe, val_start_date, val_stop_date)
env_kwargs = {
    "stock_dim": stock_dimension,
    "hmax": 1000, 
    "initial_amount": 1000000, 
    "buy_cost_pct":6.87e-5,
    "sell_cost_pct":1.0687e-3,
    "reward_scaling": 1e-4,
    "state_space": state_space, 
    "action_space": stock_dimension,
    "tech_indicator_list": config.INDICATORS, 
    "print_verbosity": 1,
    "initial_buy":False,
    "hundred_each_trade":True
}
e_trade_gym = StockTradingEnv(df = trade, **env_kwargs)

df_account_value, df_actions = DRLAgent.DRL_prediction(model=trained_ddpg,
                       environment = e_trade_gym)

df_actions.to_csv("action.csv",index=False)

print(trade['time'])
