import tushare as ts
pro = ts.pro_api('27080ec403c0218f96f388bca1b1d85329d563c91a43672239619ef5')
df = pro.trade_cal(exchange='', start_date='20180901', end_date='20181001', fields='exchange,cal_date,is_open,pretrade_date', is_open='0')