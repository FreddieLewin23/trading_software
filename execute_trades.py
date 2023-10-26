import alpaca.common.exceptions
import pandas as pd
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest, OrderSide
from alpaca.trading.enums import OrderSide, TimeInForce
from backtest_json_redo_MVG import find_sd_from_model_data
import datetime
from find_current_day_trades import find_models_to_buy, check_current_trades, quantify_buy_amount
from _delete import check_current_trades_temp, find_models_to_buy_temp
import qi_client
import os
import Qi_wrapper
from icecream import ic

# request ID - 828da785282630321fe5bf62cec5e2ca
# API-KEY AKU0QQ19OHEO77PEWZPL
# secret key LRK1xDNhwHVt3CgCVMGLDYP6vAwhWXDVbYaSsOuH
# $ curl -v https://paper-api.alpaca.markets/v2/account
# trading_client = TradingClient('AKU0QQ19OHEO77PEWZPL', 'LRK1xDNhwHVt3CgCVMGLDYP6vAwhWXDVbYaSsOuH')

os.environ['QI_API_KEY'] = 'API-KEY-QI'
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = 'QI-API-KEY'
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))


API_KEY = "ALPACA-API-KEY"
SECRET_KEY = "ALPACA-SECRET-KEY"
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
account = trading_client.get_account()
positions = trading_client.get_all_positions()
orders = trading_client.get_orders()

df = pd.read_csv('currently_open_trades.csv')


def main():
    models_currently_open = []
    df_open = pd.read_csv('currently_open_trades.csv')
    for index, row in df_open.iterrows():
        models_currently_open.append(row['model'])
    models_to_buy = [item[0] for item in find_models_to_buy()[0]]
    models_to_exit = [item[0] for item in check_current_trades()[0]]
    today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
    # this adds the buy signals, for models where i am not currently in the trade
    models_to_buy = [model for model in models_to_buy if model not in models_currently_open]
    for model in models_to_buy:
        order_size = quantify_buy_amount(model=model, date_of_trade_entry=today_date)

        # if i don't have the money, don't try to enter the trade
        if float(order_size[1]) > float(account.non_marginable_buying_power):
            continue
        print(f'{model} BUY')
        market_order_data = MarketOrderRequest(
                              symbol=model,
                              qty=order_size[0],
                              side=OrderSide.BUY,
                              time_in_force=TimeInForce.GTC)
        try:
            account_request = trading_client.submit_order(market_order_data)
        except alpaca.common.exceptions.APIError:
            continue

    # this sells the qty avaialble for the models the AI tells me to sell, if i am in an active trade with the model
    for model in models_to_exit:
        print(f'{model} SELL')
        for position in positions:
            if position.symbol == model[0]:
                order_size = position.qty_available
                market_order_data = MarketOrderRequest(
                    symbol=model[0],
                    qty=order_size,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.GTC)
                account_request = trading_client.submit_order(market_order_data)

                
if __name__ == '__main__':
    main()


