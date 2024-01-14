import alpaca.common.exceptions
import pandas as pd
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetAssetsRequest, MarketOrderRequest, OrderSide
from alpaca.trading.enums import OrderSide, TimeInForce
from find_current_day_trades_long import find_models_to_buy_long, check_current_trades_long
from find_current_vol_adjusted_order_sizes import quantify_order_size_vol_adjusted
import qi_client
import os

os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

API_KEY = ""
SECRET_KEY = ""
trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
clock = trading_client.get_clock()
print(f"Market is {'open' if clock.is_open else 'closed'}")
account = trading_client.get_account()
positions = trading_client.get_all_positions()
orders = trading_client.get_orders()

df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/currently_open_trades.csv')


def main_long_vol_adjusted():
    models_currently_open = []
    df_open = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/currently_open_trades.csv')
    for index, row in df_open.iterrows():
        models_currently_open.append(row['model'])
    models_to_exit = [item[0] for item in check_current_trades_long()[0]]
    today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
    for model in models_to_exit:
        print(f'{model} SELL')
        for position in positions:
            if position.symbol == model[0]:
                order_size = position.qty_available
                market_order_data = MarketOrderRequest(
                    symbol=model[0],
                    qty=order_size,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY)
                account_request = trading_client.submit_order(market_order_data)
    models_to_buy = [item[0] for item in find_models_to_buy_long()[0]]
    models_to_buy = [model for model in models_to_buy if model not in models_currently_open]
    current_trades_df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/currently_open_trades.csv')
    illiquid = 0
    for index, row in current_trades_df.iterrows():
        illiquid += row['order_size']
    for model in models_to_buy:
        order_size = quantify_order_size_vol_adjusted(model=model, date_of_trade_entry=today_date)
        print(f"Order Size: {order_size[1]}, Buying Power: {account.non_marginable_buying_power}")
        if 2 * illiquid >= account.equity:  # limit max leverage to 2 times equity of the account
            print('Maximum Leverage limit reached. No more trades may be placed.')
            continue
        market_order_data = MarketOrderRequest(
                              symbol=model,
                              qty=order_size[0],
                              side=OrderSide.BUY,
                              time_in_force=TimeInForce.DAY)
        try:
            account_request = trading_client.submit_order(market_order_data)
            print(f'{model} BUY')
        except alpaca.common.exceptions.APIError:
            print(f'there was a alpaca.common.exceptions.APIError for {model}')
            continue






if __name__ == '__main__':
    main_long_vol_adjusted()



