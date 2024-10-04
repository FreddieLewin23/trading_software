import pandas as pd
import math
from refactored_algo.algo_main import (
    find_models_to_buy_long,
    check_current_trades_long,
    quantify_order_size
)
import qi_client
import time
import os
import Qi_wrapper
import numpy as np
from datetime import datetime, timedelta
import sys

sys.path.append('/Users/FreddieLewin/miniconda3/envs/new_dl_token/lib/python3.11/site-packages/alpaca')
import alpaca.common.exceptions
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import (
    GetAssetsRequest,
    MarketOrderRequest,
    OrderSide
)
from alpaca.trading.enums import (OrderSide, TimeInForce, AssetStatus, OrderType)

os.environ['QI_API_KEY'] = ''
configuration = qi_client.Configuration()
configuration.api_key['X-API-KEY'] = ''
api_instance = qi_client.DefaultApi(qi_client.ApiClient(configuration))

# _________LONG TERM PAPER ACCOUNT________________________________________
API_KEY = ""
SECRET_KEY = ""
# ________________________________________________________________________
# MESS AROUND PAPER ACCOUNT________________________________________
# API_KEY = ""
# SECRET_KEY = ""
# ________________________________________________________________________


trading_client = TradingClient(API_KEY, SECRET_KEY, paper=True)
clock = trading_client.get_clock()
print(f"Market is {'open' if clock.is_open else 'closed'}")
account = trading_client.get_account()
positions = trading_client.get_all_positions()
orders = trading_client.get_orders()
df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')


def calculate_total_delta():
    total_delta = 0.0
    positions = trading_client.get_all_positions()
    for position in positions:
        qty = float(position.qty)
        current_price = float(position.current_price)

        if position.asset_class == 'option':
            # check this works properly
            option_delta = float(position.delta)
            delta = qty * option_delta * 100
        else:
            delta = qty * current_price
        total_delta += delta
    return total_delta


def return_r3k_option_ticker():
    today = datetime.today()
    r3k_data = Qi_wrapper.get_model_data(model='IWV', start=str(today)[:10], end=str(today)[:10], term='Long term')
    r3k_value = r3k_data['Model Value'].iloc[0] + r3k_data['Absolute Gap'].iloc[0]
    expiry = (today.replace(day=1) + timedelta(days=32)).replace(day=1) + timedelta(days=14)
    strike_price = round(r3k_value / 5) * 5
    ticker = f"IWV{expiry.strftime('%y%m%d')}P{str(int(strike_price * 1000)).zfill(8)}"
    return ticker


def save_hedging_transaction(transaction):
    df = pd.DataFrame([transaction])
    try:
        df.to_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/hedging_transcations.csv',
                  mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error saving transaction: {e}")


def execute_delta_hedge():
    total_delta = calculate_total_delta()
    hedge_ticker = return_r3k_option_ticker()

    curr_trades = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')
    model_positions = [position.symbol for position in positions]
    total_equity = 0
    macro_equity = 0

    today_str = str(datetime.today())[:10]
    for position in positions:
        model = position.symbol
        model_data = Qi_wrapper.get_model_data(model=model, start=today_str, end=today_str, term='Long term')
        today_rsq = model_data['Rsq'][0]
        if today_rsq > 75:
            macro_equity += position.market_value
        total_equity += position.market_value

    macro_equity_percentage = macro_equity / total_equity
    hedge_qty = math.floor(abs(total_delta) // 100 * macro_equity_percentage)
    # IE only hedge positions whose RSq have gone below 75 and are no longer in a macro regime

    if hedge_qty > 1000:
        # ONLY BUY PUTS< NEVER SELL PUTS (UNLIMITED DOWNSIDE)
        if total_delta > 0:
            side = OrderSide.BUY
            market_order_data = MarketOrderRequest(
                symbol=hedge_ticker,
                qty=hedge_qty,
                side=OrderSide.BUY,
                type=OrderType.MARKET,
                time_in_force=TimeInForce.DAY)

            account_request = trading_client.submit_order(market_order_data)
            print(f"Executed {side} order for {hedge_qty} contracts of {hedge_ticker}.")
            transaction = {
                "date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "ticker": hedge_ticker,
                "side": side.value,
                "quantity": hedge_qty,
                "delta": total_delta
            }
            save_hedging_transaction(transaction)
    else:
        print("No significant delta to hedge.")


def qi_vol_indicator(date):
    date = str(date).split()[0]
    indexes = ['S&P500', 'Euro Stoxx 600', 'USDJPY', 'EURUSD', 'USD 10Y Swap', 'EUR 10Y Swap']
    st_rsqs = [row['Rsq'] for model in indexes
               for _, row in Qi_wrapper.get_model_data(model=model, start=date, end=date, term='Short term').iterrows()]
    return 100 - np.mean(st_rsqs)


def current_max_leverage(date):
    current_volatility = qi_vol_indicator(date)
    base_leverage = 1
    average_volatility = 29.44
    adjustment_factor = max(0.5, 1 - 0.65 * (current_volatility - average_volatility) / average_volatility)
    adjusted_max_leverage = base_leverage * adjustment_factor
    return min(2, adjusted_max_leverage)


def main_long_vol_adjusted():
    curr_equity = account.equity
    print(f'Account Value is at {curr_equity}')
    df_account_tracker = pd.read_csv(
        "/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/account_value_timeseries_updated.csv")

    now_date = datetime.now()
    formatted_date = now_date.strftime("%Y-%m-%d %H:%M:%S")

    new_row = pd.DataFrame({'datetime': [formatted_date], 'index': [len(df_account_tracker) + 1], 'account_value': [curr_equity]})
    df_account_tracker = pd.concat([df_account_tracker, new_row], ignore_index=True)
    df_account_tracker.to_csv(
        "/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/account_value_timeseries_updated.csv",
        index=False)

    models_currently_open = []
    df_open = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')
    for index, row in df_open.iterrows():
        models_currently_open.append(row['model'])
    models_to_exit = [item[0] for item in check_current_trades_long()[0]]
    today_date = str(pd.Timestamp.today(tz='America/New_York').date().isoformat()).split()[0]
    for model in models_to_exit:
        for position in positions:
            if position.symbol == model[0]:
                order_size = position.qty_available
                market_order_data = MarketOrderRequest(
                    symbol=model[0],
                    qty=order_size,
                    side=OrderSide.SELL,
                    time_in_force=TimeInForce.DAY)
                account_request = trading_client.submit_order(market_order_data)
                print(f'{model} SELL')
    models_to_buy = [item[0] for item in find_models_to_buy_long()[0]]
    models_to_buy = [model for model in models_to_buy if model not in models_currently_open]
    current_trades_df = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv')
    illiquid = 0
    for index, row in current_trades_df.iterrows():
        illiquid += row['order_size']
    illiquid = float(account.long_market_value)
    for model in models_to_buy:
        order_size = quantify_order_size(model=model, date_of_trade_entry=today_date)
        if order_size[0] >= float(account.equity) / 15:
            continue
        curr_max_lev = current_max_leverage(today_date)
        print(f"Order Size: {order_size[1]}, Buying Power: {curr_max_lev * float(account.equity) - illiquid}")
        if illiquid >= curr_max_lev * float(account.equity):
            print('Maximum Leverage limit reached. No more trades may be placed.')
            print(f'Invested: {illiquid}, 2 * Equity: {2 * float(account.equity)}')
            continue
        market_order_data = MarketOrderRequest(
                              symbol=model,
                              qty=order_size[0],
                              side=OrderSide.BUY,
                              time_in_force=TimeInForce.DAY)
        try:
            account_request = trading_client.submit_order(market_order_data)
            print(f'{model} BUY, order size: {order_size[1]}')
            illiquid += order_size[1]
        except alpaca.common.exceptions.APIError:
            print(f'there was a alpaca.common.exceptions.APIError for {model}. Removed from trades')
            csv_path = '/Users/FreddieLewin/PycharmProjects/new_dl_token/refactored_algo/current_trades_refactor.csv'
            df_current_trades = pd.read_csv(csv_path)
            df_current_trades = df_current_trades[df_current_trades['model'] != model]
            df_current_trades.to_csv(csv_path, index=False)
            continue
    time.sleep(5)
    # wait 5 seconds to ensure orders have gone through ensuring hedging calculations are done correctly


if __name__ == '__main__':
    main_long_vol_adjusted()


    # res = execute_delta_hed

