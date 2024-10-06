import Qi_wrapper
from backtesting_20251001.backtest_one_day import (check_current_trades,
                                                   find_models_to_buy,
                                                   filter_csv_by_date_bfo)
import warnings
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from scipy.stats import norm
from dateutil.relativedelta import relativedelta
import math
import time


warnings.simplefilter(action='ignore', category=FutureWarning)


def get_current_vol(date: str, lookback_days=21) -> float:
    """
    Function to calculate the current annualized volatility of the Russell 3000 (or proxy like IWV ETF)
    based on the past lookback_days of data.
    """
    date_str = str(date)[:10]
    end_date = date_str
    start_date = str(pd.to_datetime(end_date) - pd.Timedelta(days=lookback_days))
    r3k_data = filter_csv_by_date_bfo(model='IWV', start_date=start_date, end_date=end_date)
    r3k_data['True Value'] = r3k_data['Model Value'] + r3k_data['Absolute Gap']
    r3k_data = r3k_data.sort_values(by='Date')
    r3k_data['Log Returns'] = np.log(r3k_data['True Value'] / r3k_data['True Value'].shift(1))
    r3k_data.dropna(inplace=True)
    daily_volatility = r3k_data['Log Returns'].std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility


def calculate_option_value(date: str, strike_price: float, expiry_date: str,
                           risk_free_rate: float, option_type='put') -> list[int]:
    """
    Function to calculate the Black-Scholes value of an option (call or put).

    Args:
    - date (datetime or string): The current date.
    - strike_price (float): The strike price of the option.
    - expiry_date (datetime or string): The option's expiry date.
    - risk_free_rate (float): The risk-free interest rate (annualized).
    - option_type (str): 'call' or 'put'. Default is 'put'.

    Returns:
    - float: The Black-Scholes price of the option.
    """
    current_r3k_vol = get_current_vol(date=date)

    r3k_data = filter_csv_by_date_bfo(model='IWV', start_date=date, end_date=date)
    r3k_data['True Value'] = r3k_data['Model Value'] + r3k_data['Absolute Gap']
    current_r3k_value = r3k_data['True Value'].iloc[0]

    expiry_date = pd.to_datetime(expiry_date)
    date = pd.to_datetime(date)
    time_to_expiry = (expiry_date - date).days / 365.0

    d1 = (np.log(current_r3k_value / strike_price) + (risk_free_rate + (current_r3k_vol ** 2) / 2) * time_to_expiry) / (
                current_r3k_vol * np.sqrt(time_to_expiry))
    d2 = d1 - current_r3k_vol * np.sqrt(time_to_expiry)

    if option_type == 'call':
        option_price = (current_r3k_value * norm.cdf(d1)) - (
                    strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(d2))
        delta = norm.cdf(d1)
    elif option_type == 'put':
        option_price = (strike_price * np.exp(-risk_free_rate * time_to_expiry) * norm.cdf(-d2)) - (
                    current_r3k_value * norm.cdf(-d1))
        delta = norm.cdf(d1) - 1
    else:
        raise ValueError("option_type must be 'call' or 'put'")

    return [option_price, delta]


def calculate_total_delta(date: str) -> list[int]:
    total_delta = 0.0
    equity_positions = pd.read_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv')
    for index, row in equity_positions.iterrows():
        model_data = filter_csv_by_date_bfo(model=row['model'], start_date=date, end_date=date)
        model_data['price'] = model_data['Model Value'] + model_data['Absolute Gap']
        entry_price = row['real_value']
        order_size = row['order_size']
        return_trade = (model_data['price'].tolist()[0] - entry_price) / entry_price
        current_position_delta = order_size * (1 + return_trade)
        total_delta += current_position_delta

    current_hedging_positions = pd.read_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/hedging_current.csv')
    options_notional_value = 0
    for index, row in current_hedging_positions.iterrows():

        _hedging = calculate_option_value(date=date, strike_price=row['strike_price'],
                                          expiry_date=row['expiry_date'], risk_free_rate=0.01)
        delta = _hedging[1]
        value_bsm = _hedging[0]
        grouped_value = value_bsm * row['number_of_contracts']
        options_notional_value += grouped_value

        # this should be negative.
        '''
        CHECK THIS, THE * 100 MIGHT BE TOO MUCH (X100), CAUSING A 100X HEDGE, PUTTING ME EXTREMELY SHORT
        '''
        grouped_delta = 100 * row['number_of_contracts'] * delta
        total_delta += grouped_delta

    # delta should be close to zero, and options_notional_value varies a lot
    return [total_delta, options_notional_value]


def return_r3k_option_ticker(date: str) -> list[int]:
    date_dt = pd.to_datetime(date, format='%Y-%m-%d')

    r3k_data = filter_csv_by_date_bfo(model='IWV', start_date=str(date_dt)[:10], end_date=str(date_dt)[:10])
    r3k_value = r3k_data['Model Value'].iloc[0] + r3k_data['Absolute Gap'].iloc[0]
    strike_price_lower = round(r3k_value * 0.9 / 5) * 5

    expiry = date_dt + relativedelta(weeks=7)
    if expiry.weekday() >= 5:
        expiry = expiry - relativedelta(days=2)
    ticker = f"IWV{expiry.strftime('%y%m%d')}P{str(int(strike_price_lower * 1000)).zfill(8)}"

    premium_per_contract = calculate_option_value(date=date, strike_price=strike_price_lower,
                                                  expiry_date=expiry.strftime('%Y-%m-%d'), risk_free_rate=0.01,
                                                  option_type='put')[0]
    return [ticker, premium_per_contract]


def save_hedging_transaction(transaction: dict) -> None:
    df = pd.DataFrame([transaction])
    try:
        df.to_csv(
            '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/hedging_current.csv',
            mode='a', header=False, index=False)
    except Exception as e:
        print(f"Error saving transaction: {e}")


def execute_delta_hedge(date: str) -> float:
    total_delta = calculate_total_delta(date)
    _hedge = return_r3k_option_ticker(date)
    hedge_ticker = _hedge[0]
    hedge_premium_paid = _hedge[1]

    '''
    strike_price = int(hedge_ticker[-8:]) / 1000
    date_str = hedge_ticker[3:9]
    expiry_date = datetime.datetime.strptime(date_str, '%y%m%d').strftime('%Y-%m-%d')
    '''

    hedge_qty = math.floor(abs(total_delta) // 100)
    if hedge_qty > 10:
        if total_delta > 0:
            transaction = {
                "date": date,
                "hedge_ticker": hedge_ticker,
                "number_of_contracts": hedge_qty,
                'premium_paid': hedge_premium_paid
            }

            save_hedging_transaction(transaction)

            # when you initially buy the puts, take away the premium paid from the most recent account value
            account_value_df = pd.read_csv(
                '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv')
            last_value = account_value_df['account_value'].iloc[-1]
            new_last_value = last_value - hedge_premium_paid
            account_value_df.at[account_value_df.index[-1], 'account_value'] = new_last_value
            account_value_df.to_csv(
                '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv',
                index=False)

    else:
        print(f"No significant delta to hedge on: {date}")

    return hedge_premium_paid


def check_current_hedge(date: str) -> float:
    # iterate over current hedges. if they are ITM and profitable +20% of premium paid, exercise the contracts and take
    # profit. if they have expired check if they are ITM, otherwise they are worthless. if not expired and above strike,
    # not ITM do nothing. then add the completed hedging transactions to completed hedges csv file.

    hedging_positions = pd.read_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/hedging_current.csv')
    completed_hedges = []
    profit = 0

    # Iterate over each hedge position
    for index, row in hedging_positions.iterrows():

        contract_ticker = row['hedge_ticker']
        entry_date = row['date']

        strike_price = int(contract_ticker[-8:]) / 1000
        expiry_date_dt = pd.to_datetime(row['expiry_date'])

        number_of_contracts = row['number_of_contracts']
        premium_paid = row['premium_paid']

        # Check if the option has expired
        if expiry_date_dt < pd.to_datetime(date):
            # If expired, check if ITM (price below strike for puts)
            r3k_data = filter_csv_by_date_bfo(model='IWV', start_date=date, end_date=date)
            r3k_value = r3k_data['Model Value'].iloc[0] + r3k_data['Absolute Gap'].iloc[0]

            if r3k_value < strike_price:
                # Calculate profit from exercising
                profit_per_contract = strike_price - r3k_value
                total_profit = profit_per_contract * 100 * number_of_contracts

                if profit_per_contract <= 0:
                    print(f'Put expired worthless (strike price > r3k_value), profit = -1 * premium paid')
                    completed_hedges.append({
                        "date_entry": entry_date,
                        'date_exit': date,
                        "hedge_ticker": row['hedge_ticker'],
                        "number_of_contracts": number_of_contracts,
                        "profit": total_profit,
                        'premium_paid': premium_paid
                    })
                    # no need to add profit, since you receive nothing when a put expires OTM.
                else:
                    # made money on the put contract
                    completed_hedges.append({
                        "date_entry": entry_date,
                        'date_exit': date,
                        "hedge_ticker": row['hedge_ticker'],
                        "number_of_contracts": number_of_contracts,
                        "profit": total_profit,
                        'premium_paid': premium_paid
                    })
                    print(f"Exercised {row['hedge_ticker']} for profit of {total_profit}")
                    profit += total_profit

            print(f"{row['hedge_ticker']} expired worthless.")

        else:
            # If not expired, check if ITM and profitable
            r3k_data = filter_csv_by_date_bfo(model='IWV', start_date=date, end_date=date)
            r3k_value = r3k_data['Model Value'].iloc[0] + r3k_data['Absolute Gap'].iloc[0]

            if r3k_value < strike_price:
                # Check if the profit exceeds twice the premium paid
                profit_per_contract = strike_price - r3k_value
                total_profit = 100 * profit_per_contract * number_of_contracts
                if total_profit > 2 * premium_paid * number_of_contracts:
                    completed_hedges.append({
                        "date_entry": entry_date,
                        'date_exit': date,
                        "hedge_ticker": row['hedge_ticker'],
                        "number_of_contracts": number_of_contracts,
                        "profit": total_profit,
                        'premium_paid': premium_paid
                    })
                    profit += total_profit
                    print(f"Exercised {row['hedge_ticker']} for profit of {total_profit}")
                else:
                    print(f"{row['hedge_ticker']} is ITM but not profitable enough to exercise.")
            else:
                print(f"{row['hedge_ticker']} is not ITM.")

    if completed_hedges:
        completed_hedges_df = pd.DataFrame(completed_hedges)

        completed_hedges_df.to_csv(
            '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/completed_hedges.csv',
            mode='a', header=False, index=False
        )

    print(f"Completed hedge check for {date}.")
    account_value_df = pd.read_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv')
    last_value = account_value_df['account_value'].iloc[-1]
    new_last_value = last_value + profit
    account_value_df.at[account_value_df.index[-1], 'account_value'] = new_last_value
    account_value_df.to_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv',
        index=False)
    return profit


def find_account_value_and_update_account_value(date: str) -> float:
    df_curr = pd.read_csv("/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/current.csv")
    current_trade_illiquid = sum(df_curr['order_size'].tolist())
    value_of_investment = 0
    for index, row in df_curr.iterrows():
        buy_price = row['real_value']
        trade_data_curr = filter_csv_by_date_bfo(model=row['model'], start_date=date, end_date=date)
        # this just says, when adding unrealised gains to the account, if there is no model data
        if len(trade_data_curr) == 0:
            trade_data_curr = Qi_wrapper.get_model_data(model=row['model'], start=date, end=date, term='Long term')
        if len(trade_data_curr) == 0:
            sell_price = buy_price
        else:
            sell_price = trade_data_curr['Model Value'][0] + trade_data_curr['Absolute Gap'][0]
        value_of_investment += row['order_size'] * (1 + ((sell_price - buy_price) / buy_price))
    change_in_current_investments = value_of_investment - current_trade_illiquid

    value = 100_000
    df_completed_trades = pd.read_csv('/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/completed.csv')

    for index_j, row1 in df_completed_trades.iterrows():
        buy = row1['trade_entry_price']
        sell = row1['trade_exit_price']
        order_size = row1['order_size']
        percent_return = ((sell - buy) / buy) * 100
        value_added = order_size * (percent_return / 100)
        value += value_added

    # this loops over current contracts I own, calculates the BS warranted value and adds it to current price
    current_hedges_df = pd.read_csv(
        '/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/hedging_current.csv')
    for index, row in current_hedges_df:
        ticker = row['hedge_ticker']
        expiry_str = ticker[3:9]
        strike_price = int(ticker[-8:]) / 1000
        expiry_date = datetime.datetime.strptime(expiry_str, '%y%m%d').strftime('%Y-%m-%d')
        _current_hedge = calculate_option_value(date=date, strike_price=strike_price, expiry_date=expiry_date,
                                                risk_free_rate=0.01, option_type='put')
        current_hedge_value = _current_hedge[0]
        current_hedge_delta = _current_hedge[1]

        value += current_hedge_value
        '''
        I want to print the current portfolio delta here to ensure I am properly hedged
        '''

    current_account_value = value + change_in_current_investments
    df_account_tracker = pd.read_csv(
        "/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv")
    new_row = pd.DataFrame({'date': [date], 'account_value': [current_account_value]})
    df_account_tracker = pd.concat([df_account_tracker, new_row], ignore_index=True)
    df_account_tracker.to_csv(
        "/Users/FreddieLewin/PycharmProjects/new_dl_token/backtesting_20241005/data/account_value.csv", index=False)
    return current_account_value


def weekdays_between_dates(start_date_str: str, end_date_str: str) -> list[int]:
    start_date = datetime.strptime(start_date_str, "%Y-%m-%d")
    end_date = datetime.strptime(end_date_str, "%Y-%m-%d")
    if start_date > end_date:
        return [1, 2]
    weekdays = []
    current_date = start_date
    while current_date <= end_date:
        if current_date.weekday() < 5:
            weekdays.append(current_date.strftime("%Y-%m-%d"))
        current_date += timedelta(days=1)
    return weekdays


curr_pitc_model_data = None


if __name__ == '__main__':
    dates_for_backtest = weekdays_between_dates('2013-01-01', '2024-01-01')
    for date_curr in dates_for_backtest:
        print(date_curr)
        find_models_to_buy(date_curr)
        check_current_trades(date_curr)
        find_account_value_and_update_account_value(date_curr)
        time.sleep(1)
        # ensure new trades and completed trades are placed in csv files
        check_current_hedge(date_curr)
        time.sleep(1)
        # ensure old hedges are executed or left alone
        execute_delta_hedge(date_curr)
