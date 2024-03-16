import os
from datetime import datetime, timedelta

import pandas as pd
import pandas_ta as ta
import pytz
from lumibot.entities import Asset, TradingFee
from lumibot.strategies.strategy import Strategy

from credentials import IS_BACKTESTING

"""
Strategy Description

This strategy creates a butterfly spread and condorizes it if the trade goes against us. 
The strategy is designed to be run on a daily basis and will create a butterfly spread 
at a specific time of the day. The strategy will then monitor the trade and condorize 
the butterfly if the trade goes against us. The strategy will also close the trade if 
the trade hits the maximum loss or take profit.

"""


class OptionsButterflyCondor(Strategy):
    parameters = {
        "symbol": "SPY", # The symbol of the underlying asset
        "days_to_expiry": 0,  # The number of days to expiry 
        "strike_step_size": 1,  # How far apart each strike should be
        "min_wing_size": 2,  # The minimum spread between the wings
        "time_to_start": "10:30",  # The time to start trading
        # "time_to_start": "12:00",  # The time to start trading
        "max_loss": 0.25,  # The maximum loss to take before getting out of the trade
        "first_adjustment_loss": 0.10,  # The first adjustment to make if the trade goes against us
        "take_profit": 0.20,  # The profit to take before getting out of the trade
        "pct_to_trade": 0.25,  # The percentage of the portfolio to trade
        "time_to_close": "15:30",  # The time to close the trade
        # "time_to_close": "13:00",  # The time to close the trade
        "days_of_week_to_trade": "024",  # The days of the week to trade, where 0 is Monday and 4 is Friday
        "wing_size_adjustment": 0,  # The amount to adjust the wing size by (0.1 = 10%)
        "max_adx": 20,  # The maximum ADX value to create a butterfly (ADX is a trend strength indicator)
        "min_gamma_risk": 8,  # The minimum gamma risk to take on the trade (it will only take a trade if the gamma risk is greater than this value)
        "expected_iv_collapse": 2.5,  # The expected implied volatility collapse
        "adx_length": 14,  # The length of the ADX indicator
    }

    def initialize(self):
        # The time to sleep between each trading iteration
        self.sleeptime = "5M"  # 1 minute = 1M, 1 hour = 1H,  1 day = 1D

        # Set the initial value of the variables
        self.num_losses = 0
        self.last_butterfly_purchase = None
        self.portfolio_value_at_purchase = None

        self.minutes_before_closing = 1
        self.pause_counter = None
        self.stop_for_today = None
        self.last_wing_size = None
        self.last_quantity = None
        self.underlying_price_at_purchase = None
        self.take_profit_portfolio_value = None
        self.max_loss_portfolio_value = None
        self.first_adjustment_loss_portfolio_value = None
        self.last_condorized = None

    def on_trading_iteration(self):
        # Get the parameters
        symbol = self.parameters["symbol"]
        days_to_expiry = self.parameters["days_to_expiry"]
        strike_step_size = self.parameters["strike_step_size"]
        min_wing_size = self.parameters["min_wing_size"]
        time_to_start = self.parameters["time_to_start"]
        max_loss = self.parameters["max_loss"]
        first_adjustment_loss = self.parameters["first_adjustment_loss"]
        take_profit = self.parameters["take_profit"]
        pct_to_trade = self.parameters["pct_to_trade"]
        time_to_close = self.parameters["time_to_close"]
        days_of_week_to_trade = self.parameters["days_of_week_to_trade"]
        wing_size_adjustment = self.parameters["wing_size_adjustment"]
        max_adx = self.parameters["max_adx"]
        min_gamma_risk = self.parameters["min_gamma_risk"]
        expected_iv_collapse = self.parameters["expected_iv_collapse"]
        adx_length = self.parameters["adx_length"]

        # Get the price of the underlying asset
        underlying_price = self.get_last_price(symbol)

        # Check if we got the price
        if underlying_price is None:
            self.log_message(f"Could not get price for {symbol}", color="red")
            return

        # Add lines to the chart
        self.add_line(f"{symbol}_price", underlying_price)

        # Get the current datetime
        dt = self.get_datetime()

        # Convert to New York time
        dt = dt.astimezone(pytz.timezone("America/New_York"))

        # Check if it is time to close the trade
        if dt.time() >= datetime.strptime(time_to_close, "%H:%M").time():
            # If it is time to close the trade, then sell all the positions
            self.sell_all()

            # Log the time of the trade closing
            self.log_message("Trade closed because of the time to close", color="red")

            # Add a marker to the chart
            self.add_marker(
                "trade_closed", symbol="square", color="blue", detail_text="Trade closed"
            )
            return

        # Check if we should be stopped for today
        if self.stop_for_today is not None:
            if dt.date() == self.stop_for_today.date():
                self.log_message("Stopped for today", color="red")
                return
            else:
                self.stop_for_today = None

        # Check if it is time to create the butterfly
        if dt.time() >= datetime.strptime(time_to_start, "%H:%M").time():
            # Check if it's a day of the week to trade
            day_of_the_week = str(dt.weekday())
            if day_of_the_week not in days_of_week_to_trade:
                return

            # Get the current portfolio value
            portfolio_value = self.get_portfolio_value()

            # If we already created a butterfly today, then return
            if (
                self.last_butterfly_purchase
                and self.last_butterfly_purchase.date() == dt.date()
            ):
                # Check if we hit the max loss
                if (
                    portfolio_value <= self.max_loss_portfolio_value
                ):
                    # If we hit the max loss, then sell all the positions
                    self.sell_all()

                    # Log the max loss hit
                    self.log_message("Max loss hit", color="red")
                    self.stop_for_today = dt

                    # Add a marker to the chart
                    self.add_marker(
                        "max_loss_hit", symbol="square", color="red", detail_text="Max loss hit"
                    )
                
                # Check if we hit the take profit
                if (
                    portfolio_value >= self.take_profit_portfolio_value
                ):
                    # If we hit the take profit, then sell all the positions
                    self.sell_all()

                    # Log the take profit hit
                    self.log_message("Take profit hit", color="green")
                    self.stop_for_today = dt

                    # Add a marker to the chart
                    self.add_marker(
                        "take_profit_hit", symbol="square", color="green", detail_text="Take profit hit"
                    )
                
                # Check if we pierced the tent
                upper_price = self.underlying_price_at_purchase + self.last_wing_size
                lower_price = self.underlying_price_at_purchase - self.last_wing_size
                if (
                    underlying_price > upper_price
                    or underlying_price < lower_price
                ):
                    # Log the tent pierced
                    self.log_message("Tent pierced", color="red")

                    # Condorize the butterfly
                    self.condorize_butterfly(symbol, days_to_expiry, strike_step_size, underlying_price, dt)

                    # Add marker to the chart
                    self.add_marker(
                        "tent_pierced", symbol="triangle-up", color="purple", detail_text="Tent pierced"
                    )

                # Check if we hit the first adjustment loss
                if (
                    portfolio_value <= self.first_adjustment_loss_portfolio_value
                ):
                    # Log the first adjustment loss hit
                    self.log_message("First adjustment loss hit", color="red")

                    # Condorize the butterfly
                    self.condorize_butterfly(symbol, days_to_expiry, strike_step_size, underlying_price, dt)

                    # Add marker to the chart
                    self.add_marker(
                        "first_adjustment_loss_hit", symbol="triangle-down", color="purple", detail_text="First adjustment loss hit"
                    )
                
                return 

            # Get historical prices for the underlying asset
            historical_prices = self.get_historical_prices(symbol, 25, "day")
            df = historical_prices.df

            # Calculate teh ADX indicator
            adx = ta.adx(df["high"], df["low"], df["close"], length=adx_length)

            # Get the last ADX value
            last_adx = adx.iloc[-1][f"ADX_{adx_length}"]

            # Add the ADX line to the chart
            self.add_line("ADX", last_adx)

            # Check if the ADX is greater than the max ADX
            if last_adx > max_adx:
                # Log that the ADX is too high
                self.log_message(f"ADX is too high: {last_adx}, skipping", color="red")

                # Add a marker to the chart
                self.add_marker(
                    "adx_too_high", symbol="circle", color="blue", detail_text=f"ADX too high: {last_adx}"
                )

                # Stop for today if the ADX is too high
                self.stop_for_today = dt

                return

            # Calculate the portfolio value to risk
            portfolio_value_to_risk = portfolio_value * pct_to_trade

            # Get the expiry date
            expiry = dt.date() + timedelta(days=days_to_expiry)

            # Get the nearest strike to the underlying price
            rounded_underlying_price = (
                round(underlying_price / strike_step_size) * strike_step_size
            )

            # Create the call asset
            call_atm_asset = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=rounded_underlying_price,
                right="call",
            )

            # Get the price of the call option
            call_atm_price = self.get_last_price(call_atm_asset)

            # If we got the price, then break because this expiry is valid
            if call_atm_price is None:
                return
            
            # Get the greeks for the option
            greeks = self.get_greeks(call_atm_asset)

            # Get the theta of the option
            abs_theta = abs(greeks["theta"] * 100)

            # Get the gamma of the option
            abs_gamma = abs(greeks["gamma"] * 100)

            # Get the vega of the option
            abs_vega = abs(greeks["vega"] * 100)

            # Calculate the gamme risk (Theta + (Expected IV collapse * Vega)) / Gamma
            gamma_risk = (abs_theta + (expected_iv_collapse * abs_vega)) / abs_gamma

            # Add a marker to the chart for the gamma risk
            self.add_marker(
                "gamma_risk", symbol="hexagon", color="blue", value=gamma_risk, detail_text=f"Gamma risk: {gamma_risk}"
            )

            # If the gamma risk is less than the minimum gamma risk, then return
            if gamma_risk < min_gamma_risk:
                self.log_message(f"Gamma risk is too low: {gamma_risk}, skipping", color="red")

                # Stop for today if the gamma risk is too low
                self.stop_for_today = dt

                return

            # Get the implied volatility
            implied_volatility = greeks["implied_volatility"]

            # Calculate the expected move using the implied volatility
            expected_move = underlying_price * (implied_volatility / 100) * ((days_to_expiry + 1) / 365) ** 0.5

            # Round the expected move to the nearest strike
            rounded_expected_move = round(expected_move / strike_step_size) * strike_step_size * (1 + wing_size_adjustment)
            
            # For now, set the wing size to the minimum wing size
            wing_size = max(min_wing_size, rounded_expected_move)
            
            # Set the quantity to trade
            quantity = int(portfolio_value_to_risk / wing_size / 100)

            # Create the butterfly
            self.create_butterfly(
                symbol,
                expiry,
                quantity,
                wing_size,
                rounded_underlying_price,
            )

            # Log last time of butterfly purchase
            self.last_butterfly_purchase = dt

            # Log the portfolio value at purchase
            self.portfolio_value_at_purchase = self.get_portfolio_value()

            # Set the last wing size
            self.last_wing_size = wing_size

            # Set the last quantity
            self.last_quantity = quantity

            # Set the underlying price at purchase
            self.underlying_price_at_purchase = underlying_price

            # Calculate the take profit portfolio value
            # 7 * num_contracts * 100 * 0.2
            take_profit_move = wing_size * quantity * 100 * take_profit
            self.take_profit_portfolio_value = self.get_portfolio_value() + take_profit_move

            # Calculate the max loss portfolio value
            # 7 * num_contracts * 100 * 0.2
            max_loss_move = wing_size * quantity * 100 * max_loss
            self.max_loss_portfolio_value = self.get_portfolio_value() - max_loss_move

            # Calculate the first adjustment loss portfolio value
            # 7 * num_contracts * 100 * 0.1
            first_adjustment_loss_move = wing_size * quantity * 100 * first_adjustment_loss
            self.first_adjustment_loss_portfolio_value = self.get_portfolio_value() - first_adjustment_loss_move

    def condorize_butterfly(self, symbol, days_to_expiry, strike_step_size, underlying_price, dt):
        # Check if we already condorized today
        if self.last_condorized and self.last_condorized.date() == dt.date():
            return

        # Get the expiry date
        expiry = dt.date() + timedelta(days=days_to_expiry)

        # Calculate the rounded underlying price
        rounded_underlying_price = (
            round(self.underlying_price_at_purchase / strike_step_size) * strike_step_size
        )

        # Determine which side of the tent was pierced
        if (
            underlying_price
            > self.underlying_price_at_purchase + self.last_wing_size
        ):
            self.log_message("Tent pierced on the upside", color="red")
            
            # Buy back the short call at the the original strike
            short_call_buyback_option = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=rounded_underlying_price,
                right="call",
            )

            # Create the order
            short_call_buyback_order = self.create_order(short_call_buyback_option, self.last_quantity, "buy")

            # Submit the order
            self.submit_order(short_call_buyback_order)

            # Sleep for 5 seconds to make sure the buy order is filled
            self.sleep(5)

            # Sell 2x the call option that is one wing size above the underlying price
            call_sell_asset = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=rounded_underlying_price + self.last_wing_size,
                right="call",
            )

            # Create the order
            call_sell_order = self.create_order(call_sell_asset, self.last_quantity * 2, "sell")

            # Submit the order
            self.submit_order(call_sell_order)

            # Sleep for 5 seconds to make sure the sell order is filled
            self.sleep(5)

            # Buy the call option that is two wing sizes above the underlying price
            call_buy_asset = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=rounded_underlying_price + self.last_wing_size * 2,
                right="call",
            )

            # Create the order
            call_buy_order = self.create_order(call_buy_asset, self.last_quantity, "buy")

            # Submit the order
            self.submit_order(call_buy_order)

            # Add a marker to the chart
            self.add_marker(
                "condorized upside", symbol="circle", color="blue", detail_text="Condorized on the upside"
            )

        else:
            self.log_message("Tent pierced on the downside", color="red")
            
            # Buy back the short put at the the original strike
            short_put_buyback_option = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=rounded_underlying_price,
                right="put",
            )

            # Create the order
            short_put_buyback_order = self.create_order(short_put_buyback_option, self.last_quantity, "buy")

            # Submit the order
            self.submit_order(short_put_buyback_order)

            # Sleep for 5 seconds to make sure the buy order is filled
            self.sleep(5)

            # Sell 2x the put option that is one wing size below the underlying price
            put_sell_asset = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=rounded_underlying_price - self.last_wing_size,
                right="put",
            )

            # Create the order
            put_sell_order = self.create_order(put_sell_asset, self.last_quantity * 2, "sell")

            # Submit the order
            self.submit_order(put_sell_order)

            # Sleep for 5 seconds to make sure the sell order is filled
            self.sleep(5)

            # Buy the put option that is two wing sizes below the underlying price
            put_buy_asset = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=rounded_underlying_price - self.last_wing_size * 2,
                right="put",
            )

            # Create the order
            put_buy_order = self.create_order(put_buy_asset, self.last_quantity, "buy")

            # Submit the order
            self.submit_order(put_buy_order)

            # Set the last condorized time
            self.last_condorized = dt

            # Add a marker to the chart
            self.add_marker(
                "condorized downside", symbol="circle", color="blue", detail_text="Condorized on the downside"
            )

    def create_butterfly(
        self,
        symbol,
        expiry,
        quantity_to_trade,
        wing_size,
        rounded_underlying_price
    ):
        # Check that the quantity to trade is greater than 0
        if quantity_to_trade <= 0:
            return

        # Create the call sell asset
        call_sell_asset = Asset(
            symbol,
            asset_type=Asset.AssetType.OPTION,
            expiration=expiry,
            strike=rounded_underlying_price,
            right=Asset.OptionRight.CALL,
        )
        # Create the call sell order
        call_sell_order = self.create_order(call_sell_asset, quantity_to_trade, "sell")

        # Create the put sell asset
        put_sell_asset = Asset(
            symbol,
            asset_type=Asset.AssetType.OPTION,
            expiration=expiry,
            strike=rounded_underlying_price,
            right=Asset.OptionRight.PUT,
        )
        # Create the put sell order
        put_sell_order = self.create_order(put_sell_asset, quantity_to_trade, "sell")

        # Create the call buy asset
        call_buy_asset = Asset(
            symbol,
            asset_type=Asset.AssetType.OPTION,
            expiration=expiry,
            strike=rounded_underlying_price + wing_size,
            right=Asset.OptionRight.CALL,
        )
        # Create the call buy order
        call_buy_order = self.create_order(call_buy_asset, quantity_to_trade, "buy")

        # Create the put buy asset
        put_buy_asset = Asset(
            symbol,
            asset_type=Asset.AssetType.OPTION,
            expiration=expiry,
            strike=rounded_underlying_price - wing_size,
            right=Asset.OptionRight.PUT,
        )
        # Create the put buy order
        put_buy_order = self.create_order(put_buy_asset, quantity_to_trade, "buy")

        # Submit the buy orders first
        self.submit_order(call_buy_order)
        self.submit_order(put_buy_order)

        # Sleep for 5 seconds to make sure the buy orders are filled
        self.sleep(5)

        # Submit the sell orders
        self.submit_order(call_sell_order)
        self.submit_order(put_sell_order)

        # Get the prices of all the assets in the butterfly
        call_sell_price = self.get_last_price(call_sell_order.asset)
        call_buy_price = self.get_last_price(call_buy_order.asset)
        put_sell_price = self.get_last_price(put_sell_order.asset)
        put_buy_price = self.get_last_price(put_buy_order.asset)

        detail_text = f"""
        Call Buy: {call_buy_order.asset.strike} @ {call_buy_price}\n
        Call Sell: {call_sell_order.asset.strike} @ {call_sell_price}\n
        Put Sell: {put_sell_order.asset.strike} @ {put_sell_price}\n
        Put Buy: {put_buy_order.asset.strike} @ {put_buy_price}
        """

        # Remove excess whitespace from the detail text
        detail_text = f"Wing Size: {wing_size} ".join(detail_text.split())

        # Add markers to the chart
        self.add_marker(
            "butterfly_created", symbol="square", color="yellow", detail_text=detail_text
        )

    def get_put_orders(
        self, symbol, expiry, strike_step_size, put_strike, quantity_to_trade, wing_size
    ):
        # Sell the put option at the put strike
        put_sell_asset = Asset(
            symbol,
            asset_type="option",
            expiration=expiry,
            strike=put_strike,
            right="put",
        )

        # Get the price of the put option
        put_sell_price = self.get_last_price(put_sell_asset)

        # Create the order
        put_sell_order = self.create_order(put_sell_asset, quantity_to_trade, "sell")

        # Buy the put option below the put strike
        put_buy_asset = Asset(
            symbol,
            asset_type="option",
            expiration=expiry,
            strike=put_strike - wing_size,
            right="put",
        )

        # Get the price of the put option
        put_buy_price = self.get_last_price(put_buy_asset)

        # Create the order
        put_buy_order = self.create_order(put_buy_asset, quantity_to_trade, "buy")

        if put_sell_price is None or put_buy_price is None:
            return None, None

        return put_sell_order, put_buy_order

    def get_call_orders(
        self,
        symbol,
        expiry,
        strike_step_size,
        call_strike,
        quantity_to_trade,
        wing_size,
    ):
        # Sell the call option at the call strike
        call_sell_asset = Asset(
            symbol,
            asset_type="option",
            expiration=expiry,
            strike=call_strike,
            right="call",
        )

        # Get the price of the call option
        call_sell_price = self.get_last_price(call_sell_asset)

        # Create the order
        call_sell_order = self.create_order(call_sell_asset, quantity_to_trade, "sell")

        # Buy the call option above the call strike
        call_buy_asset = Asset(
            symbol,
            asset_type="option",
            expiration=expiry,
            strike=call_strike + wing_size,
            right="call",
        )

        # Get the price of the call option
        call_buy_price = self.get_last_price(call_buy_asset)

        # Create the order
        call_buy_order = self.create_order(call_buy_asset, quantity_to_trade, "buy")

        if call_sell_price is None or call_buy_price is None:
            return None, None

        return call_sell_order, call_buy_order

    def get_strike_deltas(
        self,
        symbol,
        expiry,
        strikes,
        right,
        stop_greater_than=None,
        stop_less_than=None,
    ):
        """
        Get the delta for each strike

        Parameters
        ----------
        symbol : str
            The symbol of the underlying asset
        expiry : datetime
            The expiry date of the option
        strikes : list
            A list of strike prices
        right : str
            The side of the option
        stop_greater_than : float
            The delta to stop at if it is greater than this value
        stop_less_than : float
            The delta to stop at if it is less than this value

        Returns
        -------
        dict
            A dictionary of strike prices and their deltas
        """

        # Get the greeks for each strike
        strike_deltas = {}
        for strike in strikes:
            # Create the asset
            asset = Asset(
                symbol,
                asset_type="option",
                expiration=expiry,
                strike=strike,
                right=right,
            )
            greeks = self.get_greeks(asset)

            strike_deltas[strike] = greeks["delta"]

            if (
                stop_greater_than
                and greeks["delta"]
                and greeks["delta"] >= stop_greater_than
            ):
                break

            if stop_less_than and greeks["delta"] and greeks["delta"] <= stop_less_than:
                break

        return strike_deltas


if __name__ == "__main__":
    if not IS_BACKTESTING:
        from lumibot.brokers import Tradier
        from lumibot.traders import Trader

        from credentials import TRADIER_CONFIG



        trader = Trader()

        broker = Tradier(TRADIER_CONFIG)

        strategy = OptionsButterflyCondor(
            broker=broker,
            discord_webhook_url=os.environ.get("DISCORD_WEBHOOK_URL"),
            account_history_db_connection_str=os.environ.get(
                "ACCOUNT_HISTORY_DB_CONNECTION_STR"
            ),
            )

        trader.add_strategy(strategy)
        strategy_executors = trader.run_all()

    else:
        from lumibot.backtesting import PolygonDataBacktesting

        from credentials import POLYGON_CONFIG

        # backtesting_start = datetime(2020, 4, 2)
        # backtesting_start = datetime(2024, 1, 1)
        backtesting_start = datetime(2023, 1, 1)
        backtesting_end = datetime(2024, 2, 27)

        # 0.1% fee
        trading_fee = TradingFee(percent_fee=0.001)

        filename = "options_butterfly_condor.csv"

        # Load the results from the CSV file
        try:
            results_df = pd.read_csv(filename, index_col=0)
        except FileNotFoundError:
            # If the file doesn't exist, then create an empty DataFrame
            results_df = pd.DataFrame()

            # Add columns for the parameters
            for key, _ in OptionsButterflyCondor.parameters.items():
                results_df[key] = None

            # Add columns for backtesting start and end
            results_df["backtesting_start"] = None
            results_df["backtesting_end"] = None

        # Define the parameters to test
        time_to_starts = [None]
        pct_to_trades = [None]

        for time_to_start in time_to_starts:
            for pct_to_trade in pct_to_trades:
                if time_to_start is not None:
                    OptionsButterflyCondor.parameters["time_to_start"] = time_to_start
                if pct_to_trade is not None:
                    OptionsButterflyCondor.parameters["pct_to_trade"] = pct_to_trade

                # Make a string that is month/day/year for the backtesting start and end (eg. 1/1/20), where the yeqr is only two digits
                backtesting_start_str = f"{backtesting_start.month}/{backtesting_start.day}/{backtesting_start.year % 100}"
                backtesting_end_str = f"{backtesting_end.month}/{backtesting_end.day}/{backtesting_end.year % 100}"

                search_results = results_df[
                    (results_df["symbol"] == OptionsButterflyCondor.parameters["symbol"])
                    & (results_df["time_to_start"] == time_to_start)
                    & (results_df["pct_to_trade"] == pct_to_trade)
                    & (results_df["backtesting_start"] == backtesting_start_str)
                    & (results_df["backtesting_end"] == backtesting_end_str)
                ]

                if search_results.shape[0] > 0:
                    start = results_df["backtesting_start"][0]
                else:
                    start = backtesting_start

                name = f"Options Butterfly Condor {OptionsButterflyCondor.parameters['symbol']} {OptionsButterflyCondor.parameters['time_to_start']} time to start {OptionsButterflyCondor.parameters['time_to_close']} time to close {OptionsButterflyCondor.parameters['pct_to_trade']} pct to trade {OptionsButterflyCondor.parameters['max_loss']} max loss {OptionsButterflyCondor.parameters['take_profit']} take profit {OptionsButterflyCondor.parameters['first_adjustment_loss']} first adjustment loss {OptionsButterflyCondor.parameters['wing_size_adjustment']} wing size adjustment {OptionsButterflyCondor.parameters['max_adx']} max adx"

                # Check if the parameters are already in the results
                if search_results.shape[0] > 0:
                    # Print skippping message
                    print(
                        f"Skipping {name} from {backtesting_start} to {backtesting_end} because it is already in the results"
                    )
                    continue

                

                # Print out the parameters for the backtest
                print(
                    f"Backtesting {name} from {backtesting_start} to {backtesting_end}"
                )

                result = OptionsButterflyCondor.backtest(
                    PolygonDataBacktesting,
                    backtesting_start,
                    backtesting_end,
                    benchmark_asset="SPY",
                    buy_trading_fees=[trading_fee],
                    sell_trading_fees=[trading_fee],
                    polygon_api_key=POLYGON_CONFIG["API_KEY"],
                    polygon_has_paid_subscription=True,
                    budget=35000,
                    name=name,
                    show_plot=True,
                    show_tearsheet=True,
                    show_indicators=True,
                )

                # Change the max_drawdown dictionary to a single value (by only keeping the value of "drawdown" in the dictionary)
                result["max_drawdown"] = result["max_drawdown"]["drawdown"]

                result_df = pd.DataFrame(
                    [result]
                )  # Convert the dictionary to a DataFrame

                # Add columns for the parameters
                for key, value in OptionsButterflyCondor.parameters.items():
                    # If 'value' is a list, convert it to a string
                    if isinstance(value, list):
                        result_df[key] = str(value)
                    else:
                        result_df[key] = value

                # Add columns for backtesting start and end
                result_df["backtesting_start"] = backtesting_start_str
                result_df["backtesting_end"] = backtesting_end_str

                results_df = pd.concat(
                    [results_df, result_df], ignore_index=True
                )

                # Save the results to a CSV file
                # results_df.to_csv(filename)
