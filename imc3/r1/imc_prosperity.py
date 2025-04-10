from datamodel import OrderDepth, UserId, TradingState, Order, Symbol
from typing import List, Any
import jsonpickle
import json
import numpy as np

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(
            self.to_json(
                [
                    self.compress_state(state, ""),
                    self.compress_orders(orders),
                    conversions,
                    "",
                    "",
                ]
            )
        )

        max_item_length = (self.max_log_length - base_length) // 3

        print(
            self.to_json(
                [
                    self.compress_state(state, self.truncate(state.traderData, max_item_length)),
                    self.compress_orders(orders),
                    conversions,
                    self.truncate(trader_data, max_item_length),
                    self.truncate(self.logs, max_item_length),
                ]
            )
        )

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            [],  # listings placeholder
            {symbol: [depth.buy_orders, depth.sell_orders] for symbol, depth in state.order_depths.items()},
            [],  # own_trades placeholder
            [],  # market_trades placeholder
            state.position,
            [],  # observations placeholder
        ]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])
        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value
        return value[: max_length - 3] + "..."

logger = Logger()

class Product:
    RAINFOREST_RESIN = "RAINFOREST_RESIN"
    KELP = "KELP"
    SQUID_INK = "SQUID_INK"

PARAMS = {
    Product.RAINFOREST_RESIN: {
        "fair_value": 10000,
        "take_width": 1,
        "clear_width": 0,
        "disregard_edge": 1,
        "join_edge": 2,
        "default_edge": 4,
        "soft_position_limit": 40, # need change
    },
    Product.KELP: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,  # need change
        "reversion_beta": 0.0026,
        "disregard_edge": 1,
        "join_edge": 0,
        "default_edge": 1,
    },
    Product.SQUID_INK: {
        "take_width": 1,
        "clear_width": 0,
        "prevent_adverse": False,
        "adverse_volume": 15,  # need change
        "reversion_beta": 0.00029, # besides mean reversion, check for other strategies
        "disregard_edge": 1, # 
        "join_edge": 0,
        "default_edge": 1,
    },
}

class Trader:
    def __init__(self, params=None):
        if params is None:
            params = PARAMS
        self.params = params
        self.LIMIT = {
            Product.RAINFOREST_RESIN: 50,
            Product.KELP: 50,
            Product.SQUID_INK: 50,
        }

    def run(self, state: TradingState):
        traderObject = jsonpickle.decode(state.traderData) if state.traderData else {}
        result = {}

        for product in [Product.RAINFOREST_RESIN, Product.KELP, Product.SQUID_INK]:
            position = state.position.get(product, 0)
            buy_order_volume = 0
            sell_order_volume = 0
            orders = []

            order_depth = state.order_depths[product]
            fair_value = self.params[product]["fair_value"] if "fair_value" in self.params[product] else None

            if product == Product.KELP or product == Product.SQUID_INK:
                if len(order_depth.sell_orders) != 0 and len(order_depth.buy_orders) != 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_bid = max(order_depth.buy_orders.keys())
                    filtered_ask = [
                        price
                        for price in order_depth.sell_orders.keys()
                        if abs(order_depth.sell_orders[price]) >= self.params[product]["adverse_volume"]
                    ]
                    filtered_bid = [
                        price
                        for price in order_depth.buy_orders.keys()
                        if abs(order_depth.buy_orders[price]) >= self.params[product]["adverse_volume"]
                    ]
                    mm_ask = min(filtered_ask) if len(filtered_ask) > 0 else None
                    mm_bid = max(filtered_bid) if len(filtered_bid) > 0 else None
                    if mm_ask == None or mm_bid == None:
                        if traderObject.get(f"{product}_last_price", None) == None:
                            mmmid_price = (best_ask + best_bid) / 2
                        else:
                            mmmid_price = traderObject[f"{product}_last_price"]
                    else:
                        mmmid_price = (mm_ask + mm_bid) / 2

                    if traderObject.get(f"{product}_last_price", None) != None:
                        last_price = traderObject[f"{product}_last_price"]
                        last_returns = (mmmid_price - last_price) / last_price
                        pred_returns = last_returns * self.params[product]["reversion_beta"]
                        fair_value = mmmid_price + (mmmid_price * pred_returns)
                    else:
                        fair_value = mmmid_price
                    traderObject[f"{product}_last_price"] = mmmid_price

            if len(order_depth.sell_orders) != 0:
                best_ask = min(order_depth.sell_orders.keys())
                best_ask_amount = -1 * order_depth.sell_orders[best_ask]
                if best_ask <= fair_value - self.params[product]["take_width"]:
                    quantity = min(best_ask_amount, self.LIMIT[product] - position)
                    if quantity > 0:
                        orders.append(Order(product, best_ask, quantity))
                        buy_order_volume += quantity

            if len(order_depth.buy_orders) != 0:
                best_bid = max(order_depth.buy_orders.keys())
                best_bid_amount = order_depth.buy_orders[best_bid]
                if best_bid >= fair_value + self.params[product]["take_width"]:
                    quantity = min(best_bid_amount, self.LIMIT[product] + position)
                    if quantity > 0:
                        orders.append(Order(product, best_bid, -quantity))
                        sell_order_volume += quantity

            position_after_take = position + buy_order_volume - sell_order_volume
            fair_for_bid = round(fair_value - self.params[product]["clear_width"])
            fair_for_ask = round(fair_value + self.params[product]["clear_width"])

            buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
            sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

            if position_after_take > 0:
                clear_quantity = sum(
                    volume for price, volume in order_depth.buy_orders.items() if price >= fair_for_ask
                )
                clear_quantity = min(clear_quantity, position_after_take)
                sent_quantity = min(sell_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_ask, -abs(sent_quantity)))
                    sell_order_volume += abs(sent_quantity)

            if position_after_take < 0:
                clear_quantity = sum(
                    abs(volume) for price, volume in order_depth.sell_orders.items() if price <= fair_for_bid
                )
                clear_quantity = min(clear_quantity, abs(position_after_take))
                sent_quantity = min(buy_quantity, clear_quantity)
                if sent_quantity > 0:
                    orders.append(Order(product, fair_for_bid, abs(sent_quantity)))
                    buy_order_volume += abs(sent_quantity)

            asks_above_fair = [price for price in order_depth.sell_orders if price > fair_value + self.params[product]["disregard_edge"]]
            bids_below_fair = [price for price in order_depth.buy_orders if price < fair_value - self.params[product]["disregard_edge"]]

            ask = round(fair_value + self.params[product]["default_edge"])
            if asks_above_fair:
                best_ask_above_fair = min(asks_above_fair)
                if abs(best_ask_above_fair - fair_value) <= self.params[product]["join_edge"]:
                    ask = best_ask_above_fair
                else:
                    ask = best_ask_above_fair - 1

            bid = round(fair_value - self.params[product]["default_edge"])
            if bids_below_fair:
                best_bid_below_fair = max(bids_below_fair)
                if abs(fair_value - best_bid_below_fair) <= self.params[product]["join_edge"]:
                    bid = best_bid_below_fair
                else:
                    bid = best_bid_below_fair + 1

            if position > self.params[product].get("soft_position_limit", 10):
                ask -= 1
            elif position < -self.params[product].get("soft_position_limit", 10):
                bid += 1

            buy_quantity = self.LIMIT[product] - (position + buy_order_volume)
            sell_quantity = self.LIMIT[product] + (position - sell_order_volume)

            if buy_quantity > 0:
                orders.append(Order(product, round(bid), buy_quantity))
            if sell_quantity > 0:
                orders.append(Order(product, round(ask), -sell_quantity))

            result[product] = orders

        trader_data = jsonpickle.encode(traderObject)
        conversions = 1
        logger.flush(state, result, conversions, trader_data)
        return result, conversions, trader_data
        