from datamodel import OrderDepth, TradingState, Order
from typing import List, Tuple, Dict, Optional
import jsonpickle

class Product:
    KELP = "KELP"
    RESIN = "RAINFOREST_RESIN"
    INK = "SQUID_INK"
    CROISSANT = "CROISSANTS"
    JAM = "JAMS"
    DJEMBE = "DJEMBES"
    PICNIC1 = "PICNIC_BASKET1"
    PICNIC2 = "PICNIC_BASKET2"

class Trader:

    def __init__(self):
        # Per-product position limits
        self.LIMIT = {
            Product.KELP: 50,
            Product.RESIN: 50,
            Product.INK: 50,
            Product.CROISSANT: 250,
            Product.JAM: 350,
            Product.DJEMBE: 60,
            Product.PICNIC1: 60,
            Product.PICNIC2: 100
        }

        # You can define more parameters for your logic here
        self.PARAMS = {
            Product.RESIN: {
                "fair_value": 10000,
                "take_width": 1,
                "clear_width": 0.5,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 25,
                "volume_limit": 0
            },
            Product.KELP: {
                "take_width": 1,
                "clear_width": 0,
                "prevent_adverse": True,
                "adverse_volume": 15,
                "reversion_beta": 1,
                "disregard_edge": 1,
                "join_edge": 2,
                "default_edge": 4,
                "soft_position_limit": 25
            },
            "picnic_arb": {
                "threshold": 120,      # normal entry threshold for the spread
                "exit_threshold": 10,  # if spread is inside ±10, we attempt to flatten
            }
        }

    #-------------------------------------------------------------------------
    # A helper to retrieve an OrderDepth and our current position
    def extract_state(self, state: TradingState, product: str) -> Tuple[OrderDepth, int]:
        od = state.order_depths.get(product)
        pos = state.position.get(product, 0)
        return od, pos

    #-------------------------------------------------------------------------
    # 1) Basic "take" logic used for KELP, RESIN, etc.
    def take_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        take_width: float,
        position: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        limit = self.LIMIT[product]
        buy_volume = 0
        sell_volume = 0

        # If best_ask < (fair_value - take_width), we buy from that ask
        if order_depth and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders)
            best_ask_vol = -order_depth.sell_orders[best_ask]  # negative => positive quantity
            if best_ask <= fair_value - take_width:
                # Cap so we don't exceed limit
                can_buy = limit - position
                qty = min(can_buy, best_ask_vol)
                if qty > 0:
                    orders.append(Order(product, best_ask, qty))
                    buy_volume += qty

        # If best_bid > (fair_value + take_width), we sell into that bid
        if order_depth and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders)
            best_bid_vol = order_depth.buy_orders[best_bid]
            if best_bid >= fair_value + take_width:
                # Cap so we don't exceed limit
                can_sell = limit + position  # e.g. if pos= -10, limit=50 => can_sell=40
                qty = min(can_sell, best_bid_vol)
                if qty > 0:
                    orders.append(Order(product, best_bid, -qty))
                    sell_volume += qty

        return orders, buy_volume, sell_volume

    #-------------------------------------------------------------------------
    # 2) "Clear" logic: tries to partially exit a net position if the market is favorable
    def clear_orders(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        clear_width: float,
        position: int,
        buy_volume: int,
        sell_volume: int
    ) -> Tuple[List[Order], int, int]:
        orders: List[Order] = []
        net_pos = position + buy_volume - sell_volume
        limit = self.LIMIT[product]

        if net_pos > 0:
            # Net LONG => see if there's big buy interest at a high price
            # (>= fair_value+clear_width)
            # We'll only sell up to net_pos or up to the volume at that price
            if order_depth and order_depth.buy_orders:
                clear_qty = sum(
                    qty for price, qty in order_depth.buy_orders.items() 
                    if price >= fair_value + clear_width
                )
                # Also can't exceed net_pos
                qty = min(clear_qty, net_pos)
                # Also can't exceed limit on the negative side
                can_sell = limit + position  # "capacity" to go more negative
                qty = min(qty, can_sell)
                if qty > 0:
                    orders.append(Order(product, round(fair_value + clear_width), -qty))
                    sell_volume += qty

        elif net_pos < 0:
            # Net SHORT => see if there's cheap supply at price <= fair_value-clear_width
            if order_depth and order_depth.sell_orders:
                clear_qty = sum(
                    -qty for price, qty in order_depth.sell_orders.items() 
                    if price <= fair_value - clear_width
                )
                qty = min(clear_qty, -net_pos)
                # Also can't exceed limit on the positive side
                can_buy = limit - position
                qty = min(qty, can_buy)
                if qty > 0:
                    orders.append(Order(product, round(fair_value - clear_width), qty))
                    buy_volume += qty

        return orders, buy_volume, sell_volume

    #-------------------------------------------------------------------------
    # 3) Market make logic: places "limit" orders near fair_value
    def market_make(
        self,
        product: str,
        order_depth: OrderDepth,
        fair_value: float,
        position: int,
        buy_volume: int,
        sell_volume: int,
        disregard_edge: float,
        join_edge: float,
        default_edge: float,
        soft_position_limit: int
    ) -> Tuple[List[Order], int, int]:

        orders: List[Order] = []
        limit = self.LIMIT[product]
        net_pos = position + buy_volume - sell_volume

        asks_above_fair = []
        bids_below_fair = []
        if order_depth:
            asks_above_fair = [p for p in order_depth.sell_orders if p > fair_value + disregard_edge]
            bids_below_fair = [p for p in order_depth.buy_orders if p < fair_value - disregard_edge]

        best_ask = min(asks_above_fair) if asks_above_fair else round(fair_value + default_edge)
        best_bid = max(bids_below_fair) if bids_below_fair else round(fair_value - default_edge)

        # If best_ask is fairly close to fair_value, place ask at best_ask, else best_ask-1
        ask_price = (best_ask if abs(best_ask - fair_value) <= join_edge 
                     else best_ask - 1)
        # If best_bid is close, place bid at best_bid, else best_bid+1
        bid_price = (best_bid if abs(fair_value - best_bid) <= join_edge 
                     else best_bid + 1)

        # Shift if near or beyond soft_position_limit
        if net_pos > soft_position_limit:
            ask_price -= 1
        elif net_pos < -soft_position_limit:
            bid_price += 1

        # Make sure we don't exceed the hard limit
        can_buy = limit - net_pos
        can_sell = limit + net_pos

        buy_qty = min(can_buy, soft_position_limit // 2)
        sell_qty = min(can_sell, soft_position_limit // 2)

        if buy_qty > 0:
            orders.append(Order(product, bid_price, buy_qty))
        if sell_qty > 0:
            orders.append(Order(product, ask_price, -sell_qty))

        return orders, buy_volume, sell_volume

    #-------------------------------------------------------------------------
    # KELP fair value logic
    def kelp_fair_value(self, od: OrderDepth, traderObject: dict) -> Optional[float]:
        product = Product.KELP
        if od and od.buy_orders and od.sell_orders:
            best_ask = min(od.sell_orders)
            best_bid = max(od.buy_orders)

            # Filter out small volume quotes if "adverse_volume" is set
            filtered_asks = [
                price for price, vol in od.sell_orders.items()
                if abs(vol) >= self.PARAMS[product]["adverse_volume"]
            ]
            filtered_bids = [
                price for price, vol in od.buy_orders.items()
                if abs(vol) >= self.PARAMS[product]["adverse_volume"]
            ]

            mm_ask = min(filtered_asks) if filtered_asks else None
            mm_bid = max(filtered_bids) if filtered_bids else None

            if mm_ask is None or mm_bid is None:
                last_price = traderObject.get("kelp_last_price")
                mid_price = (best_ask + best_bid)/2 if last_price is None else last_price
            else:
                mid_price = (mm_ask + mm_bid) / 2

            last_price = traderObject.get("kelp_last_price")
            if last_price:
                # Example reversion to the mean
                returns = (mid_price - last_price)/last_price
                pred_returns = returns * self.PARAMS[product]["reversion_beta"]
                fair_value = mid_price + mid_price * pred_returns
            else:
                fair_value = mid_price

            traderObject["kelp_last_price"] = mid_price
            return fair_value
        return None

    def kelp(self, state: TradingState) -> List[Order]:
        product = Product.KELP
        od, position = self.extract_state(state, product)
        if not od:
            return []

        params = self.PARAMS[product]
        fair_value = self.kelp_fair_value(od, self.traderObject)
        if fair_value is None:
            return []

        take_width = params["take_width"]
        clear_width = params["clear_width"]
        disregard_edge = params["disregard_edge"]
        join_edge = params["join_edge"]
        default_edge = params["default_edge"]
        soft_limit = params["soft_position_limit"]

        all_orders: List[Order] = []
        t_orders, bv, sv = self.take_orders(product, od, fair_value, take_width, position)
        all_orders += t_orders

        c_orders, bv, sv = self.clear_orders(product, od, fair_value, clear_width, position, bv, sv)
        all_orders += c_orders

        m_orders, _, _ = self.market_make(product, od, fair_value, position,
                                          bv, sv, disregard_edge, join_edge,
                                          default_edge, soft_limit)
        all_orders += m_orders
        return all_orders

    #-------------------------------------------------------------------------
    # RESIN logic
    def resin(self, state: TradingState) -> List[Order]:
        product = Product.RESIN
        od, position = self.extract_state(state, product)
        if not od:
            return []

        params = self.PARAMS[product]
        fair_value = params["fair_value"]
        take_width = params["take_width"]
        clear_width = params["clear_width"]
        disregard_edge = params["disregard_edge"]
        join_edge = params["join_edge"]
        default_edge = params["default_edge"]
        soft_limit = params["soft_position_limit"]

        all_orders: List[Order] = []
        t_orders, bv, sv = self.take_orders(product, od, fair_value, take_width, position)
        all_orders += t_orders

        c_orders, bv, sv = self.clear_orders(product, od, fair_value, clear_width, position, bv, sv)
        all_orders += c_orders

        m_orders, _, _ = self.market_make(product, od, fair_value, position,
                                          bv, sv, disregard_edge, join_edge,
                                          default_edge, soft_limit)
        all_orders += m_orders
        return all_orders

    #-------------------------------------------------------------------------
    # placeholder for SQUID_INK
    def ink(self, state: TradingState) -> List[Order]:
        return []

    #-------------------------------------------------------------------------
    # ============== NEW CAP FUNCTIONS ==================
    def cap_sell_size(self, current_position: int, limit: int, desired_sell: int) -> int:
        """
        Return the maximum *safe* SELL size that will not exceed abs(pos) > limit.
        new_position = current_position - sell_size
        must satisfy abs(new_position) <= limit
        
        => -limit <= current_position - sell_size <= limit
           sell_size <= current_position + limit
        """
        # e.g. if pos=10, limit=60 => current_position + limit=70 => can SELL up to 70
        max_sell = current_position + limit  
        if max_sell < 0:
            # Means we're over the limit on the short side, can't sell more
            return 0
        return min(desired_sell, max_sell)

    def cap_buy_size(self, current_position: int, limit: int, desired_buy: int) -> int:
        """
        Return the maximum *safe* BUY size that will not exceed abs(pos) > limit.
        new_position = current_position + buy_size
        must satisfy abs(new_position) <= limit
        
        => -limit <= current_position + buy_size <= limit
           buy_size <= limit - current_position
        """
        # e.g. if pos=-10, limit=60 => limit - pos=60 - (-10)=70 => can BUY up to 70
        max_buy = limit - current_position
        if max_buy < 0:
            return 0
        return min(desired_buy, max_buy)

    #-------------------------------------------------------------------------
    # 4) PICNIC ARB
    def picnic_arb(self, state: TradingState) -> List[Order]:
        """
        1) If spread = (Picnic1 - 1.5*Picnic2 - Djembe) > +threshold => short spread
             => SELL 2 P1, BUY 3 P2, BUY 2 DJ
        2) If spread < -threshold => long spread
             => BUY 2 P1, SELL 3 P2, SELL 2 DJ
        3) If abs(spread) < exit_threshold => attempt to flatten net positions
        """
        p_params = self.PARAMS["picnic_arb"]
        threshold = p_params["threshold"]
        exit_threshold = p_params["exit_threshold"]

        # We'll collect all new Orders here
        orders: List[Order] = []

        # Grab order depths & positions
        od_p1 = state.order_depths.get(Product.PICNIC1)
        od_p2 = state.order_depths.get(Product.PICNIC2)
        od_dj = state.order_depths.get(Product.DJEMBE)
        if not od_p1 or not od_p2 or not od_dj:
            return orders

        pos_p1 = state.position.get(Product.PICNIC1, 0)
        pos_p2 = state.position.get(Product.PICNIC2, 0)
        pos_dj = state.position.get(Product.DJEMBE, 0)

        # A midpoint helper
        def midpoint(od: OrderDepth) -> Optional[float]:
            if not od.buy_orders or not od.sell_orders:
                return None
            best_bid = max(od.buy_orders.keys())
            best_ask = min(od.sell_orders.keys())
            return 0.5 * (best_bid + best_ask)

        m1 = midpoint(od_p1)
        m2 = midpoint(od_p2)
        m_dj = midpoint(od_dj)
        if m1 is None or m2 is None or m_dj is None:
            return orders

        spread = m1 - 1.5*m2 - m_dj

        # Flatten leftover positions if spread is close to 0
        if abs(spread) < exit_threshold:
            fl_orders, pos_p1, pos_p2, pos_dj = self.flatten_picnic_positions(
                pos_p1, pos_p2, pos_dj, od_p1, od_p2, od_dj
            )
            orders.extend(fl_orders)

        # Then see if we want to open new positions
        elif spread > threshold:
            new_orders, pos_p1, pos_p2, pos_dj = self.short_spread_loop(
                pos_p1, pos_p2, pos_dj, od_p1, od_p2, od_dj
            )
            orders.extend(new_orders)

        elif spread < -threshold:
            new_orders, pos_p1, pos_p2, pos_dj = self.long_spread_loop(
                pos_p1, pos_p2, pos_dj, od_p1, od_p2, od_dj
            )
            orders.extend(new_orders)

        return orders

    #-------------------------------------------------------------------------
    # 4a) Flatten positions
    def flatten_picnic_positions(
        self, p1: int, p2: int, dj: int,
        od_p1: OrderDepth, od_p2: OrderDepth, od_dj: OrderDepth
    ) -> Tuple[List[Order], int, int, int]:
        flatten_orders: List[Order] = []

        def flatten_product(symbol: str, od: OrderDepth, pos: int, limit: int) -> Tuple[List[Order], int]:
            orders: List[Order] = []
            remaining = pos
            if remaining == 0:
                return orders, 0

            if remaining > 0:
                # SELL to flatten
                # how many we can safely sell
                safe_sell = self.cap_sell_size(pos, limit, remaining)
                if safe_sell <= 0:
                    return orders, remaining
                fill, sells = self.feasible_sell(od, safe_sell, symbol)
                if fill > 0:
                    new_pos = pos - fill
                    self.reduce_sell_vol(od, sells)
                    orders.extend(sells)
                    remaining = new_pos
            else:
                # BUY to flatten negative pos
                safe_buy = self.cap_buy_size(pos, limit, -remaining)  # -remaining is how many we want
                if safe_buy <= 0:
                    return orders, remaining
                fill, buys = self.feasible_buy(od, safe_buy, symbol)
                if fill > 0:
                    new_pos = pos + fill
                    self.reduce_buy_vol(od, buys)
                    orders.extend(buys)
                    remaining = new_pos

            return orders, remaining

        # Flatten P1
        p1_orders, p1 = flatten_product(Product.PICNIC1, od_p1, p1, self.LIMIT[Product.PICNIC1])
        flatten_orders.extend(p1_orders)

        # Flatten P2
        p2_orders, p2 = flatten_product(Product.PICNIC2, od_p2, p2, self.LIMIT[Product.PICNIC2])
        flatten_orders.extend(p2_orders)

        # Flatten DJ
        dj_orders, dj = flatten_product(Product.DJEMBE, od_dj, dj, self.LIMIT[Product.DJEMBE])
        flatten_orders.extend(dj_orders)

        return flatten_orders, p1, p2, dj

    #-------------------------------------------------------------------------
    # 4b) short_spread_loop (example repeated approach)
    def short_spread_loop(
        self, p1: int, p2: int, dj: int,
        od_p1: OrderDepth, od_p2: OrderDepth, od_dj: OrderDepth
    ) -> Tuple[List[Order], int, int, int]:
        result: List[Order] = []
        while True:
            # Attempt one chunk: SELL 2 P1, BUY 3 P2, BUY 2 DJ
            new_orders, p1, p2, dj = self.short_spread_once(p1, p2, dj, od_p1, od_p2, od_dj)
            if not new_orders:
                break
            result.extend(new_orders)
        return result, p1, p2, dj

    #-------------------------------------------------------------------------
    # 4c) short_spread_once
    def short_spread_once(
        self, p1: int, p2: int, dj: int, 
        od_p1: OrderDepth, od_p2: OrderDepth, od_dj: OrderDepth
    ) -> Tuple[List[Order], int, int, int]:
        """
        Attempt to SELL 2 x PICNIC1, BUY 3 x PICNIC2, BUY 2 x DJEMBE,
        capping the trade so we never exceed the limit.
        """
        limit_p1 = self.LIMIT[Product.PICNIC1]
        limit_p2 = self.LIMIT[Product.PICNIC2]
        limit_dj = self.LIMIT[Product.DJEMBE]

        # 1) Check how many we can SELL for PICNIC1 (max 2)
        desired_sell_p1 = 2
        safe_sell_p1 = self.cap_sell_size(p1, limit_p1, desired_sell_p1)
        if safe_sell_p1 <= 0:
            return [], p1, p2, dj

        # 2) Check how many we can BUY for PICNIC2 (max 3)
        desired_buy_p2 = 3
        safe_buy_p2 = self.cap_buy_size(p2, limit_p2, desired_buy_p2)
        if safe_buy_p2 <= 0:
            return [], p1, p2, dj

        # 3) Check how many we can BUY for DJEMBE (max 2)
        desired_buy_dj = 2
        safe_buy_dj = self.cap_buy_size(dj, limit_dj, desired_buy_dj)
        if safe_buy_dj <= 0:
            return [], p1, p2, dj

        # Attempt partial or all-or-none:
        # We'll do partial if, say, safe_sell_p1=1 but we wanted 2
        fill_p1, sell_p1 = self.feasible_sell(od_p1, safe_sell_p1, Product.PICNIC1)
        fill_p2, buy_p2  = self.feasible_buy(od_p2, safe_buy_p2, Product.PICNIC2)
        fill_dj, buy_dj  = self.feasible_buy(od_dj, safe_buy_dj, Product.DJEMBE)

        if fill_p1 == 0 or fill_p2 == 0 or fill_dj == 0:
            # can't fill anything => no trade
            return [], p1, p2, dj

        new_p1 = p1 - fill_p1
        new_p2 = p2 + fill_p2
        new_dj = dj + fill_dj

        # reduce volumes in the order book
        self.reduce_sell_vol(od_p1, sell_p1)  # matched buy side
        self.reduce_buy_vol(od_p2, buy_p2)
        self.reduce_buy_vol(od_dj, buy_dj)

        all_orders = []
        all_orders.extend(sell_p1)
        all_orders.extend(buy_p2)
        all_orders.extend(buy_dj)

        return all_orders, new_p1, new_p2, new_dj

    #-------------------------------------------------------------------------
    # 4d) long_spread_loop
    def long_spread_loop(
        self, p1: int, p2: int, dj: int,
        od_p1: OrderDepth, od_p2: OrderDepth, od_dj: OrderDepth
    ) -> Tuple[List[Order], int, int, int]:
        result: List[Order] = []
        while True:
            # Attempt one chunk: BUY 2 P1, SELL 3 P2, SELL 2 DJ
            new_orders, p1, p2, dj = self.long_spread_once(p1, p2, dj, od_p1, od_p2, od_dj)
            if not new_orders:
                break
            result.extend(new_orders)
        return result, p1, p2, dj

    #-------------------------------------------------------------------------
    # 4e) long_spread_once
    def long_spread_once(
        self, p1: int, p2: int, dj: int,
        od_p1: OrderDepth, od_p2: OrderDepth, od_dj: OrderDepth
    ) -> Tuple[List[Order], int, int, int]:
        """
        Attempt to BUY 2 x PICNIC1, SELL 3 x PICNIC2, SELL 2 x DJEMBE,
        capping the trade so we never exceed limit.
        """
        limit_p1 = self.LIMIT[Product.PICNIC1]
        limit_p2 = self.LIMIT[Product.PICNIC2]
        limit_dj = self.LIMIT[Product.DJEMBE]

        # 1) feasible BUY for P1
        desired_buy_p1 = 2
        safe_buy_p1 = self.cap_buy_size(p1, limit_p1, desired_buy_p1)
        if safe_buy_p1 <= 0:
            return [], p1, p2, dj

        # 2) feasible SELL for P2
        desired_sell_p2 = 3
        safe_sell_p2 = self.cap_sell_size(p2, limit_p2, desired_sell_p2)
        if safe_sell_p2 <= 0:
            return [], p1, p2, dj

        # 3) feasible SELL for DJ
        desired_sell_dj = 2
        safe_sell_dj = self.cap_sell_size(dj, limit_dj, desired_sell_dj)
        if safe_sell_dj <= 0:
            return [], p1, p2, dj

        fill_p1, buy_p1   = self.feasible_buy(od_p1, safe_buy_p1, Product.PICNIC1)
        fill_p2, sell_p2  = self.feasible_sell(od_p2, safe_sell_p2, Product.PICNIC2)
        fill_dj, sell_dj  = self.feasible_sell(od_dj, safe_sell_dj, Product.DJEMBE)

        if fill_p1 == 0 or fill_p2 == 0 or fill_dj == 0:
            return [], p1, p2, dj

        new_p1 = p1 + fill_p1
        new_p2 = p2 - fill_p2
        new_dj = dj - fill_dj

        self.reduce_buy_vol(od_p1, buy_p1)
        self.reduce_sell_vol(od_p2, sell_p2)
        self.reduce_sell_vol(od_dj, sell_dj)

        all_orders = []
        all_orders.extend(buy_p1)
        all_orders.extend(sell_p2)
        all_orders.extend(sell_dj)
        return all_orders, new_p1, new_p2, new_dj

    #-------------------------------------------------------------------------
    # Helpers for crossing best bids/asks in picnic arb
    def feasible_sell(self, od: OrderDepth, need: int, symbol: str) -> Tuple[int, List[Order]]:
        """
        Attempt to SELL 'need' units of 'symbol' at the best bids.
        Return (filled, orders_list).
        """
        if not od or not od.buy_orders:
            return 0, []
        fill_count = 0
        sells: List[Order] = []
        # sort descending by price
        bids = sorted(od.buy_orders.items(), key=lambda x: x[0], reverse=True)
        for price, bid_vol in bids:
            if bid_vol <= 0:
                continue
            to_fill = need - fill_count
            if to_fill <= 0:
                break
            can_fill = min(to_fill, bid_vol)
            if can_fill > 0:
                sells.append(Order(symbol, price, -can_fill))
                fill_count += can_fill
        return fill_count, sells

    def feasible_buy(self, od: OrderDepth, need: int, symbol: str) -> Tuple[int, List[Order]]:
        """
        Attempt to BUY 'need' units of 'symbol' at the best asks.
        Return (filled, orders_list).
        """
        if not od or not od.sell_orders:
            return 0, []
        fill_count = 0
        buys: List[Order] = []
        # sort ascending by price
        asks = sorted(od.sell_orders.items(), key=lambda x: x[0])
        for price, ask_vol in asks:
            vol_avail = -ask_vol  # negative => +volume
            if vol_avail <= 0:
                continue
            to_fill = need - fill_count
            if to_fill <= 0:
                break
            can_fill = min(to_fill, vol_avail)
            if can_fill > 0:
                buys.append(Order(symbol, price, can_fill))
                fill_count += can_fill
        return fill_count, buys

    #-------------------------------------------------------------------------
    # Reduce volumes from the OrderDepth after a SELL
    def reduce_sell_vol(self, od: OrderDepth, sell_orders: List[Order]):
        """
        For each SELL order in 'sell_orders', we matched the buy side in od.buy_orders.
        Subtract that matched volume from the relevant buy_orders.
        """
        for so in sell_orders:
            matched_qty = -so.quantity  # SELL => negative quantity
            if so.price in od.buy_orders:
                od.buy_orders[so.price] -= matched_qty
                # if that volume is used up, you could remove the dict entry

    #-------------------------------------------------------------------------
    # Reduce volumes from the OrderDepth after a BUY
    def reduce_buy_vol(self, od: OrderDepth, buy_orders: List[Order]):
        """
        For each BUY order in 'buy_orders', we matched the sell side in od.sell_orders.
        Subtract that matched volume from the relevant sell_orders (which are negative in dict).
        """
        for bo in buy_orders:
            if bo.price in od.sell_orders:
                od.sell_orders[bo.price] += bo.quantity
                # if that volume is used up, you could remove the dict entry

    #-------------------------------------------------------------------------
    # RUN method: merges all strategies and returns final dictionary
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        # 1) Load any persistent data
        if state.traderData:
            self.traderObject = jsonpickle.decode(state.traderData)
        else:
            self.traderObject = {}

        # 2) Compute orders for KELP, RESIN, INK, etc.
        kelp_orders = self.kelp(state)
        resin_orders = self.resin(state)
        ink_orders = self.ink(state)

        # 3) Put these in a dictionary keyed by product
        orders: Dict[str, List[Order]] = {
            Product.KELP: kelp_orders,
            Product.RESIN: resin_orders,
            Product.INK: ink_orders
        }

        # 4) Generate picnic_arb orders (for PICNIC1, PICNIC2, DJEMBE)
        picnic_arb_orders = self.picnic_arb(state)

        # 5) Distribute them by product
        arb_by_product: Dict[str, List[Order]] = {
            Product.PICNIC1: [],
            Product.PICNIC2: [],
            Product.DJEMBE: []
        }
        for o in picnic_arb_orders:
            if o.symbol in arb_by_product:
                arb_by_product[o.symbol].append(o)

        # 6) Merge them into our final 'orders' dictionary
        for prod in arb_by_product:
            if len(arb_by_product[prod]) > 0:
                if prod not in orders:
                    orders[prod] = []
                orders[prod].extend(arb_by_product[prod])

        # 7) Prepare final return
        conversions = 1
        traderData = jsonpickle.encode(self.traderObject)
        return orders, conversions, traderData