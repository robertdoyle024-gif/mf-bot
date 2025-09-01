
from alpaca_trade_api.rest import REST
from dataclasses import dataclass

@dataclass
class Broker:
    api: REST

    def equity(self) -> float:
        return float(self.api.get_account().equity)

    def position_qty(self, symbol: str) -> float:
        try:
            pos = self.api.get_position(symbol)
            return float(pos.qty)
        except Exception:
            return 0.0

    def market_buy(self, symbol: str, qty: int):
        return self.api.submit_order(symbol=symbol, qty=qty, side="buy", type="market", time_in_force="day")

    def market_sell(self, symbol: str, qty: int):
        return self.api.submit_order(symbol=symbol, qty=qty, side="sell", type="market", time_in_force="day")
