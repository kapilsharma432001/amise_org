from pydantic import BaseModel, Field
from tools.base import BaseTool
import yfinance as yf
import asyncio


class StockQuoteSchema(BaseModel):
    ticker: str = Field(..., description="The stock ticker symbol (e.g., AAPL, TSLA).")

class StockQuoteTool(BaseTool):
    name = "stock_quote"
    description = "Fetches the current stock price and volume for a given ticker."
    args_schema = StockQuoteSchema

    async def execute(self, ticker: str) -> str:
        print(f"[Network] Fetching live quote for: {ticker}")

        def fetch():
            stock = yf.Ticker(ticker)
            data = stock.info
            price = data.get("currentPrice")
            volume = data.get("volume")
            return price, volume

        price, volume = await asyncio.to_thread(fetch)

        if price is None:
            return f"Could not fetch data for {ticker}"

        return f"{ticker} is trading at ${price}, volume: {volume}"
    
