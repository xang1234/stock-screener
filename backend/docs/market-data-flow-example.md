# Market Data Flow Example

This document shows a concrete end-to-end example of how market data is fetched and stored in this backend.

## Example: fetching and using stock info for AAPL

### 1. Entry point: `YFinanceService.get_stock_info`
The method in [backend/app/services/yfinance_service.py](backend/app/services/yfinance_service.py) is used to fetch basic stock metadata from Yahoo Finance.

```python
from app.services.yfinance_service import YFinanceService

service = YFinanceService()
data = service.get_stock_info("AAPL")
```

### 2. What the method does internally
Inside `get_stock_info`, the service:

1. Applies rate limiting through `_wait_for_yfinance_rate_limit()`.
2. Creates a yfinance ticker object:
   ```python
   ticker = yf.Ticker(symbol)
   ```
3. Fetches metadata using `ticker.info`.
4. Validates that the response is meaningful.
5. Returns a normalized dictionary.

### 3. Example returned payload
The method returns a dictionary similar to:

```python
{
    "symbol": "AAPL",
    "name": "Apple Inc.",
    "sector": "Technology",
    "industry": "Consumer Electronics",
    "market_cap": 3000000000000,
    "current_price": 200.5,
    "pe_ratio": 30.2,
    "price_to_book": 45.7,
    "shares_outstanding": 15000000000,
    "institutional_ownership": 58.3
}
```

## Example: fetching historical price data
A second common flow is `get_historical_data`, also in [backend/app/services/yfinance_service.py](backend/app/services/yfinance_service.py).

```python
price_df = service.get_historical_data("AAPL", period="2y")
```

### What happens here
1. The service first checks whether price data is already available in the cache layer.
2. If cache is available, it returns that cached data.
3. If not, it directly calls Yahoo Finance using:
   ```python
   ticker.history(period=period, interval=interval)
   ```
4. The returned data is a Pandas DataFrame with OHLCV columns such as:
   - `Open`
   - `High`
   - `Low`
   - `Close`
   - `Volume`

## Where the data is stored

### A. Fast access layer: Redis
The hot cache is managed by:
- [backend/app/services/price_cache_service.py](backend/app/services/price_cache_service.py)
- [backend/app/services/fundamentals_cache_service.py](backend/app/services/fundamentals_cache_service.py)

These services keep recently used price and fundamentals data in Redis for fast reuse.

### B. Durable storage: PostgreSQL
The long-term persistence model is defined in [backend/app/models/stock.py](backend/app/models/stock.py).

It uses models such as:
- `StockPrice` → persisted in `stock_prices`
- `StockFundamental` → persisted in `stock_fundamentals`

### C. Background refresh pipeline
Bulk refreshes and re-caching are handled by Celery tasks in [backend/app/tasks/cache_tasks.py](backend/app/tasks/cache_tasks.py).

These tasks:
1. Pull market data for many symbols.
2. Store it in Redis.
3. Persist it to PostgreSQL in batches.

## End-to-end flow summary

```text
User/API request
    -> YFinanceService
    -> live fetch from Yahoo Finance
    -> optional cache check (Redis)
    -> optional persistence to PostgreSQL
    -> downstream scanner / UI consumption
```

## Exact call chain example: from API request to fetcher

Here is the full path for a single-symbol request such as `/api/v1/stocks/AAPL/info`.

### 1. Client sends a request
A frontend, script, or another backend service calls an endpoint such as:

```text
/api/v1/stocks/AAPL/info
```

### 2. FastAPI captures the symbol from the URL
In [backend/app/api/v1/stocks.py](backend/app/api/v1/stocks.py), the route signature receives the path parameter:

```python
async def get_stock_info(symbol: str = Depends(require_valid_symbol)):
```

So the value `AAPL` is captured as the argument `symbol`.

### 3. The symbol is validated and normalized
The dependency in [backend/app/services/symbol_format.py](backend/app/services/symbol_format.py) runs:

```python
normalized = normalize_symbol(symbol)
```

This step:
- trims whitespace
- uppercases the ticker
- checks the format
- rejects invalid values early

For example:

```python
" aapl " -> "AAPL"
```

### 4. The normalized symbol is passed to the service layer
The route then calls the helper that reaches the fetcher:

```python
return _get_stock_info_or_404(symbol)
```

That helper uses the yfinance service:

```python
info = _get_yfinance_service().get_stock_info(symbol.upper())
```

### 5. The yfinance service fetches the data
In [backend/app/services/yfinance_service.py](backend/app/services/yfinance_service.py), the method receives the symbol and performs the live fetch:

```python
def get_stock_info(self, symbol: str) -> Optional[Dict[str, Any]]:
    ticker = yf.Ticker(symbol)
    info = ticker.info
```

### 6. Result
The symbol has now traveled through the following chain:

```text
Client request
    -> FastAPI route
    -> require_valid_symbol()
    -> normalized symbol
    -> YFinanceService.get_stock_info(symbol)
    -> yf.Ticker(symbol)
    -> live Yahoo Finance fetch
```

## Who provides the ticker in bulk jobs?
For background refreshes and scans, the ticker is usually not coming from a user request. Instead, the backend pulls it from:
- the stock universe database
- a scan request payload
- a batch refresh list

Then it loops over those symbols and calls the same fetch layer for each one.

## Short version
- Source: Yahoo Finance (and occasionally other market-specific providers)
- Fast cache: Redis
- Durable storage: PostgreSQL tables such as `stock_prices` and `stock_fundamentals`
- Refreshes: handled by Celery background tasks
