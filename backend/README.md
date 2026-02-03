# Stock Scanner Backend API

FastAPI backend for CANSLIM + Minervini stock scanner with industry group analysis.

## Features

- **Stock Data Fetching**: yfinance for price/volume, Alpha Vantage for fundamentals
- **Rate Limiting**: Respects API rate limits (1 req/sec yfinance, 25 req/day Alpha Vantage)
- **Caching**: SQLite database caching to minimize API calls
- **RESTful API**: FastAPI with automatic OpenAPI documentation

## Setup

### 1. Create Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Copy `.env.example` to `.env` and configure:

```bash
cp .env.example .env
```

Edit `.env` and add your Alpha Vantage API key (optional, for detailed fundamentals):

```
ALPHA_VANTAGE_API_KEY=your_key_here
```

If you want to use the config endpoints (`/api/v1/config/*`), set an admin key:

```
ADMIN_API_KEY=your_admin_key
```

Optional safety and ops settings:

```
# Limit how much data read_url can download
RESEARCH_READ_URL_MAX_BYTES=5000000

# Max time to wait for the data fetch lock
DATA_FETCH_LOCK_WAIT_SECONDS=7200

# Enable one-time cleanup for legacy scan universes
INVALID_UNIVERSE_CLEANUP_ENABLED=false
```

### 4. Initialize Database

The database will be created automatically on first run, or you can initialize it manually:

```python
python -c "from app.database import init_db; init_db()"
```

### 5. Run Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Server will start at `http://localhost:8000`

## API Documentation

Once the server is running, visit:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Available Endpoints

### Health & Info
- `GET /` - API information
- `GET /health` - Health check

### Stock Data
- `GET /api/v1/stocks/{symbol}/info` - Basic stock information
- `GET /api/v1/stocks/{symbol}/fundamentals` - Fundamental data (EPS, revenue, margins)
- `GET /api/v1/stocks/{symbol}/technicals` - Technical indicators (MAs, RS rating, 52-week range)
- `GET /api/v1/stocks/{symbol}` - Complete stock data (info + fundamentals + technicals)
- `GET /api/v1/stocks/{symbol}/industry` - Industry classification

## Example Usage

### Get Stock Info

```bash
curl http://localhost:8000/api/v1/stocks/AAPL/info
```

### Get Fundamentals

```bash
curl "http://localhost:8000/api/v1/stocks/AAPL/fundamentals?use_alpha_vantage=false"
```

### Get Complete Data

```bash
curl http://localhost:8000/api/v1/stocks/AAPL
```

## Database Schema

SQLite database stored in `../data/stockscanner.db` with tables for:
- `stock_prices` - Historical OHLCV data
- `stock_fundamentals` - EPS, revenue, institutional ownership
- `stock_technicals` - Moving averages, RS rating, stage, VCP score
- `stock_industry` - Industry classification (GICS sectors/industries)
- `scan_results` - Stock scan results
- `watchlist` - User watchlist
- `market_status` - Daily market trend data

## Rate Limiting

- **yfinance**: 1 request/second (self-imposed, respectful limit)
- **Alpha Vantage Free**: 25 requests/day (hard limit)
  - Fundamentals cached for 7 days
  - Use `use_alpha_vantage=false` to rely on yfinance data

## Next Steps (Phase 2)

- Implement technical analysis calculations (RS rating, Weinstein stages, VCP detection)
- Implement CANSLIM scoring algorithms
- Implement Minervini template criteria
- Add stock scanning endpoints
- Industry group analysis endpoints

## Development

### Run Tests

```bash
pytest tests/
```

### Code Structure

```
app/
├── main.py              # FastAPI application
├── config.py            # Configuration settings
├── database.py          # Database setup
├── models/              # SQLAlchemy models
├── schemas/             # Pydantic schemas
├── api/v1/              # API endpoints
├── services/            # Data fetching services
├── scanners/            # Scanning algorithms (Phase 2+)
├── industry/            # Industry analysis (Phase 7)
└── utils/               # Utilities (rate limiter, etc.)
```

## License

MIT
