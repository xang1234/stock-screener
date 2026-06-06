/**
 * Calculate EMA (Exponential Moving Average)
 */
export const calculateEMA = (data, period) => {
  if (!data || data.length < period) return [];

  const k = 2 / (period + 1);
  const emaData = [];

  // Calculate initial SMA as first EMA value
  let ema = 0;
  for (let i = 0; i < period; i++) {
    ema += data[i].close;
  }
  ema = ema / period;
  emaData.push({ time: data[period - 1].date, value: ema });

  // Calculate EMA for remaining data
  for (let i = period; i < data.length; i++) {
    ema = data[i].close * k + ema * (1 - k);
    emaData.push({ time: data[i].date, value: ema });
  }

  return emaData;
};

/**
 * Aggregate daily data to weekly
 */
export const aggregateToWeekly = (dailyData) => {
  if (!dailyData || dailyData.length === 0) return [];

  const weeklyData = [];
  let currentWeek = null;

  dailyData.forEach((day) => {
    const date = new Date(day.date);
    const weekStart = new Date(date);
    weekStart.setDate(date.getDate() - date.getDay()); // Start of week (Sunday)
    const weekKey = weekStart.toISOString().split('T')[0];

    if (!currentWeek || currentWeek.weekKey !== weekKey) {
      if (currentWeek) {
        weeklyData.push(currentWeek.data);
      }
      currentWeek = {
        weekKey,
        data: {
          date: day.date,
          open: day.open,
          high: day.high,
          low: day.low,
          close: day.close,
          volume: day.volume,
        }
      };
    } else {
      currentWeek.data.high = Math.max(currentWeek.data.high, day.high);
      currentWeek.data.low = Math.min(currentWeek.data.low, day.low);
      currentWeek.data.close = day.close;
      currentWeek.data.volume += day.volume;
      currentWeek.data.date = day.date; // Use latest date for the week
    }
  });

  if (currentWeek) {
    weeklyData.push(currentWeek.data);
  }

  return weeklyData;
};

/**
 * Transform API data to TradingView Lightweight Charts format
 */
export const transformToCandlestickData = (apiData, timeframe = 'daily') => {
  if (!apiData || apiData.length === 0) {
    return { candlesticks: [], volume: [], ema10: [], ema20: [], ema50: [] };
  }

  // Aggregate to weekly if needed
  const processedData = timeframe === 'weekly' ? aggregateToWeekly(apiData) : apiData;

  const candlesticks = [];
  const volume = [];

  processedData.forEach((d) => {
    // Candlestick data
    candlesticks.push({
      time: d.date,
      open: d.open,
      high: d.high,
      low: d.low,
      close: d.close,
    });

    // Volume data
    volume.push({
      time: d.date,
      value: d.volume,
      color: d.close >= d.open ? 'rgba(33, 150, 243, 0.5)' : 'rgba(230, 25, 205, 0.5)',
    });
  });

  // Calculate EMAs
  const ema10 = calculateEMA(processedData, 10);
  const ema20 = calculateEMA(processedData, 20);
  const ema50 = calculateEMA(processedData, 50);

  return { candlesticks, volume, ema10, ema20, ema50 };
};
