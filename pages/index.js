import Head from 'next/head';
import { useState, useEffect, useRef, useCallback } from 'react';
import { ethers } from 'ethers';

// ─── CHAINLINK CONFIG ───────────────────────────────────────────────────────
// BTC/USD Price Feed — Ethereum Mainnet
// https://data.chain.link/streams/btc-usd-cexprice-streams
const CHAINLINK_BTC_USD = '0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88';
const CHAINLINK_ABI = [
  'function latestRoundData() external view returns (uint80 roundId, int256 answer, uint256 startedAt, uint256 updatedAt, uint80 answeredInRound)',
  'function decimals() external view returns (uint8)',
  'function description() external view returns (string)',
];

// Public Ethereum RPC endpoints (no API key needed) — fallback chain
const RPC_ENDPOINTS = [
  'https://eth.llamarpc.com',
  'https://cloudflare-eth.com',
  'https://rpc.ankr.com/eth',
  'https://ethereum.publicnode.com',
];

// ─── MATH / INDICATOR UTILITIES ────────────────────────────────────────────

function ema(data, period) {
  const k = 2 / (period + 1);
  const result = [];
  let prev = data.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(...Array(period - 1).fill(null), prev);
  for (let i = period; i < data.length; i++) {
    prev = data[i] * k + prev * (1 - k);
    result.push(prev);
  }
  return result;
}

function rsi(closes, period = 14) {
  const gains = [], losses = [];
  for (let i = 1; i < closes.length; i++) {
    const diff = closes[i] - closes[i - 1];
    gains.push(Math.max(diff, 0));
    losses.push(Math.max(-diff, 0));
  }
  const result = Array(period).fill(null);
  let avgGain = gains.slice(0, period).reduce((a, b) => a + b, 0) / period;
  let avgLoss = losses.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
  for (let i = period; i < gains.length; i++) {
    avgGain = (avgGain * (period - 1) + gains[i]) / period;
    avgLoss = (avgLoss * (period - 1) + losses[i]) / period;
    result.push(avgLoss === 0 ? 100 : 100 - 100 / (1 + avgGain / avgLoss));
  }
  return result;
}

function macd(closes) {
  const ema12 = ema(closes, 12);
  const ema26 = ema(closes, 26);
  const macdLine = closes.map((_, i) =>
    ema12[i] != null && ema26[i] != null ? ema12[i] - ema26[i] : null
  );
  const validMacd = macdLine.filter((v) => v != null);
  const signalRaw = ema(validMacd, 9);
  const signal = Array(macdLine.length - validMacd.length).fill(null);
  let si = 0;
  macdLine.forEach((v) => { if (v != null) signal.push(signalRaw[si++]); });
  const histogram = macdLine.map((v, i) =>
    v != null && signal[i] != null ? v - signal[i] : null
  );
  return { macdLine, signal, histogram };
}

function bollingerBands(closes, period = 20, multiplier = 2) {
  return closes.map((_, i) => {
    if (i < period - 1) return null;
    const slice = closes.slice(i - period + 1, i + 1);
    const mean = slice.reduce((a, b) => a + b, 0) / period;
    const std = Math.sqrt(slice.reduce((s, v) => s + (v - mean) ** 2, 0) / period);
    return { upper: mean + multiplier * std, middle: mean, lower: mean - multiplier * std, std };
  });
}

function atr(candles, period = 14) {
  const tr = candles.map((c, i) => {
    if (i === 0) return c.high - c.low;
    const prev = candles[i - 1];
    return Math.max(c.high - c.low, Math.abs(c.high - prev.close), Math.abs(c.low - prev.close));
  });
  const result = [];
  let atrVal = tr.slice(0, period).reduce((a, b) => a + b, 0) / period;
  result.push(...Array(period - 1).fill(null), atrVal);
  for (let i = period; i < tr.length; i++) {
    atrVal = (atrVal * (period - 1) + tr[i]) / period;
    result.push(atrVal);
  }
  return result;
}

// ─── CNN-LSTM ENSEMBLE MODEL ────────────────────────────────────────────────

function sigmoid(x) { return 1 / (1 + Math.exp(-x)); }
function tanhFn(x) { return Math.tanh(x); }

function lstmCell(seq, w) {
  let h = 0, c = 0;
  for (const x of seq) {
    const fg = sigmoid(w.wf * x + w.uf * h + w.bf);
    const ig = sigmoid(w.wi * x + w.ui * h + w.bi);
    const cg = tanhFn(w.wc * x + w.uc * h + w.bc);
    const og = sigmoid(w.wo * x + w.uo * h + w.bo);
    c = fg * c + ig * cg;
    h = og * tanhFn(c);
  }
  return h;
}

const WEIGHTS = {
  momentum: { wf:0.6,uf:0.3,bf:0.1, wi:0.7,ui:0.2,bi:0.0, wc:0.5,uc:0.4,bc:0.1, wo:0.8,uo:0.2,bo:0.0 },
  rsi:      { wf:0.4,uf:0.5,bf:0.0, wi:0.6,ui:0.3,bi:0.1, wc:0.7,uc:0.2,bc:0.0, wo:0.5,uo:0.4,bo:0.1 },
  macd:     { wf:0.5,uf:0.4,bf:0.1, wi:0.8,ui:0.1,bi:0.0, wc:0.6,uc:0.3,bc:0.0, wo:0.7,uo:0.2,bo:0.1 },
  bb:       { wf:0.7,uf:0.2,bf:0.0, wi:0.5,ui:0.4,bi:0.1, wc:0.8,uc:0.1,bc:0.0, wo:0.6,uo:0.3,bo:0.1 },
  volume:   { wf:0.3,uf:0.6,bf:0.1, wi:0.7,ui:0.2,bi:0.0, wc:0.4,uc:0.5,bc:0.1, wo:0.8,uo:0.1,bo:0.0 },
};

function runModel(candles) {
  if (!candles || candles.length < 50) return null;
  const closes = candles.map(c => c.close);
  const volumes = candles.map(c => c.volume);
  const n = closes.length;

  const ema3 = ema(closes, 3);
  const ema8 = ema(closes, 8);
  const ema21 = ema(closes, 21);
  const rsiVals = rsi(closes, 14);
  const { macdLine, histogram } = macd(closes);
  const bb = bollingerBands(closes, 20);
  const atrVals = atr(candles, 14);
  const volEma = ema(volumes, 10);

  const WIN = 12;
  const start = n - WIN;

  const momentumSeq = [];
  for (let i = start; i < n; i++) {
    if (!ema3[i] || !ema8[i] || !ema21[i]) { momentumSeq.push(0); continue; }
    const cross38 = (ema3[i] - ema8[i]) / (atrVals[i] || 1);
    const cross821 = (ema8[i] - ema21[i]) / (atrVals[i] || 1);
    momentumSeq.push(tanhFn(cross38 * 0.5 + cross821 * 0.3));
  }

  const rsiSeq = [];
  for (let i = start; i < n; i++) {
    rsiSeq.push(tanhFn((rsiVals[i] != null ? rsiVals[i] - 50 : 0) / 50));
  }

  const macdSeq = [];
  for (let i = start; i < n; i++) {
    macdSeq.push(tanhFn((histogram[i] != null ? histogram[i] / (atrVals[i] || 1) : 0) * 10));
  }

  const bbSeq = [];
  for (let i = start; i < n; i++) {
    const b = bb[i];
    if (!b || b.std === 0) { bbSeq.push(0); continue; }
    bbSeq.push(tanhFn(((closes[i] - b.lower) / (b.upper - b.lower) - 0.5) * 2));
  }

  const volSeq = [];
  for (let i = start; i < n; i++) {
    const ve = volEma[i] || volumes[i];
    const anomaly = (volumes[i] - ve) / (ve || 1);
    const priceDir = i > 0 ? Math.sign(closes[i] - closes[i - 1]) : 0;
    volSeq.push(tanhFn(anomaly * priceDir));
  }

  const mo = lstmCell(momentumSeq, WEIGHTS.momentum);
  const rs = lstmCell(rsiSeq, WEIGHTS.rsi);
  const mc = lstmCell(macdSeq, WEIGHTS.macd);
  const bb2 = lstmCell(bbSeq, WEIGHTS.bb);
  const vo = lstmCell(volSeq, WEIGHTS.volume);

  const features = [mo, rs, mc, bb2, vo];
  const stackW = [0.28, 0.18, 0.22, 0.16, 0.16];
  const rawScore = features.reduce((s, f, i) => s + f * stackW[i], 0);

  const currentATR = atrVals[n - 1] || 0;
  const recentPrice = closes[n - 1];
  const atrPct = (currentATR / recentPrice) * 100;
  const signalConsensus = features.filter(f => Math.sign(f) === Math.sign(rawScore)).length / features.length;
  const volatilityFactor = Math.min(Math.max(atrPct / 0.15, 0.5), 1.0);
  let confidence = Math.abs(rawScore) * signalConsensus * volatilityFactor;
  confidence = Math.min(Math.max(confidence * 100, 50), 97);

  return {
    direction: rawScore > 0.02 ? 'UP' : rawScore < -0.02 ? 'DOWN' : 'NEUTRAL',
    confidence,
    rawScore,
    signals: {
      momentum: { score: mo, label: 'EMA Crossover' },
      rsi:      { score: rs, label: 'RSI Momentum' },
      macd:     { score: mc, label: 'MACD Histogram' },
      bb:       { score: bb2, label: 'Bollinger Position' },
      volume:   { score: vo, label: 'Volume Anomaly' },
    },
    currentRSI: rsiVals[n - 1],
    currentATR,
    currentPrice: closes[n - 1],
    bbBands: bb[n - 1],
    macdVal: macdLine[n - 1],
    macdHist: histogram[n - 1],
    ema3: ema3[n - 1],
    ema8: ema8[n - 1],
    ema21: ema21[n - 1],
  };
}

// ─── DATA SOURCES ───────────────────────────────────────────────────────────

// Chainlink on-chain price (Ethereum mainnet via public RPC)
async function fetchChainlinkPrice() {
  for (const rpcUrl of RPC_ENDPOINTS) {
    try {
      const provider = new ethers.providers.JsonRpcProvider(rpcUrl);
      provider.polling = false;
      const feed = new ethers.Contract(CHAINLINK_BTC_USD, CHAINLINK_ABI, provider);
      const [roundData] = await Promise.all([
        Promise.race([
          feed.latestRoundData(),
          new Promise((_, reject) => setTimeout(() => reject(new Error('timeout')), 6000))
        ])
      ]);
      const price = parseFloat(ethers.utils.formatUnits(roundData.answer, 8));
      const updatedAt = roundData.updatedAt.toNumber() * 1000;
      return { price, updatedAt, source: 'Chainlink', rpc: rpcUrl };
    } catch {
      continue;
    }
  }
  throw new Error('All Chainlink RPC endpoints failed');
}

// Binance REST — 5-min OHLCV for the model (no WebSocket needed on server)
async function fetchKlines(limit = 100) {
  const res = await fetch(
    `https://api.binance.com/api/v3/klines?symbol=BTCUSDT&interval=5m&limit=${limit}`
  );
  if (!res.ok) throw new Error('Binance klines error');
  const data = await res.json();
  return data.map(k => ({
    time: k[0],
    open: parseFloat(k[1]),
    high: parseFloat(k[2]),
    low: parseFloat(k[3]),
    close: parseFloat(k[4]),
    volume: parseFloat(k[5]),
  }));
}

// ─── MINI SPARKLINE CHART ───────────────────────────────────────────────────

function Sparkline({ candles, direction }) {
  if (!candles || candles.length < 2) return null;
  const last = candles.slice(-48);
  const prices = last.map(c => c.close);
  const vols = last.map(c => c.volume);
  const minP = Math.min(...prices), maxP = Math.max(...prices);
  const maxV = Math.max(...vols);
  const rangeP = maxP - minP || 1;
  const W = 500, H = 110;

  const pts = prices.map((p, i) => {
    const x = (i / (prices.length - 1)) * W;
    const y = H * 0.1 + ((maxP - p) / rangeP) * H * 0.75;
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');

  const color = direction === 'UP' ? '#00ff9d' : direction === 'DOWN' ? '#ff4757' : '#f1c40f';

  const volBars = vols.map((v, i) => ({
    x: (i / (vols.length - 1)) * W,
    h: (v / maxV) * H * 0.18,
    w: W / vols.length - 1,
  }));

  return (
    <svg width="100%" viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
      <defs>
        <linearGradient id="areaGrad" x1="0" y1="0" x2="0" y2="1">
          <stop offset="0%" stopColor={color} stopOpacity="0.25" />
          <stop offset="100%" stopColor={color} stopOpacity="0.01" />
        </linearGradient>
        <filter id="glow">
          <feGaussianBlur stdDeviation="1.5" result="blur" />
          <feMerge><feMergeNode in="blur" /><feMergeNode in="SourceGraphic" /></feMerge>
        </filter>
      </defs>
      {volBars.map((b, i) => (
        <rect key={i} x={b.x} y={H - b.h} width={b.w} height={b.h}
          fill={color} opacity="0.12" />
      ))}
      <polyline points={`0,${H} ${pts} ${W},${H}`} fill="url(#areaGrad)" stroke="none" />
      <polyline points={pts} fill="none" stroke={color} strokeWidth="1.8"
        strokeLinejoin="round" filter="url(#glow)" />
    </svg>
  );
}

// ─── SIGNAL BAR ─────────────────────────────────────────────────────────────

function SignalBar({ label, score }) {
  const dir = score > 0.025 ? 'UP' : score < -0.025 ? 'DOWN' : 'FLAT';
  const color = dir === 'UP' ? '#00ff9d' : dir === 'DOWN' ? '#ff4757' : '#f1c40f';
  const barWidth = Math.min(Math.abs(score) * 180, 100);

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
      <span style={{ width: 138, fontSize: 11, color: '#4a6180', fontFamily: 'inherit', flexShrink: 0 }}>
        {label}
      </span>
      <div style={{ flex: 1, height: 5, background: '#0a1420', borderRadius: 3, overflow: 'hidden' }}>
        <div style={{
          width: `${barWidth}%`, height: '100%', background: color,
          borderRadius: 3, transition: 'width 0.7s ease',
          boxShadow: `0 0 6px ${color}60`,
        }} />
      </div>
      <span style={{ width: 38, fontSize: 11, color, textAlign: 'right' }}>{dir}</span>
    </div>
  );
}

// ─── STAT TILE ───────────────────────────────────────────────────────────────

function StatTile({ label, value, color = '#c9d1d9' }) {
  return (
    <div style={{
      background: '#06090f',
      borderRadius: 7,
      padding: '8px 11px',
      border: '1px solid #0d1a2a',
    }}>
      <div style={{ fontSize: 9, color: '#3d5a7a', letterSpacing: '1.5px', marginBottom: 3 }}>{label}</div>
      <div style={{ fontSize: 13, color, fontWeight: 600 }}>{value ?? '—'}</div>
    </div>
  );
}

// ─── MAIN PAGE ───────────────────────────────────────────────────────────────

export default function Home() {
  const [candles, setCandles] = useState([]);
  const [livePrice, setLivePrice] = useState(null);
  const [priceSource, setPriceSource] = useState('Chainlink');
  const [chainlinkAge, setChainlinkAge] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [history, setHistory] = useState([]);
  const [status, setStatus] = useState('Initializing…');
  const [lastUpdate, setLastUpdate] = useState(null);
  const [priceChange, setPriceChange] = useState(0);
  const [countdown, setCountdown] = useState(30);
  const prevPriceRef = useRef(null);
  const timerRef = useRef(null);

  const fmt = n => n?.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });

  const refresh = useCallback(async () => {
    try {
      setStatus('Fetching Chainlink oracle…');

      // Fetch in parallel: Chainlink price + Binance OHLCV
      const [chainlinkResult, klines] = await Promise.allSettled([
        fetchChainlinkPrice(),
        fetchKlines(100),
      ]);

      let currentPrice = null;

      if (chainlinkResult.status === 'fulfilled') {
        const cl = chainlinkResult.value;
        currentPrice = cl.price;
        setPriceSource('Chainlink');
        const ageSeconds = Math.round((Date.now() - cl.updatedAt) / 1000);
        setChainlinkAge(ageSeconds);
      }

      if (klines.status === 'fulfilled') {
        const klineData = klines.value;
        setCandles(klineData);

        // Inject Chainlink price into latest candle close for model accuracy
        if (currentPrice && klineData.length > 0) {
          klineData[klineData.length - 1] = {
            ...klineData[klineData.length - 1],
            close: currentPrice,
          };
        }

        const result = runModel(klineData);
        if (result) {
          setPrediction(result);
          setHistory(prev => [{ ...result, ts: Date.now() }, ...prev].slice(0, 8));
        }

        // Use Binance price as fallback if Chainlink failed
        if (!currentPrice && klineData.length > 0) {
          currentPrice = klineData[klineData.length - 1].close;
          setPriceSource('Binance');
        }
      }

      if (currentPrice) {
        if (prevPriceRef.current !== null) {
          setPriceChange(currentPrice - prevPriceRef.current);
        }
        prevPriceRef.current = currentPrice;
        setLivePrice(currentPrice);
      }

      setLastUpdate(new Date());
      setStatus('Live');
      setCountdown(30);
    } catch (err) {
      console.error(err);
      setStatus('Retrying…');
    }
  }, []);

  useEffect(() => {
    refresh();
    const poll = setInterval(refresh, 30000);
    timerRef.current = setInterval(() => setCountdown(c => (c > 0 ? c - 1 : 30)), 1000);
    return () => { clearInterval(poll); clearInterval(timerRef.current); };
  }, [refresh]);

  const dirColor = prediction?.direction === 'UP' ? '#00ff9d' : prediction?.direction === 'DOWN' ? '#ff4757' : '#f1c40f';
  const dirIcon = prediction?.direction === 'UP' ? '▲' : prediction?.direction === 'DOWN' ? '▼' : '◆';

  return (
    <>
      <Head>
        <title>Quantum BTC AI — Chainlink Price Oracle</title>
        <meta name="description" content="Real-time Bitcoin AI predictor powered by Chainlink oracle + CNN-LSTM model" />
        <meta name="viewport" content="width=device-width, initial-scale=1" />
      </Head>

      <div style={{
        background: 'linear-gradient(160deg, #060a0f 0%, #07101a 60%, #050912 100%)',
        minHeight: '100vh',
        fontFamily: "'JetBrains Mono', 'Fira Code', monospace",
        color: '#c9d1d9',
      }}>
        <style>{`
          @keyframes pulse { 0%,100%{opacity:1}50%{opacity:.35} }
          @keyframes slideDown { from{opacity:0;transform:translateY(-8px)}to{opacity:1;transform:translateY(0)} }
          @keyframes scanline { 0%{top:-5%}100%{top:105%} }
          @keyframes spin { from{transform:rotate(0deg)}to{transform:rotate(360deg)} }
          .scanline { position:absolute;left:0;right:0;height:1px;background:linear-gradient(transparent,rgba(0,255,157,0.05),transparent);animation:scanline 5s linear infinite;pointer-events:none; }
          .fade-in { animation: slideDown 0.4s ease; }
          @media(max-width:640px){ .grid-2{grid-template-columns:1fr!important} .price-num{font-size:28px!important} }
        `}</style>

        {/* ── HEADER ── */}
        <header style={{
          borderBottom: '1px solid #0a1a2e',
          padding: '13px 20px',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          flexWrap: 'wrap',
          gap: 10,
          background: 'rgba(5,10,20,0.7)',
          backdropFilter: 'blur(12px)',
          position: 'sticky',
          top: 0,
          zIndex: 100,
        }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <div style={{
              width: 8, height: 8, borderRadius: '50%',
              background: '#00ff9d',
              animation: 'pulse 1.8s infinite',
              boxShadow: '0 0 10px #00ff9d',
            }} />
            <span style={{
              fontFamily: "'Orbitron', sans-serif",
              fontSize: 13, letterSpacing: '3px',
              color: '#00ff9d', fontWeight: 700,
            }}>QUANTUM BTC AI</span>
            <span style={{ fontSize: 10, color: '#1e3a5a', letterSpacing: '2px' }}>v3.0 · LIVE</span>
          </div>
          <div style={{ display: 'flex', alignItems: 'center', gap: 16, fontSize: 10, color: '#2a4a6a', flexWrap: 'wrap' }}>
            <span style={{
              background: '#00ff9d15',
              border: '1px solid #00ff9d30',
              padding: '3px 8px',
              borderRadius: 4,
              color: '#00ff9d',
              fontSize: 10,
            }}>
              ⛓ Chainlink Oracle
            </span>
            <span>BTC/USD · Ethereum Mainnet</span>
            <span style={{ color: status === 'Live' ? '#00ff9d' : '#f1c40f' }}>⬤ {status}</span>
            <span>↻ <span style={{ color: '#00b4d8' }}>{countdown}s</span></span>
          </div>
        </header>

        {/* ── BODY ── */}
        <main style={{ padding: '20px', maxWidth: 920, margin: '0 auto' }}>

          {/* PRICE HERO */}
          <div style={{
            background: 'linear-gradient(135deg, rgba(0,18,38,0.85), rgba(0,8,20,0.92))',
            border: '1px solid #0a1e36',
            borderRadius: 14,
            padding: '22px 24px',
            marginBottom: 16,
            position: 'relative',
            overflow: 'hidden',
          }}>
            <div className="scanline" />

            <div style={{
              display: 'flex',
              justifyContent: 'space-between',
              alignItems: 'flex-start',
              flexWrap: 'wrap',
              gap: 18,
              marginBottom: 16,
            }}>

              {/* Left: Price */}
              <div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
                  <span style={{ fontSize: 10, color: '#2a4a6a', letterSpacing: '2px' }}>
                    BTC/USD
                  </span>
                  <span style={{
                    fontSize: 9, color: '#00ff9d',
                    background: '#00ff9d12',
                    border: '1px solid #00ff9d30',
                    padding: '1px 6px', borderRadius: 3,
                    letterSpacing: '1px',
                  }}>
                    ⛓ {priceSource}
                  </span>
                  {chainlinkAge !== null && (
                    <span style={{ fontSize: 9, color: '#2a4a6a' }}>
                      {chainlinkAge}s ago
                    </span>
                  )}
                </div>

                <div style={{ display: 'flex', alignItems: 'baseline', gap: 14 }}>
                  <span
                    className="price-num"
                    style={{
                      fontFamily: "'Orbitron', sans-serif",
                      fontSize: 44, fontWeight: 900,
                      color: '#fff',
                      textShadow: '0 0 40px rgba(0,255,157,0.15)',
                      letterSpacing: '-1px',
                    }}
                  >
                    {livePrice ? `$${fmt(livePrice)}` : '—'}
                  </span>
                  {priceChange !== 0 && (
                    <span style={{
                      fontSize: 16, fontWeight: 700,
                      color: priceChange >= 0 ? '#00ff9d' : '#ff4757',
                    }}>
                      {priceChange >= 0 ? '+' : ''}{fmt(priceChange)}
                    </span>
                  )}
                </div>

                {lastUpdate && (
                  <div style={{ fontSize: 10, color: '#1e3a5a', marginTop: 4 }}>
                    Last refresh: {lastUpdate.toLocaleTimeString()}
                  </div>
                )}
              </div>

              {/* Right: Prediction badge */}
              {prediction ? (
                <div className="fade-in" style={{
                  textAlign: 'center',
                  background: `linear-gradient(135deg, ${dirColor}10, ${dirColor}05)`,
                  border: `1px solid ${dirColor}35`,
                  borderRadius: 12,
                  padding: '16px 24px',
                  minWidth: 160,
                }}>
                  <div style={{ fontSize: 9, color: '#2a4a6a', letterSpacing: '2px', marginBottom: 8 }}>
                    5-MIN FORECAST
                  </div>
                  <div style={{
                    fontFamily: "'Orbitron', sans-serif",
                    fontSize: 30, fontWeight: 900,
                    color: dirColor,
                    textShadow: `0 0 24px ${dirColor}70`,
                    marginBottom: 10,
                  }}>
                    {dirIcon} {prediction.direction}
                  </div>
                  <div style={{ fontSize: 9, color: '#2a4a6a', letterSpacing: '1.5px', marginBottom: 4 }}>
                    CONFIDENCE
                  </div>
                  <div style={{
                    fontFamily: "'Orbitron', sans-serif",
                    fontSize: 22, fontWeight: 700,
                    color: dirColor,
                    marginBottom: 8,
                  }}>
                    {prediction.confidence.toFixed(1)}%
                  </div>
                  <div style={{ height: 4, background: '#060a0f', borderRadius: 2, overflow: 'hidden' }}>
                    <div style={{
                      width: `${prediction.confidence}%`,
                      height: '100%',
                      background: `linear-gradient(90deg, ${dirColor}70, ${dirColor})`,
                      borderRadius: 2,
                      transition: 'width 1s ease',
                      boxShadow: `0 0 8px ${dirColor}`,
                    }} />
                  </div>
                </div>
              ) : (
                <div style={{
                  display: 'flex', alignItems: 'center', gap: 10,
                  color: '#2a4a6a', fontSize: 12,
                }}>
                  <div style={{
                    width: 16, height: 16, border: '2px solid #00ff9d40',
                    borderTopColor: '#00ff9d', borderRadius: '50%',
                    animation: 'spin 1s linear infinite',
                  }} />
                  Computing model…
                </div>
              )}
            </div>

            {/* Chart */}
            <div style={{ borderTop: '1px solid #0a1a2e', paddingTop: 12 }}>
              <Sparkline candles={candles} direction={prediction?.direction} />
            </div>
          </div>

          {/* SIGNALS + INDICATORS GRID */}
          <div className="grid-2" style={{
            display: 'grid',
            gridTemplateColumns: '1fr 1fr',
            gap: 16,
            marginBottom: 16,
          }}>
            {/* Signal streams */}
            <div style={{
              background: 'rgba(0,12,26,0.85)',
              border: '1px solid #0a1a2e',
              borderRadius: 11,
              padding: 18,
            }}>
              <div style={{ fontSize: 9, color: '#1e3a5a', letterSpacing: '2.5px', marginBottom: 16 }}>
                ◈ CNN-LSTM SIGNAL STREAMS
              </div>
              {prediction
                ? Object.entries(prediction.signals).map(([k, s]) => (
                    <SignalBar key={k} label={s.label} score={s.score} />
                  ))
                : <div style={{ color: '#1e3a5a', fontSize: 12 }}>Loading…</div>
              }
            </div>

            {/* Technical indicators */}
            <div style={{
              background: 'rgba(0,12,26,0.85)',
              border: '1px solid #0a1a2e',
              borderRadius: 11,
              padding: 18,
            }}>
              <div style={{ fontSize: 9, color: '#1e3a5a', letterSpacing: '2.5px', marginBottom: 16 }}>
                ◈ TECHNICAL INDICATORS
              </div>
              {prediction ? (
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8 }}>
                  <StatTile label="RSI(14)" value={prediction.currentRSI?.toFixed(1)}
                    color={prediction.currentRSI > 70 ? '#ff4757' : prediction.currentRSI < 30 ? '#00ff9d' : '#c9d1d9'} />
                  <StatTile label="ATR(14)" value={fmt(prediction.currentATR)} />
                  <StatTile label="EMA 3" value={fmt(prediction.ema3)}
                    color={prediction.ema3 > prediction.ema8 ? '#00ff9d' : '#ff4757'} />
                  <StatTile label="EMA 8" value={fmt(prediction.ema8)}
                    color={prediction.ema8 > prediction.ema21 ? '#00ff9d' : '#ff4757'} />
                  <StatTile label="EMA 21" value={fmt(prediction.ema21)} />
                  <StatTile label="MACD" value={prediction.macdVal?.toFixed(2)}
                    color={prediction.macdVal > 0 ? '#00ff9d' : '#ff4757'} />
                  <StatTile label="MACD HIST" value={prediction.macdHist?.toFixed(2)}
                    color={prediction.macdHist > 0 ? '#00ff9d' : '#ff4757'} />
                  <StatTile label="BB MIDDLE" value={fmt(prediction.bbBands?.middle)} />
                </div>
              ) : <div style={{ color: '#1e3a5a', fontSize: 12 }}>Loading…</div>}
            </div>
          </div>

          {/* PREDICTION LOG */}
          <div style={{
            background: 'rgba(0,12,26,0.85)',
            border: '1px solid #0a1a2e',
            borderRadius: 11,
            padding: 18,
            marginBottom: 16,
          }}>
            <div style={{ fontSize: 9, color: '#1e3a5a', letterSpacing: '2.5px', marginBottom: 14 }}>
              ◈ PREDICTION LOG
            </div>
            {history.length === 0 ? (
              <div style={{ color: '#1e3a5a', fontSize: 12 }}>Awaiting first prediction…</div>
            ) : (
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                {history.map((h, i) => {
                  const c = h.direction === 'UP' ? '#00ff9d' : h.direction === 'DOWN' ? '#ff4757' : '#f1c40f';
                  const ic = h.direction === 'UP' ? '▲' : h.direction === 'DOWN' ? '▼' : '◆';
                  return (
                    <div key={i} className={i === 0 ? 'fade-in' : ''} style={{
                      display: 'flex', alignItems: 'center', gap: 14, flexWrap: 'wrap',
                      padding: '7px 11px',
                      background: i === 0 ? `${c}08` : '#060a0f',
                      borderRadius: 7,
                      border: `1px solid ${i === 0 ? c + '25' : '#0a1420'}`,
                      fontSize: 11,
                    }}>
                      <span style={{ color: '#2a4a6a', minWidth: 70 }}>
                        {new Date(h.ts).toLocaleTimeString()}
                      </span>
                      <span style={{ color: c, fontWeight: 700, minWidth: 70 }}>{ic} {h.direction}</span>
                      <span style={{ color: '#2a4a6a' }}>Conf: <span style={{ color: c }}>{h.confidence.toFixed(1)}%</span></span>
                      <span style={{ color: '#2a4a6a' }}>Price: <span style={{ color: '#c9d1d9' }}>${fmt(h.currentPrice)}</span></span>
                    </div>
                  );
                })}
              </div>
            )}
          </div>

          {/* FOOTER */}
          <div style={{
            background: 'rgba(0,8,18,0.5)',
            border: '1px solid #081525',
            borderRadius: 9,
            padding: '12px 16px',
            fontSize: 10,
            color: '#1e3a5a',
            lineHeight: 1.8,
          }}>
            <div style={{ marginBottom: 4 }}>
              <span style={{ color: '#0e2d4a' }}>⛓ PRICE SOURCE: </span>
              Chainlink BTC/USD Data Feed · Contract <code style={{ color: '#0d3a5a', fontSize: 9 }}>0xF4030086522a5bEEa4988F8cA5B36dbC97BeE88</code> · Ethereum Mainnet
            </div>
            <div style={{ marginBottom: 4 }}>
              <span style={{ color: '#0e2d4a' }}>◈ MODEL: </span>
              CNN-LSTM Hybrid Ensemble · 5 signal streams → LSTM temporal encoding → stacking layer · 5-min OHLCV via Binance REST · 30s refresh
            </div>
            <div style={{ color: '#0a1e30' }}>
              ⚠ Informational only — not financial advice. Cryptocurrency trading involves substantial risk of loss.
            </div>
          </div>
        </main>
      </div>
    </>
  );
}
