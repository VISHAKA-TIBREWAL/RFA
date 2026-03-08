import './App.css'
import { useEffect, useMemo, useState } from 'react'

type RiskRegime = 'Stable' | 'Early Stress' | 'Crisis' | 'Recovery'

type RiskStatePoint = {
  date: string
  close: number
  short_vol: number
  long_vol: number
  drawdown: number
  volume_stress: number
  instability: number
  recovery_signal: number
  state: RiskRegime
}

type SimulationResponse = {
  paths: string[][]
  states: string[]
}

type HistoryStats = {
  shortVols: number[]
  drawdowns: number[]
  volumeStress: number[]
}

const API_BASE = 'http://127.0.0.1:8000'

function useApi<T>(url: string) {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<string | null>(null)

  const reload = () => {
    setLoading(true)
    setError(null)
    fetch(url)
      .then(async (res) => {
        if (!res.ok) {
          throw new Error(`${res.status} ${res.statusText}`)
        }
        return res.json()
      })
      .then((json) => setData(json as T))
      .catch((err) => setError(err.message))
      .finally(() => setLoading(false))
  }

  useEffect(() => {
    reload()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [url])

  return { data, loading, error, reload }
}

function badgeColor(state: RiskRegime) {
  switch (state) {
    case 'Stable':
      return '#16a34a'
    case 'Early Stress':
      return '#f97316'
    case 'Crisis':
      return '#dc2626'
    case 'Recovery':
      return '#0ea5e9'
    default:
      return '#6b7280'
  }
}

function percentileRank(value: number, sortedValues: number[]): number {
  if (sortedValues.length === 0) return 0.5
  let idx = sortedValues.findIndex((v) => value <= v)
  if (idx === -1) idx = sortedValues.length - 1
  return (idx + 1) / sortedValues.length
}

function levelFromPercentile(p: number, [low, mid, high]: [string, string, string]): string {
  if (p < 0.33) return low
  if (p < 0.66) return mid
  return high
}

function describeVolatilityLevel(current: number, stats: HistoryStats | null): string {
  if (!stats) return ''
  const p = percentileRank(current, stats.shortVols)
  return levelFromPercentile(p, ['Low', 'Medium', 'High'])
}

function describeDrawdownLevel(current: number, stats: HistoryStats | null): string {
  if (!stats) return ''
  const p = percentileRank(current, stats.drawdowns)
  // More negative drawdowns are "deeper", so invert the interpretation
  if (p < 0.33) return 'Deep'
  if (p < 0.66) return 'Moderate'
  return 'Shallow'
}

function describeVolumeStressLevel(current: number, stats: HistoryStats | null): string {
  if (!stats) return ''
  const p = percentileRank(current, stats.volumeStress)
  return levelFromPercentile(p, ['Calm', 'Elevated', 'Unusually high'])
}

function App() {
  const {
    data: health,
    loading: healthLoading,
    error: healthError,
    reload: reloadHealth,
  } = useApi<{ status: string; time: string }>(`${API_BASE}/health`)

  const {
    data: latest,
    loading: latestLoading,
    error: latestError,
    reload: reloadLatest,
  } = useApi<RiskStatePoint>(`${API_BASE}/risk-state/latest`)

  const {
    data: history,
    loading: historyLoading,
    error: historyError,
    reload: reloadHistory,
  } = useApi<RiskStatePoint[]>(`${API_BASE}/risk-state/history?limit=365`)

  const [simRequest, setSimRequest] = useState({
    start_state: 'Stable' as RiskRegime,
    n_steps: 60,
    n_paths: 10,
    random_seed: 42,
  })
  const [simResult, setSimResult] = useState<SimulationResponse | null>(null)
  const [simLoading, setSimLoading] = useState(false)
  const [simError, setSimError] = useState<string | null>(null)

  const handleSimulate = async () => {
    setSimLoading(true)
    setSimError(null)
    try {
      const res = await fetch(`${API_BASE}/risk-state/simulate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json', Accept: 'application/json' },
        body: JSON.stringify(simRequest),
      })
      if (!res.ok) {
        const text = await res.text()
        throw new Error(text || `${res.status} ${res.statusText}`)
      }
      const json = (await res.json()) as SimulationResponse
      setSimResult(json)
    } catch (err: any) {
      setSimError(err.message ?? String(err))
      setSimResult(null)
    } finally {
      setSimLoading(false)
    }
  }

  const historyWithStateIndex = useMemo(() => {
    if (!history) return []
    const regimeOrder: RiskRegime[] = ['Stable', 'Early Stress', 'Crisis', 'Recovery']
    const regimeToIndex = Object.fromEntries(regimeOrder.map((r, i) => [r, i]))
    return history.map((h) => ({
      ...h,
      stateIndex: regimeToIndex[h.state],
    }))
  }, [history])

  const historyStats: HistoryStats | null = useMemo(() => {
    if (!history || history.length === 0) return null
    const shortVols = history.map((h) => h.short_vol).slice().sort((a, b) => a - b)
    const drawdowns = history.map((h) => h.drawdown).slice().sort((a, b) => a - b)
    const volumeStress = history.map((h) => h.volume_stress).slice().sort((a, b) => a - b)
    return { shortVols, drawdowns, volumeStress }
  }, [history])

  return (
    <div className="app-root">
      <header className="app-header">
        <div>
          <h1>NIFTY Bank Market Mood</h1>
          <p>
            A simple view of how risky the NIFTY Bank index looks right now, how it has behaved
            recently, and what could happen next.
          </p>
        </div>
        <div className="health-indicator">
          {healthLoading && <span className="pill pill-neutral">Checking health…</span>}
          {healthError && (
            <button className="pill pill-error" onClick={reloadHealth}>
              Health error — retry
            </button>
          )}
          {health && !healthError && (
            <span className="pill pill-ok">
              API {health.status} • {new Date(health.time).toLocaleTimeString()}
            </span>
          )}
        </div>
      </header>

      <main className="layout">
        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>Today&apos;s Market Mood</h2>
              <p className="muted">
                Where the market stands right now in simple terms.
              </p>
            </div>
            <button onClick={reloadLatest}>Refresh</button>
          </div>
          {latestLoading && <p className="muted">Loading latest risk state…</p>}
          {latestError && <p className="error-text">{latestError}</p>}
          {latest && !latestError && (
            <div className="latest-grid">
              <div className="latest-main">
                <div className="state-badge" style={{ borderColor: badgeColor(latest.state) }}>
                  <span
                    className="state-dot"
                    style={{ backgroundColor: badgeColor(latest.state) }}
                  />
                  <span className="state-label">{latest.state}</span>
                </div>
                <p className="muted">
                  As of {new Date(latest.date).toLocaleString()} • Close{' '}
                  <strong>{latest.close.toFixed(2)}</strong>
                </p>
              </div>
              <div className="metrics-grid">
                <div>
                  <h3>Volatility</h3>
                  <p>Short-term: {latest.short_vol.toFixed(4)}</p>
                  <p>Long-term: {latest.long_vol.toFixed(4)}</p>
                  <p className="muted small">
                    Overall: <strong>{describeVolatilityLevel(latest.short_vol, historyStats)}</strong>{' '}
                    compared to the past year.
                  </p>
                </div>
                <div>
                  <h3>Recent Losses & Volume</h3>
                  <p>From recent peak: {(latest.drawdown * 100).toFixed(2)}%</p>
                  <p>Trading activity score: {latest.volume_stress.toFixed(2)}</p>
                  <p className="muted small">
                    Losses: <strong>{describeDrawdownLevel(latest.drawdown, historyStats)}</strong>{' '}
                    • Activity:{' '}
                    <strong>{describeVolumeStressLevel(latest.volume_stress, historyStats)}</strong>
                  </p>
                </div>
                <div>
                  <h3>Choppiness & Recovery Signs</h3>
                  <p>Instability: {latest.instability.toFixed(4)}</p>
                  <p>Recovery signal: {latest.recovery_signal.toFixed(4)}</p>
                  <p className="muted small">
                    Positive recovery signal suggests the market may be starting to bounce back.
                  </p>
                </div>
              </div>
            </div>
          )}
        </section>

        <section className="panel">
          <div className="panel-header">
            <div>
              <h2>History (last ~year)</h2>
              <p className="muted">
                Each dot is a day; color shows the market mood (green = calm, red = high risk).
              </p>
            </div>
            <button onClick={reloadHistory}>Refresh</button>
          </div>
          {historyLoading && <p className="muted">Loading history…</p>}
          {historyError && <p className="error-text">{historyError}</p>}
          {historyWithStateIndex.length > 0 && (
            <div className="history-chart">
              <svg width="100%" height="160">
                {historyWithStateIndex.map((point, idx) => {
                  const x = (idx / (historyWithStateIndex.length - 1 || 1)) * 100
                  const y = 120 - point.stateIndex * 35
                  return (
                    <circle
                      key={point.date}
                      cx={`${x}%`}
                      cy={y}
                      r={3}
                      fill={badgeColor(point.state)}
                    />
                  )
                })}
              </svg>
              <div className="legend">
                {(['Stable', 'Early Stress', 'Crisis', 'Recovery'] as RiskRegime[]).map((r) => (
                  <span key={r} className="legend-item">
                    <span
                      className="legend-dot"
                      style={{ backgroundColor: badgeColor(r) }}
                    />
                    {r}
                  </span>
                ))}
              </div>
            </div>
          )}
        </section>

        <section className="panel panel-full">
          <div className="panel-header">
            <div>
              <h2>Possible Future Scenarios</h2>
              <p className="muted">
                We simulate many paths the market mood could follow, based on how it behaved in the
                past.
              </p>
            </div>
          </div>
          <div className="sim-controls">
            <label>
              Start state
              <select
                value={simRequest.start_state}
                onChange={(e) =>
                  setSimRequest((s) => ({
                    ...s,
                    start_state: e.target.value as RiskRegime,
                  }))
                }
              >
                <option value="Stable">Stable</option>
                <option value="Early Stress">Early Stress</option>
                <option value="Crisis">Crisis</option>
                <option value="Recovery">Recovery</option>
              </select>
            </label>
            <label>
              Steps
              <input
                type="number"
                min={10}
                max={365}
                value={simRequest.n_steps}
                onChange={(e) =>
                  setSimRequest((s) => ({
                    ...s,
                    n_steps: Number(e.target.value),
                  }))
                }
              />
            </label>
            <label>
              Paths
              <input
                type="number"
                min={1}
                max={50}
                value={simRequest.n_paths}
                onChange={(e) =>
                  setSimRequest((s) => ({
                    ...s,
                    n_paths: Number(e.target.value),
                  }))
                }
              />
            </label>
            <label>
              Random seed
              <input
                type="number"
                value={simRequest.random_seed}
                onChange={(e) =>
                  setSimRequest((s) => ({
                    ...s,
                    random_seed: Number(e.target.value),
                  }))
                }
              />
            </label>
            <button onClick={handleSimulate} disabled={simLoading}>
              {simLoading ? 'Simulating…' : 'Run simulation'}
            </button>
          </div>
          {simError && <p className="error-text">{simError}</p>}
          {simResult && (
            <div className="sim-matrix">
              <div className="sim-scroll">
                <table>
                  <thead>
                    <tr>
                      <th>Path \\ Step</th>
                      {simResult.paths[0]?.map((_, idx) => (
                        <th key={idx}>{idx}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {simResult.paths.map((row, i) => (
                      <tr key={i}>
                        <td>{i + 1}</td>
                        {row.map((state, j) => (
                          <td
                            key={j}
                            style={{
                              backgroundColor: badgeColor(state as RiskRegime),
                              color: '#0f172a',
                              fontSize: '0.75rem',
                            }}
                          >
                            {state[0]}
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <p className="muted">
                Cells are colored by regime; letter is the first character of the state
                (S, E, C, R).
              </p>
            </div>
          )}
        </section>
      </main>

      <footer className="app-footer">
        <span>Backend: {API_BASE}</span>
      </footer>
    </div>
  )
}

export default App
