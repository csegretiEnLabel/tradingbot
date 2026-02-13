import React, { useState, useEffect } from 'react';
import {
  TrendingUp,
  TrendingDown,
  Activity,
  DollarSign,
  PieChart,
  Terminal,
  ShieldCheck,
  AlertTriangle,
  Zap,
  Search,
  BarChart3,
  AlertCircle
} from 'lucide-react';
import {
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';

// Import quant components
import QuantSignalsPanel from './components/QuantSignalsPanel';
import StockAnalyzer from './components/StockAnalyzer';
import InterventionLog from './components/InterventionLog';
import StrategyPerformance from './components/StrategyPerformance';

const API_BASE = "http://localhost:8000/api";

function App() {
  const [data, setData] = useState(null);
  const [logs, setLogs] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [activeTab, setActiveTab] = useState('overview'); // overview, quant, analyzer, performance

  const fetchData = async () => {
    try {
      const [statusRes, logsRes] = await Promise.all([
        fetch(`${API_BASE}/status`),
        fetch(`${API_BASE}/logs?limit=50`)
      ]);

      const statusData = await statusRes.json();
      const logsData = await logsRes.json();

      setData(statusData);
      setLogs(logsData.logs || []);
      setLoading(false);
      setError(null);
    } catch (err) {
      console.error("Fetch error:", err);
      setError("Failed to connect to Bot API (is it running?)");
    }
  };

  useEffect(() => {
    fetchData();
    const interval = setInterval(fetchData, 2000);
    return () => clearInterval(interval);
  }, []);

  const { account, positions, costs, bot_status } = data || {};

  // Calculate time until next run
  const [timeLeft, setTimeLeft] = useState("");

  useEffect(() => {
    if (!bot_status?.next_run_time) return;

    const timer = setInterval(() => {
      const now = new Date().getTime();
      const next = new Date(bot_status.next_run_time).getTime();
      const diff = next - now;

      if (diff <= 0) {
        setTimeLeft("Running...");
      } else {
        const minutes = Math.floor((diff % (1000 * 60 * 60)) / (1000 * 60));
        const seconds = Math.floor((diff % (1000 * 60)) / 1000);
        setTimeLeft(`${minutes}m ${seconds}s`);
      }
    }, 1000);

    return () => clearInterval(timer);
  }, [bot_status?.next_run_time]);

  if (loading && !error) {
    return (
      <div className="dashboard-container" style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '80vh' }}>
        <div style={{ textAlign: 'center' }}>
          <div className="status-dot" style={{ margin: '0 auto 1rem', width: '20px', height: '20px' }}></div>
          <p style={{ color: 'var(--text-muted)' }}>Initializing Command Center...</p>
        </div>
      </div>
    );
  }

  const tabs = [
    { id: 'overview', label: 'Overview', icon: <PieChart size={16} /> },
    { id: 'quant', label: 'Quant Signals', icon: <Zap size={16} /> },
    { id: 'analyzer', label: 'Stock Analyzer', icon: <Search size={16} /> },
    { id: 'performance', label: 'Performance', icon: <BarChart3 size={16} /> },
  ];

  return (
    <div className="dashboard-container">
      <header className="header">
        <div>
          <h1 className="bot-title">BOT COMMAND CENTER</h1>
          <p style={{ color: 'var(--text-muted)', fontSize: '0.875rem' }}>AI Agent Brain v3.0 | Paper Trading Mode</p>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}>
          <select
            className="strategy-select"
            value={data?.settings?.strategy || "preservation"}
            onChange={async (e) => {
              const newMode = e.target.value;
              try {
                await fetch(`${API_BASE}/settings`, {
                  method: 'POST',
                  headers: { 'Content-Type': 'application/json' },
                  body: JSON.stringify({ strategy: newMode })
                });
                fetchData();
              } catch (err) {
                console.error("Failed to update strategy", err);
              }
            }}
            style={{
              padding: '0.5rem',
              borderRadius: '6px',
              border: '1px solid var(--card-border)',
              background: 'var(--card-bg)',
              color: 'var(--text)',
              fontSize: '0.875rem',
              cursor: 'pointer'
            }}
          >
            <option value="preservation">🛡️ Preservation (Default)</option>
            <option value="aggressive">🦁 Aggressive (Apex)</option>
          </select>

          <div className="status-badge">
            <div className={`status-dot ${timeLeft === "Running..." ? "blink" : ""}`}></div>
            <span>
              {timeLeft === "Running..." ? "RUNNING NOW" : `Next Run: ${timeLeft || "Computing..."}`}
            </span>
          </div>
        </div>
      </header>

      {error && (
        <div className="panel" style={{ marginBottom: '1.5rem', borderColor: 'var(--danger)', background: 'rgba(239, 68, 68, 0.05)' }}>
          <div style={{ color: 'var(--danger)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
            <AlertTriangle size={18} />
            <span>{error}</span>
          </div>
        </div>
      )}

      <div className="stats-grid">
        <StatCard
          label="Total Equity"
          value={`$${account?.equity?.toLocaleString() || '0'}`}
          icon={<DollarSign size={18} />}
          change={account?.daily_pnl_pct != null ? `${account.daily_pnl_pct >= 0 ? '+' : ''}${(account.daily_pnl_pct * 100).toFixed(2)}%` : 'N/A'}
          isUp={account?.daily_pnl_pct >= 0}
        />
        <StatCard
          label="Daily P&L"
          value={`$${account?.daily_pnl?.toFixed(2) || '0.00'}`}
          icon={<Activity size={18} />}
          isUp={account?.daily_pnl >= 0}
        />
        <StatCard
          label="API Cost (Today)"
          value={`$${costs?.today_api_cost?.toFixed(4) || '0.0000'}`}
          icon={<ShieldCheck size={18} />}
          subtext={costs?.today_budget_remaining != null ? `Budget: $${costs.today_budget_remaining.toFixed(2)} left` : 'N/A'}
        />
        <StatCard
          label="Self-Sustaining"
          value={costs?.self_sustaining ? "YES" : "NO"}
          icon={<PieChart size={18} />}
          subtext={costs?.total_net != null ? `Net: $${costs.total_net.toFixed(2)}` : 'N/A'}
          isUp={costs?.self_sustaining}
        />
      </div>

      {/* Tab Navigation */}
      <div style={{
        display: 'flex',
        gap: '0.5rem',
        marginBottom: '1.5rem',
        borderBottom: '1px solid var(--card-border)',
        padding: '0 0.5rem'
      }}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            style={{
              padding: '0.75rem 1rem',
              background: activeTab === tab.id ? 'var(--card-bg)' : 'transparent',
              border: 'none',
              borderBottom: activeTab === tab.id ? '2px solid var(--primary)' : '2px solid transparent',
              color: activeTab === tab.id ? 'var(--primary)' : 'var(--text-muted)',
              cursor: 'pointer',
              display: 'flex',
              alignItems: 'center',
              gap: '0.5rem',
              fontSize: '0.875rem',
              fontWeight: activeTab === tab.id ? 600 : 400,
              transition: 'all 0.2s'
            }}
          >
            {tab.icon}
            {tab.label}
          </button>
        ))}
      </div>

      {/* Tab Content */}
      {activeTab === 'overview' && (
        <div className="main-content">
          <div className="left-column">
            <div className="panel" style={{ marginBottom: '1.5rem' }}>
              <div className="panel-header">
                <h2 className="panel-title"><PieChart size={20} className="primary" /> Open Positions</h2>
                <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>{positions?.length} active trades</span>
              </div>
              <div className="scroll-area">
                {positions?.length > 0 ? (
                  <table>
                    <thead>
                      <tr>
                        <th>Symbol</th>
                        <th>Qty</th>
                        <th>Entry</th>
                        <th>Now</th>
                        <th>P&L %</th>
                        <th>Value</th>
                        <th>Est. Tax</th>
                      </tr>
                    </thead>
                    <tbody>
                      {positions.map(p => {
                        const tax = p.unrealized_pl > 0 ? p.unrealized_pl * 0.35 : 0;
                        return (
                          <tr key={p.symbol}>
                            <td><span className="symbol-tag">{p.symbol}</span></td>
                            <td>{p.qty.toFixed(4)}</td>
                            <td>${p.avg_entry_price.toFixed(2)}</td>
                            <td>${p.current_price.toFixed(2)}</td>
                            <td className={p.unrealized_plpc >= 0 ? 'up' : 'down'}>
                              {(p.unrealized_plpc * 100).toFixed(2)}%
                            </td>
                            <td>${p.market_value.toFixed(2)}</td>
                            <td style={{ color: 'var(--text-muted)' }}>
                              ${tax.toFixed(2)}
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                ) : (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    No open positions. Cash is a position.
                  </div>
                )}
              </div>
            </div>

            <div className="panel">
              <div className="panel-header">
                <h2 className="panel-title"><TrendingUp size={20} /> Equity History</h2>
              </div>
              <div style={{ height: '240px', width: '100%' }}>
                <ResponsiveContainer width="100%" height="100%">
                  <AreaChart data={[]}>
                    <defs>
                      <linearGradient id="colorValue" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="5%" stopColor="var(--primary)" stopOpacity={0.3} />
                        <stop offset="95%" stopColor="var(--primary)" stopOpacity={0} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                    <XAxis dataKey="time" hide />
                    <YAxis hide domain={['auto', 'auto']} />
                    <Tooltip
                      contentStyle={{ background: 'var(--card-bg)', border: '1px solid var(--card-border)', borderRadius: '8px' }}
                    />
                    <Area type="monotone" dataKey="value" stroke="var(--primary)" fillOpacity={1} fill="url(#colorValue)" />
                  </AreaChart>
                </ResponsiveContainer>
                <div style={{ textAlign: 'center', color: 'var(--text-muted)', fontSize: '0.875rem', marginTop: '-100px' }}>
                  History visualization will populate with more data points...
                </div>
              </div>
            </div>
          </div>

          <div className="right-column">
            <div className="panel" style={{ height: '100%', display: 'flex', flexDirection: 'column' }}>
              <div className="panel-header">
                <h2 className="panel-title"><Terminal size={20} /> Live Logs</h2>
              </div>
              <div className="scroll-area" style={{ flex: 1, maxHeight: 'none' }}>
                {logs.length > 0 ? logs.map((log, i) => {
                  const parts = log.split(' ');
                  const time = parts[1]?.split(',')[0] || '00:00:00';
                  const levelMatch = log.match(/\[(INFO|WARNING|ERROR|CRITICAL)\]/);
                  const level = levelMatch ? levelMatch[1] : 'INFO';
                  const message = log.split('] ').pop();

                  return (
                    <div key={i} className="log-item">
                      <span className="log-time">{time}</span>
                      <span className={`log-level ${level}`}>{level}</span>
                      <span className="log-msg">{message}</span>
                    </div>
                  );
                }) : (
                  <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                    Waiting for bot activity...
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>
      )}

      {activeTab === 'quant' && (
        <div>
          <QuantSignalsPanel />
          <InterventionLog />
        </div>
      )}

      {activeTab === 'analyzer' && (
        <StockAnalyzer />
      )}

      {activeTab === 'performance' && (
        <StrategyPerformance />
      )}
    </div>
  );
}

function StatCard({ label, value, icon, change, isUp, subtext }) {
  return (
    <div className="stat-card">
      <div className="stat-label">
        <span style={{ color: 'var(--primary)' }}>{icon}</span>
        {label}
      </div>
      <div className="stat-value">{value}</div>
      {change && (
        <div className={`stat-change ${isUp ? 'up' : 'down'}`}>
          {isUp ? <TrendingUp size={12} style={{ marginRight: 4 }} /> : <TrendingDown size={12} style={{ marginRight: 4 }} />}
          {change} vs yesterday
        </div>
      )}
      {subtext && (
        <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)', marginTop: '0.5rem' }}>
          {subtext}
        </div>
      )}
    </div>
  );
}

export default App;
