import React, { useState, useEffect } from 'react';
import { TrendingUp, TrendingDown, Target, Zap, Wind, BarChart3 } from 'lucide-react';

const API_BASE = "http://localhost:8000/api";

export default function QuantSignalsPanel() {
    const [signals, setSignals] = useState(null);
    const [loading, setLoading] = useState(true);
    const [selectedSymbol, setSelectedSymbol] = useState(null);

    useEffect(() => {
        const fetchSignals = async () => {
            try {
                const res = await fetch(`${API_BASE}/quant/signals`);
                const data = await res.json();
                setSignals(data);
                setLoading(false);
            } catch (err) {
                console.error("Failed to fetch quant signals:", err);
            }
        };

        fetchSignals();
        const interval = setInterval(fetchSignals, 5000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading quant signals...</div>;
    if (!signals || signals.error) return <div style={{ padding: '2rem' }}>No quant strategies enabled</div>;

    const symbols = Object.keys(signals).sort();
    const strategies = ['kama', 'trend_follow', 'momentum'];

    const getSignalBadge = (signal) => {
        if (!signal) return <span style={{ color: 'var(--text-muted)' }}>—</span>;

        const colors = {
            buy: 'var(--success)',
            sell: 'var(--danger)',
            hold: 'var(--text-muted)'
        };

        const icons = {
            buy: <TrendingUp size={14} />,
            sell: <TrendingDown size={14} />,
            hold: <span>—</span>
        };

        return (
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '0.25rem',
                color: colors[signal.action]
            }}>
                {icons[signal.action]}
                <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>
                    {signal.action}
                </span>
                <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                    {(signal.strength * 100).toFixed(0)}%
                </span>
            </div>
        );
    };

    const getConsensus = (symbolSignals) => {
        const validSignals = Object.values(symbolSignals).filter(s => s);
        if (validSignals.length === 0) return null;

        const buys = validSignals.filter(s => s.action === 'buy').length;
        const sells = validSignals.filter(s => s.action === 'sell').length;
        const total = validSignals.length;

        if (buys / total >= 0.66) return { action: 'buy', agreement: buys / total };
        if (sells / total >= 0.66) return { action: 'sell', agreement: sells / total };
        return { action: 'hold', agreement: 0 };
    };

    return (
        <div className="panel" style={{ marginBottom: '1.5rem' }}>
            <div className="panel-header">
                <h2 className="panel-title">
                    <Zap size={20} className="primary" /> Quant Strategy Signals
                </h2>
                <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                    Live signals from KAMA, Trend Follow, and Momentum
                </span>
            </div>

            <div style={{ overflowX: 'auto' }}>
                <table>
                    <thead>
                        <tr>
                            <th>Symbol</th>
                            <th><Target size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} />KAMA</th>
                            <th><Wind size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} />Trend</th>
                            <th><BarChart3 size={14} style={{ marginRight: '4px', verticalAlign: 'middle' }} />Momentum</th>
                            <th>Consensus</th>
                        </tr>
                    </thead>
                    <tbody>
                        {symbols.map(symbol => {
                            const symbolSignals = signals[symbol];
                            const consensus = getConsensus(symbolSignals);

                            return (
                                <tr key={symbol}
                                    onClick={() => setSelectedSymbol(selectedSymbol === symbol ? null : symbol)}
                                    style={{ cursor: 'pointer' }}>
                                    <td>
                                        <span className="symbol-tag">{symbol}</span>
                                    </td>
                                    <td>{getSignalBadge(symbolSignals.kama)}</td>
                                    <td>{getSignalBadge(symbolSignals.trend_follow)}</td>
                                    <td>{getSignalBadge(symbolSignals.momentum)}</td>
                                    <td>
                                        {consensus && consensus.action !== 'hold' ? (
                                            <div style={{
                                                display: 'flex',
                                                alignItems: 'center',
                                                gap: '0.25rem',
                                                color: consensus.action === 'buy' ? 'var(--success)' : 'var(--danger)',
                                                fontWeight: 600
                                            }}>
                                                {consensus.action === 'buy' ? <TrendingUp size={14} /> : <TrendingDown size={14} />}
                                                {consensus.action.toUpperCase()}
                                                <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                                                    ({(consensus.agreement * 100).toFixed(0)}%)
                                                </span>
                                            </div>
                                        ) : (
                                            <span style={{ color: 'var(--text-muted)' }}>No consensus</span>
                                        )}
                                    </td>
                                </tr>
                            );
                        })}
                    </tbody>
                </table>
            </div>

            {selectedSymbol && signals[selectedSymbol] && (
                <div style={{
                    marginTop: '1rem',
                    padding: '1rem',
                    background: 'rgba(99, 102, 241, 0.05)',
                    border: '1px solid var(--card-border)',
                    borderRadius: '8px'
                }}>
                    <h3 style={{ marginBottom: '0.75rem', color: 'var(--primary)' }}>{selectedSymbol} Signal Details</h3>
                    {strategies.map(strategy => {
                        const signal = signals[selectedSymbol][strategy];
                        if (!signal) return null;

                        return (
                            <div key={strategy} style={{ marginBottom: '0.75rem' }}>
                                <div style={{ fontWeight: 600, fontSize: '0.875rem', marginBottom: '0.25rem' }}>
                                    {strategy.toUpperCase().replace('_', ' ')}
                                </div>
                                <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                    {signal.reasoning}
                                </div>
                                <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                                    Metrics: {Object.entries(signal.metrics).map(([k, v]) =>
                                        `${k}=${typeof v === 'number' ? v.toFixed(3) : v}`
                                    ).join(', ')}
                                </div>
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
