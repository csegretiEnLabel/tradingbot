import React, { useState } from 'react';
import { Search, TrendingUp, TrendingDown, Target, AlertCircle } from 'lucide-react';

const API_BASE = "http://localhost:8000/api";

export default function StockAnalyzer() {
    const [symbol, setSymbol] = useState('');
    const [daysBack, setDaysBack] = useState(30);
    const [analysis, setAnalysis] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const analyzeStock = async () => {
        if (!symbol.trim()) return;

        setLoading(true);
        setError(null);

        try {
            const res = await fetch(`${API_BASE}/quant/analyze`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    symbol: symbol.toUpperCase(),
                    days_back: daysBack
                })
            });

            const data = await res.json();
            if (data.error) {
                setError(data.error);
            } else {
                setAnalysis(data);
            }
        } catch (err) {
            setError('Failed to analyze stock. Check if API is running.');
        } finally {
            setLoading(false);
        }
    };

    const handleKeyPress = (e) => {
        if (e.key === 'Enter') {
            analyzeStock();
        }
    };

    return (
        <div className="panel" style={{ marginBottom: '1.5rem' }}>
            <div className="panel-header">
                <h2 className="panel-title">
                    <Search size={20} className="primary" /> Individual Stock Analyzer
                </h2>
            </div>

            <div style={{ padding: '1rem' }}>
                <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1rem' }}>
                    <input
                        type="text"
                        placeholder="Enter symbol (e.g., AAPL)"
                        value={symbol}
                        onChange={(e) => setSymbol(e.target.value.toUpperCase())}
                        onKeyPress={handleKeyPress}
                        style={{
                            flex: 1,
                            padding: '0.5rem',
                            borderRadius: '6px',
                            border: '1px solid var(--card-border)',
                            background: 'var(--bg)',
                            color: 'var(--text)',
                            fontSize: '0.875rem'
                        }}
                    />
                    <input
                        type="number"
                        placeholder="Days"
                        value={daysBack}
                        onChange={(e) => setDaysBack(parseInt(e.target.value) || 30)}
                        style={{
                            width: '80px',
                            padding: '0.5rem',
                            borderRadius: '6px',
                            border: '1px solid var(--card-border)',
                            background: 'var(--bg)',
                            color: 'var(--text)',
                            fontSize: '0.875rem'
                        }}
                    />
                    <button
                        onClick={analyzeStock}
                        disabled={loading || !symbol.trim()}
                        style={{
                            padding: '0.5rem 1rem',
                            borderRadius: '6px',
                            border: 'none',
                            background: 'var(--primary)',
                            color: 'white',
                            fontSize: '0.875rem',
                            cursor: loading ? 'wait' : 'pointer',
                            opacity: loading || !symbol.trim() ? 0.5 : 1
                        }}
                    >
                        {loading ? 'Analyzing...' : 'Analyze'}
                    </button>
                </div>

                {error && (
                    <div style={{
                        padding: '0.75rem',
                        background: 'rgba(239, 68, 68, 0.1)',
                        border: '1px solid var(--danger)',
                        borderRadius: '6px',
                        color: 'var(--danger)',
                        fontSize: '0.875rem',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem'
                    }}>
                        <AlertCircle size={16} />
                        {error}
                    </div>
                )}

                {analysis && (
                    <div style={{ marginTop: '1rem' }}>
                        <div style={{
                            padding: '1rem',
                            background: 'rgba(99, 102, 241, 0.05)',
                            border: '1px solid var(--card-border)',
                            borderRadius: '8px',
                            marginBottom: '1rem'
                        }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.75rem' }}>
                                <div>
                                    <span className="symbol-tag" style={{ fontSize: '1.125rem' }}>{analysis.symbol}</span>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
                                        {new Date(analysis.timestamp).toLocaleString()}
                                    </div>
                                </div>
                                <div style={{ textAlign: 'right' }}>
                                    <div style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                                        ${analysis.current_price}
                                    </div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        ATR: ${analysis.atr_14?.toFixed(2)}
                                    </div>
                                </div>
                            </div>

                            <div style={{
                                display: 'grid',
                                gridTemplateColumns: 'repeat(3, 1fr)',
                                gap: '0.75rem',
                                marginTop: '0.75rem'
                            }}>
                                <div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Entry</div>
                                    <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--success)' }}>
                                        ${analysis.suggested_entry}
                                    </div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Stop Loss</div>
                                    <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--danger)' }}>
                                        ${analysis.stop_loss}
                                    </div>
                                </div>
                                <div>
                                    <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Take Profit</div>
                                    <div style={{ fontSize: '0.875rem', fontWeight: 600, color: 'var(--success)' }}>
                                        ${analysis.take_profit}
                                    </div>
                                </div>
                            </div>
                        </div>

                        <div style={{ marginBottom: '1rem' }}>
                            <h3 style={{ fontSize: '0.875rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--primary)' }}>
                                Consensus: {analysis.consensus?.action?.toUpperCase() || 'HOLD'}
                                {analysis.consensus?.agreement !== undefined && (
                                    <span style={{ fontSize: '0.75rem', opacity: 0.7, marginLeft: '0.5rem' }}>
                                        ({(analysis.consensus.agreement * 100).toFixed(0)}% agreement)
                                    </span>
                                )}
                            </h3>
                        </div>

                        <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem' }}>
                            {Object.entries(analysis.strategies).map(([name, data]) => (
                                <div key={name} style={{
                                    padding: '0.75rem',
                                    background: 'var(--card-bg)',
                                    border: '1px solid var(--card-border)',
                                    borderRadius: '6px'
                                }}>
                                    <div style={{
                                        display: 'flex',
                                        justifyContent: 'space-between',
                                        alignItems: 'center',
                                        marginBottom: '0.5rem'
                                    }}>
                                        <div style={{ fontWeight: 600, fontSize: '0.8125rem' }}>
                                            {name.toUpperCase().replace('_', ' ')}
                                        </div>
                                        <div style={{
                                            display: 'flex',
                                            alignItems: 'center',
                                            gap: '0.25rem',
                                            color: data.action === 'buy' ? 'var(--success)' : data.action === 'sell' ? 'var(--danger)' : 'var(--text-muted)'
                                        }}>
                                            {data.action === 'buy' && <TrendingUp size={14} />}
                                            {data.action === 'sell' && <TrendingDown size={14} />}
                                            <span style={{ fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase' }}>
                                                {data.action}
                                            </span>
                                            <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>
                                                {(data.strength * 100).toFixed(0)}%
                                            </span>
                                        </div>
                                    </div>
                                    <div style={{ fontSize: '0.8125rem', color: 'var(--text-muted)' }}>
                                        {data.reasoning}
                                    </div>
                                </div>
                            ))}
                        </div>
                    </div>
                )}
            </div>
        </div>
    );
}
