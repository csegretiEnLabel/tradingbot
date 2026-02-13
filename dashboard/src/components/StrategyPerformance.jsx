import React, { useState, useEffect } from 'react';
import { BarChart3, TrendingUp } from 'lucide-react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';

const API_BASE = "http://localhost:8000/api";

export default function StrategyPerformance() {
    const [performance, setPerformance] = useState(null);
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchPerformance = async () => {
            try {
                const res = await fetch(`${API_BASE}/quant/performance`);
                const data = await res.json();
                setPerformance(data);
                setLoading(false);
            } catch (err) {
                console.error("Failed to fetch strategy performance:", err);
            }
        };

        fetchPerformance();
        const interval = setInterval(fetchPerformance, 15000);
        return () => clearInterval(interval);
    }, []);

    if (loading) return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading performance...</div>;
    if (!performance) return null;

    // Transform data for chart
    const chartData = Object.entries(performance).map(([strategy, stats]) => ({
        name: strategy.toUpperCase().replace('_', ' '),
        approvalRate: stats.approval_rate * 100,
        total: stats.total_signals
    }));

    return (
        <div className="panel" style={{ marginBottom: '1.5rem' }}>
            <div className="panel-header">
                <h2 className="panel-title">
                    <BarChart3 size={20} className="primary" /> Strategy Performance
                </h2>
                <span style={{ fontSize: '0.875rem', color: 'var(--text-muted)' }}>
                    AI approval rates by strategy
                </span>
            </div>

            <div style={{ padding: '1rem' }}>
                <div style={{ height: '200px', marginBottom: '1rem' }}>
                    <ResponsiveContainer width="100%" height="100%">
                        <BarChart data={chartData}>
                            <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                            <XAxis
                                dataKey="name"
                                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                            />
                            <YAxis
                                tick={{ fill: 'var(--text-muted)', fontSize: 12 }}
                                domain={[0, 100]}
                                label={{ value: 'Approval %', angle: -90, position: 'insideLeft', fill: 'var(--text-muted)', fontSize: 12 }}
                            />
                            <Tooltip
                                contentStyle={{
                                    background: 'var(--card-bg)',
                                    border: '1px solid var(--card-border)',
                                    borderRadius: '8px',
                                    fontSize: '0.875rem'
                                }}
                                formatter={(value) => [`${value.toFixed(1)}%`, 'Approval Rate']}
                            />
                            <Bar dataKey="approvalRate" radius={[4, 4, 0, 0]}>
                                {chartData.map((entry, index) => (
                                    <Cell key={`cell-${index}`} fill="var(--primary)" />
                                ))}
                            </Bar>
                        </BarChart>
                    </ResponsiveContainer>
                </div>

                <div style={{
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
                    gap: '0.75rem'
                }}>
                    {Object.entries(performance).map(([strategy, stats]) => (
                        <div key={strategy} style={{
                            padding: '0.75rem',
                            background: 'rgba(99, 102, 241, 0.05)',
                            border: '1px solid var(--card-border)',
                            borderRadius: '6px'
                        }}>
                            <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '0.5rem' }}>
                                {strategy.toUpperCase().replace('_', ' ')}
                            </div>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                marginBottom: '0.25rem'
                            }}>
                                <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Approval Rate</span>
                                <span style={{
                                    fontSize: '0.875rem',
                                    fontWeight: 600,
                                    color: stats.approval_rate >= 0.5 ? 'var(--success)' : 'var(--danger)'
                                }}>
                                    {(stats.approval_rate * 100).toFixed(1)}%
                                </span>
                            </div>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)'
                            }}>
                                <span>Total Signals</span>
                                <span>{stats.total_signals}</span>
                            </div>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)'
                            }}>
                                <span>Approved</span>
                                <span style={{ color: 'var(--success)' }}>{stats.approved}</span>
                            </div>
                            <div style={{
                                display: 'flex',
                                justifyContent: 'space-between',
                                fontSize: '0.75rem',
                                color: 'var(--text-muted)'
                            }}>
                                <span>Rejected</span>
                                <span style={{ color: 'var(--danger)' }}>{stats.rejected}</span>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
}
