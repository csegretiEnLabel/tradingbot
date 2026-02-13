import React, { useState, useEffect } from 'react';
import { AlertCircle, CheckCircle, XCircle, Filter } from 'lucide-react';

const API_BASE = "http://localhost:8000/api";

export default function InterventionLog() {
    const [interventions, setInterventions] = useState([]);
    const [stats, setStats] = useState(null);
    const [filter, setFilter] = useState('all'); // all, rejected, approved
    const [loading, setLoading] = useState(true);

    useEffect(() => {
        const fetchData = async () => {
            try {
                const [interventionsRes, statsRes] = await Promise.all([
                    fetch(`${API_BASE}/quant/interventions?limit=20`),
                    fetch(`${API_BASE}/quant/interventions/stats?days=7`)
                ]);

                const interventionsData = await interventionsRes.json();
                const statsData = await statsRes.json();

                setInterventions(interventionsData);
                setStats(statsData);
                setLoading(false);
            } catch (err) {
                console.error("Failed to fetch interventions:", err);
            }
        };

        fetchData();
        const interval = setInterval(fetchData, 10000);
        return () => clearInterval(interval);
    }, []);

    const filteredInterventions = interventions.filter(i => {
        if (filter === 'rejected') return i.action === 'REJECTED';
        if (filter === 'approved') return i.action === 'APPROVED';
        return true;
    });

    const getIcon = (action) => {
        if (action === 'APPROVED') return <CheckCircle size={16} style={{ color: 'var(--success)' }} />;
        if (action === 'REJECTED') return <XCircle size={16} style={{ color: 'var(--danger)' }} />;
        return <AlertCircle size={16} style={{ color: 'var(--warning)' }} />;
    };

    const getColor = (action) => {
        if (action === 'APPROVED') return 'var(--success)';
        if (action === 'REJECTED') return 'var(--danger)';
        return 'var(--warning)';
    };

    if (loading) return <div style={{ padding: '2rem', textAlign: 'center' }}>Loading interventions...</div>;

    return (
        <div className="panel" style={{ marginBottom: '1.5rem' }}>
            <div className="panel-header">
                <h2 className="panel-title">
                    <AlertCircle size={20} className="primary" /> AI/Risk Interventions
                </h2>
                <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <Filter size={16} style={{ color: 'var(--text-muted)' }} />
                    <select
                        value={filter}
                        onChange={(e) => setFilter(e.target.value)}
                        style={{
                            padding: '0.25rem 0.5rem',
                            borderRadius: '4px',
                            border: '1px solid var(--card-border)',
                            background: 'var(--card-bg)',
                            color: 'var(--text)',
                            fontSize: '0.75rem'
                        }}
                    >
                        <option value="all">All ({interventions.length})</option>
                        <option value="rejected">Rejected ({interventions.filter(i => i.action === 'REJECTED').length})</option>
                        <option value="approved">Approved ({interventions.filter(i => i.action === 'APPROVED').length})</option>
                    </select>
                </div>
            </div>

            {stats && (
                <div style={{
                    padding: '1rem',
                    background: 'rgba(99, 102, 241, 0.05)',
                    border: '1px solid var(--card-border)',
                    borderRadius: '8px',
                    margin: '1rem 1rem 0',
                    display: 'grid',
                    gridTemplateColumns: 'repeat(auto-fit, minmax(150px, 1fr))',
                    gap: '1rem'
                }}>
                    <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>7-Day Summary</div>
                        <div style={{ fontSize: '1.125rem', fontWeight: 600 }}>{stats.total_interventions}</div>
                    </div>
                    <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Rejection Rate</div>
                        <div style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--danger)' }}>
                            {(stats.rejection_rate * 100).toFixed(0)}%
                        </div>
                    </div>
                    <div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>Approval Rate</div>
                        <div style={{ fontSize: '1.125rem', fontWeight: 600, color: 'var(--success)' }}>
                            {((1 - stats.rejection_rate) * 100).toFixed(0)}%
                        </div>
                    </div>
                </div>
            )}

            <div className="scroll-area" style={{ maxHeight: '400px', marginTop: '1rem' }}>
                {filteredInterventions.length > 0 ? (
                    <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', padding: '0 1rem 1rem' }}>
                        {filteredInterventions.map((intervention, idx) => (
                            <div key={idx} style={{
                                padding: '0.75rem',
                                background: 'var(--card-bg)',
                                border: '1px solid var(--card-border)',
                                borderRadius: '6px'
                            }}>
                                <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: '0.5rem' }}>
                                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        {getIcon(intervention.action)}
                                        <span className="symbol-tag">{intervention.symbol}</span>
                                        <span style={{
                                            fontSize: '0.75rem',
                                            fontWeight: 600,
                                            color: getColor(intervention.action)
                                        }}>
                                            {intervention.action}
                                        </span>
                                    </div>
                                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                                        {new Date(intervention.timestamp).toLocaleTimeString()}
                                    </span>
                                </div>

                                <div style={{ fontSize: '0.8125rem', color: 'var(--text)', marginBottom: '0.5rem' }}>
                                    {intervention.reasoning}
                                </div>

                                <div style={{
                                    display: 'flex',
                                    gap: '1rem',
                                    fontSize: '0.75rem',
                                    color: 'var(--text-muted)'
                                }}>
                                    <span>Intervener: <strong>{intervention.intervener}</strong></span>
                                    <span>Strategy: <strong>{intervention.strategy}</strong></span>
                                    <span>{intervention.original_action} → {intervention.final_action}</span>
                                </div>
                            </div>
                        ))}
                    </div>
                ) : (
                    <div style={{ textAlign: 'center', padding: '2rem', color: 'var(--text-muted)' }}>
                        No interventions found for selected filter
                    </div>
                )}
            </div>

            {stats && stats.top_reasons && stats.top_reasons.length > 0 && (
                <div style={{ padding: '1rem', borderTop: '1px solid var(--card-border)' }}>
                    <div style={{ fontSize: '0.8125rem', fontWeight: 600, marginBottom: '0.5rem', color: 'var(--primary)' }}>
                        Top Rejection Reasons (7 days):
                    </div>
                    {stats.top_reasons.slice(0, 3).map((reason, idx) => (
                        <div key={idx} style={{ fontSize: '0.75rem', color: 'var(--text-muted)', marginBottom: '0.25rem' }}>
                            {idx + 1}. {reason.reason} <span style={{ color: 'var(--danger)' }}>({reason.count}×)</span>
                        </div>
                    ))}
                </div>
            )}
        </div>
    );
}
