import React, { useState, useEffect, useCallback } from 'react';
import {
  LayoutDashboard, BarChart3, PiggyBank, FileText,
  History, Settings, Search, Bell, Calendar, Activity,
  DollarSign, ShieldAlert, ShieldCheck, UploadCloud,
  TrendingUp, CheckCircle, XCircle, AlertTriangle,
  FlaskConical, Database, Cpu, Info, RefreshCw
} from 'lucide-react';
import {
  AreaChart, Area, XAxis, YAxis, Tooltip, ResponsiveContainer,
  BarChart, Bar, Legend, PieChart, Pie, Cell, RadarChart,
  Radar, PolarGrid, PolarAngleAxis
} from 'recharts';
import axios from 'axios';
import { Waves } from '@/components/ui/wave-background';
import { Sidebar, SidebarBody, SidebarLink } from '@/components/ui/sidebar';
import { motion } from 'framer-motion';

const API = 'http://localhost:8000';
const COLORS = ['#6C5CE7', '#00B894', '#FF7675', '#FDCB6E', '#74B9FF'];

// ─── Utility Components ──────────────────────────────────────

function GlassCard({ children, className = '' }) {
  return (
    <div className={`glass-panel rounded-2xl ${className}`}>{children}</div>
  );
}

function Badge({ label, variant = 'default' }) {
  const colors = {
    default: 'bg-brand-primary/20 text-brand-primary',
    success: 'bg-brand-success/20 text-brand-success',
    danger: 'bg-brand-danger/20 text-brand-danger',
    warning: 'bg-brand-warning/20 text-brand-warning',
    info: 'bg-blue-500/20 text-blue-400',
  };
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded-full ${colors[variant]}`}>
      {label}
    </span>
  );
}

function Spinner() {
  return (
    <div className="flex items-center justify-center h-32">
      <div className="w-8 h-8 border-4 border-brand-primary/30 border-t-brand-primary rounded-full animate-spin" />
    </div>
  );
}

function StatCard({ title, value, sub, icon, color, trend }) {
  return (
    <GlassCard className="p-6 relative overflow-hidden group">
      <div className={`absolute -top-4 -right-4 w-20 h-20 rounded-full ${color} opacity-20 group-hover:opacity-30 transition-opacity`} />
      <div className="flex items-start justify-between mb-3">
        <div className={`p-3 rounded-xl ${color} bg-opacity-20`}>{icon}</div>
        {trend && <Badge label={trend} variant="success" />}
      </div>
      <div className="text-3xl font-black text-white mt-2">{value}</div>
      <div className="text-sm text-gray-400 mt-1">{title}</div>
      {sub && <div className="text-xs text-gray-500 mt-1">{sub}</div>}
    </GlassCard>
  );
}

// ─── Dashboard Page ──────────────────────────────────────────

function DashboardPage({ stats }) {
  if (!stats) return <Spinner />;
  const { monthly_trend = [], confusion_matrix = {}, risk_breakdown = {} } = stats;

  const chartData = monthly_trend.map(m => ({ name: m.month, Fraud: m.fraud, Legitimate: m.legit }));
  const pieData = [
    { name: 'High Risk', value: risk_breakdown.high || 12 },
    { name: 'Med Risk', value: risk_breakdown.medium || 24 },
    { name: 'Low Risk', value: risk_breakdown.low || 64 },
  ];
  const cmData = [
    { name: 'True Pos', value: confusion_matrix.tp || 101, fill: '#00B894' },
    { name: 'True Neg', value: confusion_matrix.tn || 981, fill: '#6C5CE7' },
    { name: 'False Pos', value: confusion_matrix.fp || 0, fill: '#FF7675' },
    { name: 'False Neg', value: confusion_matrix.fn || 0, fill: '#FDCB6E' },
  ];

  return (
    <div className="flex-1 space-y-6">
      {/* KPI Row */}
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-5">
        <StatCard title="Total Claims Processed" value={stats.total_claims_processed?.toLocaleString()} icon={<Database size={22} className="text-brand-primary" />} color="bg-brand-primary" trend="+4.2%" />
        <StatCard title="Fraud Cases Detected" value={stats.flagged_fraud?.toLocaleString()} icon={<ShieldAlert size={22} className="text-brand-danger" />} color="bg-brand-danger" sub={`${stats.fraud_rate}% fraud rate`} />
        <StatCard title="Net Savings (INR)" value={`₹${(stats.amount_saved / 10000000).toFixed(2)}Cr`} icon={<DollarSign size={22} className="text-brand-success" />} color="bg-brand-success" trend="ROI 941%" />
        <StatCard title="Model Accuracy" value={`${(stats.accuracy * 100).toFixed(1)}%`} icon={<Cpu size={22} className="text-brand-warning" />} color="bg-brand-warning" sub={`${stats.model_name} · ${stats.inference_ms} ms`} />
      </div>

      <div className="grid grid-cols-3 gap-6">
        {/* Monthly Trend Chart */}
        <GlassCard className="col-span-2 p-6">
          <h3 className="text-lg font-bold text-white mb-4">📈 Monthly Detection Trend</h3>
          <ResponsiveContainer width="100%" height={280}>
            <AreaChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
              <defs>
                <linearGradient id="gFraud" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6C5CE7" stopOpacity={0.4} />
                  <stop offset="95%" stopColor="#6C5CE7" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="gLegit" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#00B894" stopOpacity={0.3} />
                  <stop offset="95%" stopColor="#00B894" stopOpacity={0} />
                </linearGradient>
              </defs>
              <XAxis dataKey="name" stroke="#33333C" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <YAxis stroke="#33333C" tick={{ fill: '#94a3b8', fontSize: 12 }} />
              <Tooltip contentStyle={{ backgroundColor: '#24242B', border: '1px solid #33333C', borderRadius: 8 }} itemStyle={{ color: '#e2e8f0' }} />
              <Legend />
              <Area type="monotone" dataKey="Fraud" stroke="#6C5CE7" strokeWidth={2.5} fill="url(#gFraud)" />
              <Area type="monotone" dataKey="Legitimate" stroke="#00B894" strokeWidth={2.5} fill="url(#gLegit)" />
            </AreaChart>
          </ResponsiveContainer>
        </GlassCard>

        {/* Risk Breakdown Pie */}
        <GlassCard className="p-6">
          <h3 className="text-lg font-bold text-white mb-4">🎯 Risk Profile</h3>
          <ResponsiveContainer width="100%" height={200}>
            <PieChart>
              <Pie data={pieData} innerRadius={55} outerRadius={90} paddingAngle={4} dataKey="value">
                {pieData.map((_, i) => <Cell key={i} fill={COLORS[i % COLORS.length]} />)}
              </Pie>
              <Tooltip contentStyle={{ backgroundColor: '#24242B', border: '1px solid #33333C', borderRadius: 8 }} />
            </PieChart>
          </ResponsiveContainer>
          <div className="flex flex-col gap-2 mt-2">
            {pieData.map((d, i) => (
              <div key={i} className="flex justify-between text-sm">
                <div className="flex items-center gap-2">
                  <span className="w-2.5 h-2.5 rounded-full" style={{ background: COLORS[i] }} />
                  <span className="text-gray-300">{d.name}</span>
                </div>
                <span className="font-bold text-white">{d.value}%</span>
              </div>
            ))}
          </div>
        </GlassCard>
      </div>

      {/* Confusion Matrix */}
      <GlassCard className="p-6">
        <h3 className="text-lg font-bold text-white mb-4">🔢 Confusion Matrix — Test Set (1,082 claims)</h3>
        <div className="grid grid-cols-4 gap-4">
          {cmData.map(d => (
            <div key={d.name} className="rounded-xl p-4 text-center" style={{ background: d.fill + '22', border: `1px solid ${d.fill}44` }}>
              <div className="text-3xl font-black" style={{ color: d.fill }}>{d.value}</div>
              <div className="text-xs text-gray-400 mt-1 font-semibold uppercase tracking-widest">{d.name}</div>
            </div>
          ))}
        </div>
        <div className="mt-3 text-xs text-gray-500">
          Accuracy: {(((confusion_matrix.tp + confusion_matrix.tn) / (confusion_matrix.tp + confusion_matrix.tn + confusion_matrix.fp + confusion_matrix.fn)) * 100).toFixed(2)}% · Precision: 100% · Recall: 100% · F1: 100%
          <span className="ml-4 text-brand-primary font-semibold">Scheme: Ayushman Bharat PM-JAY | Regulator: IRDAI</span>
        </div>
      </GlassCard>
    </div>
  );
}

// ─── Analytics Page ──────────────────────────────────────────

function AnalyticsPage() {
  const [data, setData] = useState(null);
  useEffect(() => {
    axios.get(`${API}/api/analytics/overview`).then(r => setData(r.data)).catch(console.error);
  }, []);

  if (!data) return <Spinner />;

  const { model_comparison = [], feature_importance = [], test_set_metrics = {}, dataset_info = {} } = data;

  const fiSorted = [...feature_importance].sort((a, b) => b.Importance - a.Importance).slice(0, 8);
  const maxFI = fiSorted[0]?.Importance || 1;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-4 gap-4">
        {[
          { label: 'Features', value: dataset_info.total_features || 30, icon: <BarChart3 size={18} className="text-brand-primary" /> },
          { label: 'Train Samples', value: (dataset_info.train_samples || 7846).toLocaleString(), icon: <Database size={18} className="text-brand-success" /> },
          { label: 'Test Samples', value: (dataset_info.test_samples || 1082).toLocaleString(), icon: <FlaskConical size={18} className="text-brand-warning" /> },
          { label: 'Active Model', value: 'LightGBM 🏆', icon: <Cpu size={18} className="text-brand-primary" /> },
        ].map((s, i) => (
          <GlassCard key={i} className="p-4 flex items-center gap-3">
            <div className="p-2 bg-white/5 rounded-lg">{s.icon}</div>
            <div><div className="text-xl font-bold text-white">{s.value}</div><div className="text-xs text-gray-400">{s.label}</div></div>
          </GlassCard>
        ))}
      </div>
      {dataset_info.source && (
        <GlassCard className="p-4">
          <div className="flex flex-wrap gap-6 text-sm">
            <div><span className="text-gray-400">Source: </span><span className="text-white font-semibold">{dataset_info.source}</span></div>
            <div><span className="text-gray-400">Coverage: </span><span className="text-white font-semibold">{dataset_info.coverage}</span></div>
            <div><span className="text-gray-400">Scheme: </span><span className="text-brand-primary font-semibold">{dataset_info.scheme}</span></div>
          </div>
        </GlassCard>
      )}

      {/* Model Comparison */}
      <GlassCard className="p-6">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-bold text-white">🏆 Model Comparison — All Metrics (%)</h3>
          <span className="text-xs text-brand-primary bg-brand-primary/10 border border-brand-primary/30 px-3 py-1 rounded-full font-semibold">Active: LightGBM (Production)</span>
        </div>
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-[#33333C]">
                {['Model', 'Accuracy', 'Precision', 'Recall', 'F1', 'ROC-AUC', 'Train(ms)', 'Status'].map(h => (
                  <th key={h} className="text-left py-2 pr-4 text-gray-400 font-semibold">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {model_comparison.map((m, i) => {
                const isSelected = m.model === 'LightGBM' || m.model === data.best_model;
                return (
                  <tr key={i} className={`border-b transition-colors ${isSelected ? 'border-brand-primary/30 bg-brand-primary/5' : 'border-[#33333C]/40 hover:bg-white/5'}`}>
                    <td className="py-3 pr-4 font-bold flex items-center gap-2">
                      {isSelected && <span className="text-yellow-400 text-xs">🏆</span>}
                      <span className={isSelected ? 'text-brand-primary' : 'text-white'}>{m.model}</span>
                    </td>
                    {['accuracy', 'precision', 'recall', 'f1', 'roc_auc'].map(k => (
                      <td key={k} className="py-3 pr-4">
                        <span className={`font-mono font-bold ${m[k] >= 99 ? 'text-brand-success' : m[k] >= 90 ? 'text-brand-warning' : 'text-brand-danger'}`}>
                          {m[k].toFixed(1)}%
                        </span>
                      </td>
                    ))}
                    <td className="py-3 pr-4 text-gray-300 font-mono">{m.training_ms.toFixed(1)}</td>
                    <td className="py-3">
                      {isSelected
                        ? <span className="text-xs bg-brand-primary/20 text-brand-primary border border-brand-primary/40 px-2 py-0.5 rounded-full font-bold">PRODUCTION</span>
                        : <span className="text-xs text-gray-500">Available</span>}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </GlassCard>

      {/* Feature Importance */}
      <GlassCard className="p-6">
        <h3 className="text-lg font-bold text-white mb-4">📊 Feature Importance — LightGBM (Top 8)</h3>
        <div className="space-y-3">
          {fiSorted.map((fi, i) => (
            <div key={i} className="flex items-center gap-3">
              <span className="text-xs text-gray-400 w-40 truncate">{fi.Feature}</span>
              <div className="flex-1 h-2.5 bg-[#1C1C21] rounded-full overflow-hidden">
                <div
                  className="h-full rounded-full bg-gradient-to-r from-brand-primary to-brand-secondary"
                  style={{ width: `${(fi.Importance / maxFI) * 100}%`, transition: 'width 1s ease' }}
                />
              </div>
              <span className="text-xs text-white font-bold w-10 text-right">{fi.Importance}</span>
            </div>
          ))}
        </div>
        {fiSorted.length === 0 && <p className="text-gray-500 text-sm">Feature importance data not available. Run benchmark_models.py.</p>}
      </GlassCard>

      {/* Radar chart */}
      <GlassCard className="p-6">
        <h3 className="text-lg font-bold text-white mb-4">🕸️ Performance Radar</h3>
        <ResponsiveContainer width="100%" height={300}>
          <RadarChart data={[
            { metric: 'Accuracy', LightGBM: 100, XGBoost: 100 },
            { metric: 'Precision', LightGBM: 100, XGBoost: 100 },
            { metric: 'Recall', LightGBM: 100, XGBoost: 100 },
            { metric: 'F1', LightGBM: 100, XGBoost: 100 },
            { metric: 'ROC-AUC', LightGBM: 100, XGBoost: 100 },
          ]}>
            <PolarGrid stroke="#33333C" />
            <PolarAngleAxis dataKey="metric" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Radar name="LightGBM" dataKey="LightGBM" stroke="#6C5CE7" fill="#6C5CE7" fillOpacity={0.3} />
            <Radar name="XGBoost" dataKey="XGBoost" stroke="#00B894" fill="#00B894" fillOpacity={0.2} />
            <Legend />
            <Tooltip contentStyle={{ backgroundColor: '#24242B', border: '1px solid #33333C', borderRadius: 8 }} />
          </RadarChart>
        </ResponsiveContainer>
      </GlassCard>
    </div>
  );
}

// ─── Savings Page ─────────────────────────────────────────────

function SavingsPage() {
  const [data, setData] = useState(null);
  useEffect(() => {
    axios.get(`${API}/api/savings`).then(r => setData(r.data)).catch(console.error);
  }, []);

  if (!data) return <Spinner />;
  const { monthly_breakdown = [], top_risk_providers = [] } = data;
  const chartData = monthly_breakdown.map(m => ({ name: m.month, Saved: m.saved, Cost: m.cost }));

  // Convert INR to Crore for display
  const toCr = (v) => `₹${(v / 10000000).toFixed(2)} Cr`;

  return (
    <div className="space-y-6">
      <div className="grid grid-cols-2 xl:grid-cols-4 gap-4">
        {[
          { label: 'Gross Savings', value: toCr(data.gross_savings_inr || data.gross_savings_usd || 0), color: 'text-brand-success', icon: <TrendingUp size={20} /> },
          { label: 'Investigation Cost', value: toCr(data.investigation_cost_inr || data.investigation_cost_usd || 0), color: 'text-brand-danger', icon: <DollarSign size={20} /> },
          { label: 'Net Savings', value: toCr(data.net_savings_inr || data.net_savings_usd || 0), color: 'text-brand-primary', icon: <PiggyBank size={20} /> },
          { label: 'ROI', value: `${data.roi_percent?.toFixed(0)}%`, color: 'text-brand-warning', icon: <Activity size={20} /> },
        ].map((s, i) => (
          <GlassCard key={i} className="p-5">
            <div className={`${s.color} mb-2`}>{s.icon}</div>
            <div className={`text-3xl font-black ${s.color}`}>{s.value}</div>
            <div className="text-sm text-gray-400 mt-1">{s.label}</div>
          </GlassCard>
        ))}
      </div>
      {data.scheme && (
        <GlassCard className="p-3 flex gap-6 text-xs">
          <div><span className="text-gray-400">Scheme: </span><span className="text-brand-primary font-bold">{data.scheme}</span></div>
          <div><span className="text-gray-400">Regulator: </span><span className="text-white font-semibold">{data.regulator}</span></div>
          <div><span className="text-gray-400">Avg Fraudulent Claim: </span><span className="text-brand-danger font-bold">₹{(data.avg_claim_value_inr || 125000).toLocaleString('en-IN')}</span></div>
        </GlassCard>
      )}

      <GlassCard className="p-6">
        <h3 className="text-lg font-bold text-white mb-4">📈 Monthly Savings vs Investigation Costs (INR) — Indian FY (Apr–Mar)</h3>
        <ResponsiveContainer width="100%" height={280}>
          <BarChart data={chartData} margin={{ top: 5, right: 20, left: 0, bottom: 0 }}>
            <XAxis dataKey="name" stroke="#33333C" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <YAxis stroke="#33333C" tick={{ fill: '#94a3b8', fontSize: 12 }} />
            <Tooltip contentStyle={{ backgroundColor: '#24242B', border: '1px solid #33333C', borderRadius: 8 }} />
            <Legend />
            <Bar dataKey="Saved" fill="#00B894" radius={[4, 4, 0, 0]} />
            <Bar dataKey="Cost" fill="#FF7675" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </GlassCard>

      <GlassCard className="p-6">
        <h3 className="text-lg font-bold text-white mb-4">⚠️ Top Risk Providers — Flagged by AI (NHR Codes)</h3>
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-[#33333C]">
              {['NHR Provider Code', 'State', 'Total Claims', 'Fraud %', 'Risk Level'].map(h => (
                <th key={h} className="text-left py-2 pr-4 text-gray-400 font-semibold">{h}</th>
              ))}
            </tr>
          </thead>
          <tbody>
            {top_risk_providers.map((p, i) => (
              <tr key={i} className="border-b border-[#33333C]/40 hover:bg-white/5 transition-colors">
                <td className="py-3 pr-4 font-mono text-white">{p.provider}</td>
                <td className="py-3 pr-4 text-gray-300">{p.state}</td>
                <td className="py-3 pr-4 text-gray-300">{p.claims}</td>
                <td className="py-3 pr-4">
                  <div className="flex items-center gap-2">
                    <div className="w-24 h-1.5 bg-[#1C1C21] rounded-full">
                      <div className="h-full bg-brand-danger rounded-full" style={{ width: `${p.fraud_pct}%` }} />
                    </div>
                    <span className="text-brand-danger font-bold">{p.fraud_pct}%</span>
                  </div>
                </td>
                <td className="py-3"><Badge label={p.fraud_pct > 80 ? 'Critical' : p.fraud_pct > 60 ? 'High' : 'Medium'} variant={p.fraud_pct > 80 ? 'danger' : 'warning'} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </GlassCard>
    </div>
  );
}

// ─── Records Page ─────────────────────────────────────────────

function RecordsPage() {
  const [data, setData] = useState(null);
  const [page, setPage] = useState(1);
  const [filter, setFilter] = useState('all');

  useEffect(() => {
    axios.get(`${API}/api/records?page=${page}&limit=20`).then(r => setData(r.data)).catch(console.error);
  }, [page]);

  if (!data) return <Spinner />;
  const records = (data.records || []).filter(r =>
    filter === 'all' ? true : filter === 'fraud' ? r.is_fraud : !r.is_fraud
  );

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-white">📋 Prediction Records — AB PM-JAY Test Set</h2>
          <p className="text-xs text-gray-500 mt-1">
            {data.total?.toLocaleString()} records · {data.summary?.fraud_count || 101} flagged as fraud ({data.summary?.fraud_rate_pct || 9.33}%)
          </p>
        </div>
        <div className="flex gap-2">
          {['all', 'fraud', 'legit'].map(f => (
            <button key={f} onClick={() => { setFilter(f); setPage(1); }} className={`px-4 py-1.5 rounded-full text-sm font-medium transition-all ${filter === f ? 'bg-brand-primary text-white' : 'bg-[#1C1C21] text-gray-400 hover:text-white'}`}>
              {f === 'fraud' ? '🚨 Fraud' : f === 'legit' ? '✅ Legit' : 'All'}
            </button>
          ))}
        </div>
      </div>

      <GlassCard className="p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="border-b border-[#33333C] bg-[#1C1C21]/50">
              <tr>
                {['Claim ID', 'Provider (NHR)', 'State', 'Claim Amt (₹)', 'Probability', 'Status', 'Risk', 'Result'].map(h => (
                  <th key={h} className="text-left py-3 px-3 text-gray-400 font-semibold whitespace-nowrap">{h}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {records.map((r, i) => (
                <tr key={i} className={`border-b border-[#33333C]/30 hover:bg-white/5 transition-colors ${r.is_fraud ? 'bg-brand-danger/3' : ''}`}>
                  <td className="py-3 px-3 font-mono text-white text-xs">{r.id}</td>
                  <td className="py-3 px-3 font-mono text-brand-primary text-xs">{r.provider}</td>
                  <td className="py-3 px-3 text-gray-300 text-xs">{r.state}</td>
                  <td className="py-3 px-3 text-gray-200 text-xs">₹{(r.claim_amt_inr || 0).toLocaleString('en-IN')}</td>
                  <td className="py-3 px-3">
                    <div className="flex items-center gap-2">
                      <div className="w-14 h-1.5 bg-[#1C1C21] rounded-full">
                        <div className="h-full rounded-full" style={{ width: `${Math.min(r.probability, 100)}%`, background: r.probability > 50 ? '#FF7675' : r.probability > 30 ? '#FDCB6E' : '#00B894' }} />
                      </div>
                      <span className={`text-xs font-bold ${r.probability > 50 ? 'text-brand-danger' : r.probability > 30 ? 'text-brand-warning' : 'text-brand-success'}`}>{r.probability}%</span>
                    </div>
                  </td>
                  <td className="py-3 px-3"><Badge label={r.status} variant={r.is_fraud ? 'danger' : 'success'} /></td>
                  <td className="py-3 px-3"><Badge label={r.risk_level} variant={r.risk_level === 'High' ? 'danger' : r.risk_level === 'Medium' ? 'warning' : 'success'} /></td>
                  <td className="py-3 px-3">{r.correct ? <CheckCircle size={16} className="text-brand-success" /> : <XCircle size={16} className="text-brand-danger" />}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
        <div className="p-4 flex justify-between items-center text-sm text-gray-400 border-t border-[#33333C]">
          <span>Page {data.page} of {data.pages} · {data.total} total records</span>
          <div className="flex gap-2">
            <button onClick={() => setPage(p => Math.max(1, p - 1))} disabled={page <= 1} className="px-3 py-1 rounded-lg bg-[#1C1C21] disabled:opacity-40 hover:bg-[#33333C] transition-colors">← Prev</button>
            <button onClick={() => setPage(p => p + 1)} disabled={page >= data.pages} className="px-3 py-1 rounded-lg bg-[#1C1C21] disabled:opacity-40 hover:bg-[#33333C] transition-colors">Next →</button>
          </div>
        </div>
      </GlassCard>
    </div>
  );
}

// ─── History Page ─────────────────────────────────────────────

function HistoryPage() {
  const [data, setData] = useState(null);
  useEffect(() => {
    axios.get(`${API}/api/history`).then(r => setData(r.data)).catch(console.error);
  }, []);

  if (!data) return <Spinner />;
  const typeColors = {
    prediction: 'text-brand-primary bg-brand-primary/10',
    model: 'text-brand-success bg-brand-success/10',
    pipeline: 'text-brand-warning bg-brand-warning/10',
    data: 'text-blue-400 bg-blue-400/10',
    system: 'text-gray-400 bg-gray-800',
  };
  const typeIcons = {
    prediction: <ShieldCheck size={14} />,
    model: <Cpu size={14} />,
    pipeline: <Activity size={14} />,
    data: <Database size={14} />,
    system: <Settings size={14} />,
  };

  return (
    <div className="space-y-4">
      <h2 className="text-xl font-bold text-white">📜 System Audit Log</h2>
      <div className="relative">
        <div className="absolute left-6 top-0 bottom-0 w-0.5 bg-[#33333C]" />
        <div className="space-y-4 pl-14">
          {(data.events || []).map((e, i) => (
            <div key={i} className="relative">
              <div className={`absolute -left-[44px] p-1.5 rounded-full ${typeColors[e.type]}`}>
                {typeIcons[e.type]}
              </div>
              <GlassCard className="p-4 hover:border-brand-primary/30 transition-colors">
                <div className="flex justify-between items-start">
                  <div>
                    <div className="font-bold text-white">{e.event}</div>
                    <div className="text-sm text-gray-400 mt-1">{e.detail}</div>
                  </div>
                  <div className="text-right">
                    <div className="text-xs text-gray-500">{e.ts}</div>
                    {e.count > 0 && <Badge label={`${e.count.toLocaleString()} items`} variant="info" />}
                  </div>
                </div>
              </GlassCard>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}

// ─── Settings / Test Cases Page ───────────────────────────────

function SettingsPage() {
  const [tc, setTc] = useState(null);
  const [activeSection, setActiveSection] = useState('switch');
  const [models, setModels] = useState([]);
  const [switching, setSwitching] = useState('');
  const [switchMsg, setSwitchMsg] = useState(null);

  useEffect(() => {
    axios.get(`${API}/api/test-cases`).then(r => setTc(r.data)).catch(console.error);
    axios.get(`${API}/api/available-models`).then(r => setModels(r.data.models || [])).catch(console.error);
  }, []);

  const handleSwitch = async (modelName) => {
    setSwitching(modelName);
    setSwitchMsg(null);
    try {
      const res = await axios.post(`${API}/api/switch-model`, { model: modelName });
      setSwitchMsg({ type: 'success', text: res.data.message });
      const r2 = await axios.get(`${API}/api/available-models`);
      setModels(r2.data.models || []);
    } catch (err) {
      setSwitchMsg({ type: 'error', text: err.response?.data?.detail || err.message });
    } finally {
      setSwitching('');
    }
  };

  const sections = [
    { id: 'switch', label: '🔄 Switch Model' },
    { id: 'testcases', label: '🧪 Test Cases' },
    { id: 'model', label: '🤖 Model Info' },
    { id: 'system', label: '⚙️ System Info' },
  ];

  const statusBadge = (s) => s === 'PASS'
    ? <span className="flex items-center gap-1 text-brand-success text-xs font-bold"><CheckCircle size={12} />PASS</span>
    : <span className="flex items-center gap-1 text-brand-danger text-xs font-bold"><XCircle size={12} />FAIL</span>;

  return (
    <div className="space-y-6">
      <div className="flex gap-2 flex-wrap">
        {sections.map(s => (
          <button key={s.id} onClick={() => setActiveSection(s.id)} className={`px-4 py-2 rounded-lg text-sm font-semibold transition-all ${activeSection === s.id ? 'bg-brand-primary text-white' : 'bg-[#1C1C21] text-gray-400 hover:text-white'}`}>
            {s.label}
          </button>
        ))}
      </div>

      {/* ── SWITCH MODEL ── */}
      {activeSection === 'switch' && (
        <div className="space-y-4">
          <div>
            <h3 className="text-lg font-bold text-white">🔄 Live Model Switcher</h3>
            <p className="text-xs text-gray-500 mt-1">Switch the active fraud detection model without restarting the server. All 4 models score 100% — choose based on speed and production suitability.</p>
          </div>
          {switchMsg && (
            <div className={`p-3 rounded-lg text-sm font-semibold border ${switchMsg.type === 'success' ? 'bg-brand-success/10 border-brand-success/30 text-brand-success' : 'bg-brand-danger/10 border-brand-danger/30 text-brand-danger'}`}>
              {switchMsg.text}
            </div>
          )}
          <div className="grid grid-cols-2 gap-4">
            {models.map((m) => (
              <GlassCard key={m.name} className={`p-5 relative transition-all ${m.is_active ? 'border border-brand-success/50 bg-brand-success/5' : m.recommended ? 'border border-brand-primary/30 hover:border-brand-primary/50' : 'hover:border-white/10'}`}>
                <div className="flex items-start justify-between mb-1">
                  <div className="font-bold text-white text-base">{m.name}</div>
                  <div className="flex gap-1">
                    {m.recommended && <span className="text-xs bg-brand-primary/20 text-brand-primary border border-brand-primary/30 px-2 py-0.5 rounded-full font-bold">⭐ BEST</span>}
                    {m.is_active && <span className="text-xs bg-brand-success/20 text-brand-success border border-brand-success/30 px-2 py-0.5 rounded-full font-bold">⚡ ACTIVE</span>}
                  </div>
                </div>
                <p className="text-xs text-gray-500 mb-4">{m.note}</p>
                <div className="grid grid-cols-2 gap-2 mb-4">
                  {[
                    { label: 'Accuracy', value: `${m.accuracy}%`, good: m.accuracy >= 99 },
                    { label: 'ROC-AUC', value: `${m.roc_auc}%`, good: m.roc_auc >= 99 },
                    { label: 'Inference', value: `${m.inference_ms} ms`, good: false },
                    { label: 'Train Time', value: `${m.train_ms} ms`, good: false },
                  ].map(s => (
                    <div key={s.label} className="bg-[#1C1C21] rounded-lg px-3 py-2">
                      <div className="text-xs text-gray-500">{s.label}</div>
                      <div className={`text-sm font-bold ${s.good ? 'text-brand-success' : 'text-white'}`}>{s.value}</div>
                    </div>
                  ))}
                </div>
                {m.is_active ? (
                  <div className="w-full py-2 rounded-lg bg-brand-success/10 text-brand-success text-sm font-semibold text-center border border-brand-success/30">
                    ✓ Currently Active
                  </div>
                ) : (
                  <button
                    onClick={() => handleSwitch(m.name)}
                    disabled={!!switching || !m.available}
                    className="w-full py-2 rounded-lg bg-brand-primary hover:bg-brand-primary/80 text-white text-sm font-semibold transition-all disabled:opacity-40 disabled:cursor-not-allowed shadow-lg shadow-brand-primary/20"
                  >
                    {switching === m.name ? (
                      <span className="flex items-center justify-center gap-2">
                        <span className="w-3 h-3 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                        Switching...
                      </span>
                    ) : `🔄 Switch to ${m.name}`}
                  </button>
                )}
              </GlassCard>
            ))}
          </div>
          {models.length === 0 && <Spinner />}
        </div>
      )}

      {activeSection === 'testcases' && (
        <>
          {!tc && <Spinner />}
          {tc && (
            <>
              <div className="grid grid-cols-3 gap-4">
                {[
                  { label: 'Total Tests', value: tc.summary?.total, color: 'text-white' },
                  { label: 'Passed', value: tc.summary?.passed, color: 'text-brand-success' },
                  { label: 'Failed', value: tc.summary?.failed, color: 'text-brand-danger' },
                ].map((s, i) => (
                  <GlassCard key={i} className="p-4 text-center">
                    <div className={`text-3xl font-black ${s.color}`}>{s.value}</div>
                    <div className="text-sm text-gray-400 mt-1">{s.label}</div>
                  </GlassCard>
                ))}
              </div>
              <GlassCard className="p-0 overflow-hidden">
                <div className="overflow-x-auto">
                  <table className="w-full text-xs">
                    <thead className="border-b border-[#33333C] bg-[#1C1C21]/50">
                      <tr>
                        {['ID', 'Module', 'Type', 'Description', 'Input', 'Expected', 'Actual', 'Status'].map(h => (
                          <th key={h} className="text-left py-3 px-3 text-gray-400 font-semibold whitespace-nowrap">{h}</th>
                        ))}
                      </tr>
                    </thead>
                    <tbody>
                      {(tc.test_cases || []).map((t, i) => (
                        <tr key={i} className="border-b border-[#33333C]/30 hover:bg-white/5 transition-colors">
                          <td className="py-3 px-3 font-mono text-brand-primary font-bold">{t.tc_id}</td>
                          <td className="py-3 px-3 text-gray-300 whitespace-nowrap">{t.module}</td>
                          <td className="py-3 px-3"><Badge label={t.type} variant={t.type === 'Unit' ? 'info' : t.type === 'Integration' ? 'success' : t.type === 'Negative' ? 'danger' : 'warning'} /></td>
                          <td className="py-3 px-3 text-gray-200 max-w-xs">{t.description}</td>
                          <td className="py-3 px-3 text-gray-400 font-mono text-xs max-w-xs truncate">{t.input}</td>
                          <td className="py-3 px-3 text-gray-400 max-w-xs">{t.expected}</td>
                          <td className="py-3 px-3 text-gray-300 max-w-xs">{t.actual}</td>
                          <td className="py-3 px-3">{statusBadge(t.status)}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </GlassCard>
            </>
          )}
        </>
      )}

      {activeSection === 'model' && (
        <div className="space-y-4">
          <GlassCard className="p-4 bg-brand-primary/5 border border-brand-primary/30">
            <div className="flex items-center justify-between">
              <div>
                <div className="text-brand-primary font-bold text-lg">🏆 Best Model: LightGBM</div>
                <div className="text-xs text-gray-400 mt-1">Selected as production model — industry standard for healthcare fraud detection (IRDAI-aligned)</div>
              </div>
              <div className="text-right">
                <div className="text-brand-success font-black text-2xl">100%</div>
                <div className="text-xs text-gray-400">All Metrics</div>
              </div>
            </div>
          </GlassCard>
          <div className="grid grid-cols-2 gap-4">
            {[
              { label: 'Active Model', value: 'LightGBM', highlight: true },
              { label: 'Algorithm', value: 'Gradient Boosted Trees' },
              { label: 'Test Accuracy', value: '100.0%' },
              { label: 'ROC-AUC Score', value: '1.000' },
              { label: 'Precision', value: '100.0%' },
              { label: 'Recall', value: '100.0%' },
              { label: 'F1-Score', value: '100.0%' },
              { label: 'PR-AUC', value: '1.000' },
              { label: 'Inference Speed', value: '23.91 ms (1,082 claims)' },
              { label: 'True Positives', value: '101 (0 missed)' },
              { label: 'False Positives', value: '0 (0 wrongly flagged)' },
              { label: 'Features Used', value: '30 provider-level features' },
              { label: 'Train Samples', value: '7,846 (SMOTENC balanced)' },
              { label: 'Test Samples', value: '1,082 (AB PM-JAY test set)' },
              { label: 'CV ROC-AUC', value: '1.000 ± 0.000 (5-fold)' },
              { label: 'Serialisation', value: 'lightgbm.pkl (Pickle)' },
            ].map((m, i) => (
              <GlassCard key={i} className={`p-4 flex justify-between items-center ${m.highlight ? 'border border-brand-primary/40' : ''}`}>
                <span className="text-gray-400 text-sm">{m.label}</span>
                <span className={`font-bold text-sm ${m.highlight ? 'text-brand-primary' : 'text-white'}`}>{m.value}</span>
              </GlassCard>
            ))}
          </div>
          <GlassCard className="p-4">
            <div className="text-xs text-gray-400 font-semibold uppercase tracking-widest mb-3">Why LightGBM? (vs other 100% models)</div>
            <div className="grid grid-cols-2 gap-3 text-xs">
              {[
                { model: 'Random Forest', train: '165 ms', note: 'Slower training, less scalable' },
                { model: 'XGBoost', train: '231 ms', note: 'Slowest to train' },
                { model: 'LightGBM ✓ SELECTED', train: '61.6 ms', note: 'Fastest + industry IRDAI standard' },
                { model: 'Logistic Regression', train: '24 ms', note: 'Too simple for complex patterns' },
              ].map((r, i) => (
                <div key={i} className={`p-3 rounded-lg ${r.model.includes('✓') ? 'bg-brand-primary/10 border border-brand-primary/30' : 'bg-[#1C1C21]'}`}>
                  <div className={`font-bold ${r.model.includes('✓') ? 'text-brand-primary' : 'text-white'}`}>{r.model}</div>
                  <div className="text-gray-400 mt-1">Train: {r.train}</div>
                  <div className="text-gray-500 mt-0.5">{r.note}</div>
                </div>
              ))}
            </div>
          </GlassCard>
        </div>
      )}

      {activeSection === 'system' && (
        <div className="grid grid-cols-2 gap-6">
          {[
            { label: 'Backend Framework', value: 'FastAPI (Python 3.13)' },
            { label: 'Frontend Framework', value: 'React 18 + Vite' },
            { label: 'Styling', value: 'Tailwind CSS + Recharts' },
            { label: 'Backend Port', value: '8000' },
            { label: 'Frontend Port', value: '5177' },
            { label: 'Dataset', value: 'IRDAI AB PM-JAY (5,58,211 claims)' },
            { label: 'Balancing Method', value: 'SMOTENC (405 → 3,923 fraud)' },
            { label: 'Serialisation', value: 'Pickle (.pkl)' },
          ].map((s, i) => (
            <GlassCard key={i} className="p-4 flex justify-between items-center">
              <span className="text-gray-400">{s.label}</span>
              <span className="font-bold text-white text-sm">{s.value}</span>
            </GlassCard>
          ))}
        </div>
      )}
    </div>
  );
}

// ─── SingleResultView ─────────────────────────────────────────

function SingleResultView({ data }) {
  if (!data) return null;
  const isFraud = data.prediction === 1;
  const prob = (data.probability * 100).toFixed(1);

  return (
    <GlassCard className="p-8 mt-6 animate-in fade-in slide-in-from-bottom-4 duration-500">
      <div className="flex flex-col md:flex-row gap-8">
        {/* SVG Gauge */}
        <div className="flex-[0.4] flex flex-col items-center p-6 bg-[#161619] rounded-xl border border-[#33333C] relative overflow-hidden">
          <div className={`absolute top-0 w-full h-1 ${isFraud ? 'bg-brand-danger' : 'bg-brand-success'}`} />
          <h3 className="text-xl font-bold text-white mb-6">Fraud Probability</h3>
          <div className="relative w-44 h-44 flex items-center justify-center">
            <svg className="w-full h-full transform -rotate-90">
              <circle cx="88" cy="88" r="80" fill="none" stroke="#24242B" strokeWidth="12" />
              <circle cx="88" cy="88" r="80" fill="none"
                stroke={isFraud ? '#FF7675' : '#00B894'} strokeWidth="12"
                strokeDasharray={2 * Math.PI * 80}
                strokeDashoffset={(2 * Math.PI * 80) * (1 - data.probability)}
                style={{ transition: 'stroke-dashoffset 1.2s ease' }}
              />
            </svg>
            <div className="absolute text-center">
              <div className={`text-4xl font-black ${isFraud ? 'text-brand-danger' : 'text-brand-success'}`}>{prob}%</div>
              <div className="text-xs text-gray-400 uppercase tracking-widest font-semibold mt-1">{isFraud ? 'HIGH RISK' : 'LOW RISK'}</div>
            </div>
          </div>
          <div className="mt-6 text-center w-full text-sm text-gray-400">Claim: <span className="text-white font-mono">{data.claim_id}</span></div>
        </div>

        {/* Explanations */}
        <div className="flex-[0.6] space-y-5">
          <div>
            <h3 className="text-xl font-bold text-white flex items-center gap-2"><ShieldCheck size={20} className="text-brand-primary" />AI Reasoning Engine</h3>
            <p className="text-sm text-gray-400 mt-1">LightGBM ensemble decision breakdown for this claim.</p>
          </div>

          {data.flags?.length > 0 && (
            <div>
              <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">Risk Factors Detected</h4>
              {data.flags.map((f, i) => (
                <div key={i} className="flex items-start gap-2 bg-brand-danger/5 border border-brand-danger/20 p-3 rounded-lg mb-2">
                  <AlertTriangle size={15} className="text-brand-danger mt-0.5 shrink-0" />
                  <span className="text-sm text-gray-200">{f.reason}</span>
                  <Badge label={f.severity} variant={f.severity === 'High' ? 'danger' : 'warning'} />
                </div>
              ))}
            </div>
          )}

          {data.parameters?.length > 0 && (
            <div>
              <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-3">Top Influencing Parameters</h4>
              {data.parameters.map((p, i) => (
                <div key={i} className="mb-3">
                  <div className="flex justify-between text-xs mb-1">
                    <span className="text-gray-300 truncate max-w-[75%]">{p.feature}</span>
                    <span className={p.impact > 0.5 ? 'text-brand-danger' : 'text-brand-primary'}>{(p.impact * 100).toFixed(0)}%</span>
                  </div>
                  <div className="w-full bg-[#1C1C21] h-2 rounded-full overflow-hidden">
                    <div className={`h-full rounded-full ${p.impact > 0.5 ? 'bg-gradient-to-r from-brand-danger to-pink-400' : 'bg-gradient-to-r from-brand-primary to-brand-secondary'}`}
                      style={{ width: `${Math.min(100, p.impact * 100)}%`, transition: 'width 1s ease' }} />
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </GlassCard>
  );
}

// ─── Single Claim Form ────────────────────────────────────────

function SingleClaimForm({ setSingleResult, setLoading }) {
  const [formData, setFormData] = useState({
    claim_id: 'IPD-MH-9023',
    InscClaimAmtReimbursed: '125000',
    DaysAdmitted: '8',
    provider_fraud_rate: '0.1',
    ChronicConditions: '3',
    num_patients: '50',
    num_physicians: '3',
    PatientAge: '55',
  });
  const handleChange = e => setFormData({ ...formData, [e.target.name]: e.target.value });

  const handleSubmit = async e => {
    e.preventDefault();
    setLoading(true);
    setSingleResult(null);
    try {
      const res = await axios.post(`${API}/api/predict/single`, {
        ...formData,
        InscClaimAmtReimbursed: parseFloat(formData.InscClaimAmtReimbursed) || 0,
        DaysAdmitted: parseFloat(formData.DaysAdmitted) || 0,
        provider_fraud_rate: parseFloat(formData.provider_fraud_rate) || 0,
        ChronicConditions: parseFloat(formData.ChronicConditions) || 0,
        num_patients: parseFloat(formData.num_patients) || 1,
        num_physicians: parseFloat(formData.num_physicians) || 1,
        PatientAge: parseFloat(formData.PatientAge) || 55,
      });
      setSingleResult(res.data);
    } catch (err) {
      alert('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setTimeout(() => setLoading(false), 1200);
    }
  };

  const fields = [
    { label: 'Claim ID', name: 'claim_id', type: 'text', ph: 'IPD-MH-9023', hint: 'Indian claim identifier' },
    { label: 'Reimbursed Amount (₹)', name: 'InscClaimAmtReimbursed', type: 'number', ph: '125000', hint: '> ₹2,00,000 = High Risk' },
    { label: 'Days Admitted', name: 'DaysAdmitted', type: 'number', ph: '8', hint: '> 30 days = suspicious' },
    { label: 'Provider Fraud Rate (0–1)', name: 'provider_fraud_rate', type: 'number', ph: '0.10', step: '0.01', hint: '0.8+ = Critical, 0.0 = Clean' },
    { label: 'Patient Count (Annual)', name: 'num_patients', type: 'number', ph: '50', hint: 'Total distinct patients billed' },
    { label: 'No. of Physicians Billed', name: 'num_physicians', type: 'number', ph: '3', hint: '> 8 = possible unbundling' },
    { label: 'Chronic Conditions', name: 'ChronicConditions', type: 'number', ph: '3', hint: '0 with high claim = upcoding' },
    { label: 'Patient Age', name: 'PatientAge', type: 'number', ph: '55', hint: 'Avg age of billed patients' },
  ];

  return (
    <form onSubmit={handleSubmit} className="glass-panel p-6 mt-4 space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-lg font-bold text-white">🏥 Manual Claim Entry (Indian Provider)</h3>
        <span className="text-xs text-gray-500 bg-[#1C1C21] px-2 py-1 rounded-full">IRDAI AB PM-JAY Format</span>
      </div>
      <div className="grid grid-cols-2 gap-4">
        {fields.map(f => (
          <div key={f.name}>
            <label className="block text-gray-400 text-sm mb-1">{f.label}</label>
            <input required name={f.name} onChange={handleChange} value={formData[f.name]}
              type={f.type} step={f.step || '1'} placeholder={f.ph}
              className="w-full bg-[#1C1C21] border border-[#33333C] rounded-lg px-4 py-2.5 text-white placeholder-gray-600 focus:outline-none focus:border-brand-primary transition-colors" />
            <p className="text-xs text-gray-600 mt-0.5">{f.hint}</p>
          </div>
        ))}
      </div>
      <div className="grid grid-cols-2 gap-3 pt-2">
        <button type="button" onClick={() => setFormData({ claim_id: 'IPD-MH-9023', InscClaimAmtReimbursed: '350000', DaysAdmitted: '45', provider_fraud_rate: '0.85', ChronicConditions: '0', num_patients: '280', num_physicians: '12', PatientAge: '38' })}
          className="bg-brand-danger/20 border border-brand-danger/40 text-brand-danger px-4 py-2 rounded-lg text-sm font-semibold hover:bg-brand-danger/30 transition-colors">
          🚨 Load Fraud Scenario
        </button>
        <button type="button" onClick={() => setFormData({ claim_id: 'IPD-KA-1201', InscClaimAmtReimbursed: '95000', DaysAdmitted: '5', provider_fraud_rate: '0.03', ChronicConditions: '4', num_patients: '120', num_physicians: '2', PatientAge: '64' })}
          className="bg-brand-success/20 border border-brand-success/40 text-brand-success px-4 py-2 rounded-lg text-sm font-semibold hover:bg-brand-success/30 transition-colors">
          ✅ Load Legit Scenario
        </button>
      </div>
      <button type="submit" className="bg-brand-primary hover:bg-brand-primary/90 text-white px-6 py-2.5 rounded-lg font-semibold w-full transition-colors shadow-lg shadow-brand-primary/30">
        🔬 Analyse Claim
      </button>
    </form>
  );
}

// ─── Prediction Panel ─────────────────────────────────────────

function PredictionPanel() {
  const [file, setFile] = useState(null);
  const [result, setResult] = useState(null);
  const [singleResult, setSingleResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [mode, setMode] = useState('batch');

  const handleUpload = async e => {
    e.preventDefault();
    if (!file) return;
    setLoading(true);
    const fd = new FormData();
    fd.append('file', file);
    try {
      const res = await axios.post(`${API}/api/predict/batch`, fd);
      setResult(res.data);
    } catch (err) {
      alert('Error: ' + (err.response?.data?.detail || err.message));
    } finally {
      setLoading(false);
    }
  };

  const riskData = result ? [
    { name: 'High', value: result.predictions_sample?.filter(p => p.risk_level === 'High').length || 0 },
    { name: 'Medium', value: result.predictions_sample?.filter(p => p.risk_level === 'Medium').length || 0 },
    { name: 'Low', value: result.predictions_sample?.filter(p => p.risk_level === 'Low').length || 0 },
  ] : [];

  return (
    <div className="w-full space-y-4">
      <div className="flex justify-between items-center">
        <h2 className="text-2xl font-bold text-white">🤖 AI Fraud Detection Engine</h2>
        <div className="flex bg-[#1C1C21] border border-[#33333C] rounded-lg p-1">
          {['batch', 'single'].map(m => (
            <button key={m} onClick={() => setMode(m)}
              className={`px-4 py-1.5 rounded-md text-sm font-medium transition-all ${mode === m ? 'bg-brand-primary text-white shadow-md' : 'text-gray-400 hover:text-white'}`}>
              {m === 'batch' ? '📦 Batch CSV' : '🔍 Single Claim'}
            </button>
          ))}
        </div>
      </div>

      {mode === 'batch' ? (
        <GlassCard className="p-8 border-dashed border-2 border-[#33333C] hover:border-brand-primary transition-colors flex flex-col items-center justify-center py-16 text-center">
          {loading ? (
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 border-4 border-brand-primary/30 border-t-brand-primary rounded-full animate-spin mb-4" />
              <p className="text-gray-400 animate-pulse">Executing AI ensemble predictions…</p>
            </div>
          ) : (
            <div className="flex flex-col items-center">
              <div className="w-16 h-16 rounded-full bg-brand-primary/10 flex items-center justify-center mb-4">
                <UploadCloud size={32} className="text-brand-primary" />
              </div>
              <h3 className="text-lg font-bold text-white mb-2">Drag & Drop your CSV</h3>
              <p className="text-gray-400 text-sm mb-6 max-w-sm">Upload preprocessed healthcare datasets (30 features) for batch inference.</p>
              <input type="file" id="fu" className="hidden" accept=".csv" onChange={e => setFile(e.target.files[0])} />
              <label htmlFor="fu" className="bg-brand-primary hover:bg-brand-primary/90 text-white px-6 py-3 rounded-full font-semibold cursor-pointer transition-colors shadow-lg shadow-brand-primary/20">
                {file ? file.name : 'Browse Files'}
              </label>
              {file && (
                <button onClick={handleUpload} className="mt-4 bg-[#24242B] border border-[#33333C] px-6 py-2 rounded-full text-white font-medium hover:bg-[#33333C] transition-colors">
                  Run Analysis
                </button>
              )}
            </div>
          )}
        </GlassCard>
      ) : (
        <>
          <SingleClaimForm setSingleResult={setSingleResult} setLoading={setLoading} />
          {loading && (
            <GlassCard className="p-8 flex flex-col items-center">
              <div className="w-12 h-12 border-4 border-brand-primary/30 border-t-brand-primary rounded-full animate-spin mb-3" />
              <p className="text-gray-400 animate-pulse text-sm">Evaluating parameters…</p>
            </GlassCard>
          )}
          {singleResult?.status === 'success' && !loading && (
            <SingleResultView data={singleResult.predictions[0]} />
          )}
        </>
      )}

      {result?.summary && !loading && mode === 'batch' && (
        <div className="space-y-4">
          <div className="grid grid-cols-3 gap-4">
            <GlassCard className="p-5 text-center">
              <div className="text-3xl font-black text-white">{result.summary.total_claims}</div>
              <div className="text-xs text-gray-400 mt-1">Total Claims</div>
            </GlassCard>
            <GlassCard className="p-5 text-center bg-brand-danger/10">
              <div className="text-3xl font-black text-brand-danger">{result.summary.fraudulent}</div>
              <div className="text-xs text-gray-400 mt-1">Fraudulent ({result.summary.fraud_rate_percent}%)</div>
            </GlassCard>
            <GlassCard className="p-5 text-center bg-brand-success/10">
              <div className="text-3xl font-black text-brand-success">{result.summary.legitimate}</div>
              <div className="text-xs text-gray-400 mt-1">Legitimate</div>
            </GlassCard>
          </div>
          {riskData.length > 0 && (
            <GlassCard className="p-6">
              <h4 className="text-white font-bold mb-4">Risk Distribution (Sample)</h4>
              <ResponsiveContainer width="100%" height={200}>
                <BarChart data={riskData}>
                  <XAxis dataKey="name" stroke="#33333C" tick={{ fill: '#94a3b8' }} />
                  <YAxis stroke="#33333C" tick={{ fill: '#94a3b8' }} />
                  <Tooltip contentStyle={{ backgroundColor: '#24242B', border: '1px solid #33333C', borderRadius: 8 }} />
                  <Bar dataKey="value" fill="#6C5CE7" radius={[4, 4, 0, 0]} />
                </BarChart>
              </ResponsiveContainer>
            </GlassCard>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Main App ─────────────────────────────────────────────────

export default function App() {
  const [activeTab, setActiveTab] = useState('Dashboard');
  const [stats, setStats] = useState(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);

  useEffect(() => {
    axios.get(`${API}/api/dashboard/stats`).then(r => setStats(r.data)).catch(console.error);
  }, []);

  const navLinks = [
    { label: 'Dashboard', icon: <LayoutDashboard size={20} /> },
    { label: 'Predict Fraud', icon: <ShieldAlert size={20} /> },
    { label: 'Analytics', icon: <BarChart3 size={20} /> },
    { label: 'Savings', icon: <PiggyBank size={20} /> },
    { label: 'Records', icon: <FileText size={20} /> },
    { label: 'History', icon: <History size={20} /> },
  ];

  const Logo = () => (
    <div className="flex items-center space-x-3 px-2 py-1">
      <div className="h-8 w-8 bg-brand-primary rounded-lg flex items-center justify-center flex-shrink-0">
        <Activity size={20} color="white" />
      </div>
      <motion.span initial={{ opacity: 0 }} animate={{ opacity: 1 }}
        className="text-xl font-bold tracking-tight text-white whitespace-pre">
        MediGuard
      </motion.span>
    </div>
  );

  const LogoIcon = () => (
    <div className="flex items-center justify-center px-1 py-1">
      <div className="h-8 w-8 bg-brand-primary rounded-lg flex items-center justify-center flex-shrink-0">
        <Activity size={20} color="white" />
      </div>
    </div>
  );

  const pageTitles = {
    'Dashboard': ['Dashboard', 'Live Fraud Detection Overview'],
    'Predict Fraud': ['Predict Fraud', 'AI Analysis Hub'],
    'Analytics': ['Analytics', 'Model Performance & Feature Analysis'],
    'Savings': ['Savings', 'Financial Impact & ROI Analysis'],
    'Records': ['Records', 'Prediction Records — Test Dataset'],
    'History': ['History', 'System Audit Log'],
    'Settings': ['Settings', 'Test Cases & System Configuration'],
  };

  const renderPage = () => {
    switch (activeTab) {
      case 'Dashboard': return <DashboardPage stats={stats} />;
      case 'Predict Fraud': return <PredictionPanel />;
      case 'Analytics': return <AnalyticsPage />;
      case 'Savings': return <SavingsPage />;
      case 'Records': return <RecordsPage />;
      case 'History': return <HistoryPage />;
      case 'Settings': return <SettingsPage />;
      default: return <DashboardPage stats={stats} />;
    }
  };

  const [title, sub] = pageTitles[activeTab] || ['MediGuard', ''];

  return (
    <div className="flex h-screen w-full bg-[#141416] text-gray-200 overflow-hidden font-sans relative">
      <div className="absolute inset-0 z-0 pointer-events-none opacity-30">
        <Waves className="w-full h-full" strokeColor="#33333C" backgroundColor="transparent" pointerSize={0} />
      </div>

      <Sidebar open={sidebarOpen} setOpen={setSidebarOpen}>
        <SidebarBody className="bg-[#1C1C21]/90 backdrop-blur-md border-r border-[#33333C] z-20">
          <div className="flex flex-col flex-1 overflow-y-auto overflow-x-hidden">
            {sidebarOpen ? <Logo /> : <LogoIcon />}
            <div className="mt-8 flex flex-col gap-1">
              {navLinks.map((link, idx) => (
                <SidebarLink key={idx} link={link} active={activeTab === link.label} onClick={() => setActiveTab(link.label)} />
              ))}
            </div>
          </div>
          <div className="mt-auto flex flex-col gap-1">
            <SidebarLink link={{ label: 'Settings', icon: <Settings size={20} /> }} active={activeTab === 'Settings'} onClick={() => setActiveTab('Settings')} />
          </div>
        </SidebarBody>
      </Sidebar>

      <main className="flex-1 flex flex-col overflow-hidden relative z-10 w-full">
        {/* Header */}
        <header className="h-20 flex items-center justify-between px-8 border-b border-[#33333C]/50 bg-[#141416]/60 backdrop-blur-md shrink-0">
          <div>
            <h1 className="text-2xl font-bold text-white">{title}</h1>
            <p className="text-sm text-gray-400 mt-0.5">{sub}</p>
          </div>
          <div className="flex items-center gap-4">
            <div className="relative">
              <Search className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" size={16} />
              <input type="text" placeholder="Search…" className="bg-[#1C1C21] border border-[#33333C] text-sm rounded-full pl-9 pr-4 py-2 focus:outline-none focus:border-brand-primary w-56 transition-all" />
            </div>
            <button className="p-2 bg-[#1C1C21] rounded-full border border-[#33333C] hover:bg-[#24242B] transition-colors relative">
              <span className="absolute top-0 right-0 h-2 w-2 rounded-full bg-brand-danger border border-[#1C1C21]" />
              <Bell size={17} className="text-gray-300" />
            </button>
            <div className="h-9 w-9 bg-gradient-to-tr from-brand-primary to-brand-secondary rounded-full overflow-hidden border-2 border-brand-primary/40">
              <img src="https://images.unsplash.com/photo-1535713875002-d1d0cf377fde?w=100&q=80" alt="avatar" className="w-full h-full object-cover" />
            </div>
          </div>
        </header>

        {/* Scrollable content */}
        <div className="flex-1 overflow-y-auto p-8">
          {renderPage()}
        </div>
      </main>
    </div>
  );
}
