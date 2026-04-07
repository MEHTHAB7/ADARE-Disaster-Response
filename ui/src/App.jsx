import React, { useState, useEffect } from 'react';
import { MapContainer, TileLayer, Marker, Popup, useMap } from 'react-leaflet';
import L from 'leaflet';
import { Activity, Shield, Users, Timer, Play, RotateCcw, AlertTriangle, Cpu, Globe, Info } from 'lucide-react';

// FIX: Leaflet default icon issues in React
import 'leaflet/dist/leaflet.css';

const ambulanceIcon = new L.Icon({
  iconUrl: 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="35" height="35"><circle cx="12" cy="12" r="10" fill="%233b82f6" stroke="%23ffffff" stroke-width="2"/><text x="12" y="16" font-size="12" text-anchor="middle" fill="white" font-family="sans-serif">A</text></svg>',
  iconSize: [35, 35],
  iconAnchor: [17, 17],
  popupAnchor: [0, -17],
});

const victimIcon = (severity) => new L.Icon({
  iconUrl: severity > 7 
    ? 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28"><circle cx="12" cy="12" r="10" fill="%23ef4444" stroke="%23ffffff" stroke-width="2"/><text x="12" y="16" font-size="12" text-anchor="middle" fill="white" font-weight="bold" font-family="sans-serif">!</text></svg>'
    : severity > 3 
    ? 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28"><circle cx="12" cy="12" r="10" fill="%23f97316" stroke="%23ffffff" stroke-width="2"/></svg>'
    : 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 24 24" width="28" height="28"><circle cx="12" cy="12" r="10" fill="%23eab308" stroke="%23ffffff" stroke-width="2"/></svg>',
  iconSize: [28, 28],
  iconAnchor: [14, 14],
});

function MapUpdater({ center }) {
  const map = useMap();
  useEffect(() => {
    if (center && center.lat && center.lng) {
        map.setView([center.lat, center.lng]);
    }
  }, [center]);
  return null;
}

export default function App() {
  const [data, setData] = useState({ agents: [], victims: [], rescued: 0, total: 0, time: 0, reward: 0, location: '--' });
  const [params, setParams] = useState({ difficulty: 1, agent_type: 'heuristic', location: '' });
  const [loading, setLoading] = useState(false);
  const [isLive, setIsLive] = useState(false);

  useEffect(() => {
    const interval = setInterval(() => {
      fetch('/state')
        .then(res => res.json())
        .then(d => {
            if (d.location) setData(d);
        })
        .catch(() => {});
    }, 500);
    return () => clearInterval(interval);
  }, []);

  const handleStart = async () => {
    setLoading(true);
    try {
        const res = await fetch('/start', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(params)
        });
        await res.json();
        setIsLive(true);
    } catch(e) {
        alert("Failed to connect to simulation backend. Is server.py running?");
    }
    setLoading(false);
  };

  const center = (data.agents && data.agents[0]) || { lat: 40.7128, lng: -74.0060 };

  return (
    <div style={{ display: 'flex', height: '100vh', width: '100vw', padding: '1.25rem', gap: '1.25rem', overflow: 'hidden' }}>
      
      {/* Sidebar Panel */}
      <aside className="glass" style={{ width: '420px', padding: '2rem', display: 'flex', flexDirection: 'column', zIndex: 10 }}>
        <header style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '2.5rem' }}>
          <div style={{ background: 'linear-gradient(135deg, #3b82f6, #8b5cf6)', padding: '0.6rem', borderRadius: '12px' }}>
            <Shield size={28} color="white" />
          </div>
          <div>
            <h1 style={{ fontSize: '1.65rem', fontWeight: 800, letterSpacing: '-0.03em', background: 'linear-gradient(to right, #fff, #94a3b8)', WebkitBackgroundClip: 'text', WebkitTextFillColor: 'transparent' }}>
                ADRAE++
            </h1>
            <p style={{ fontSize: '0.75rem', opacity: 0.5, fontWeight: 500, textTransform: 'uppercase', letterSpacing: '0.1em' }}>
                Digital Twin AI Optimizer
            </p>
          </div>
        </header>

        {/* Real-time Status Card */}
        <section style={{ background: 'rgba(255,255,255,0.03)', borderRadius: '16px', padding: '1.25rem', marginBottom: '2rem', border: '1px solid rgba(255,255,255,0.1)' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
                <span style={{ fontSize: '0.85rem', fontWeight: 600, color: '#94a3b8' }}>Mission Phase</span>
                <span style={{ fontSize: '0.75rem', padding: '0.2rem 0.6rem', borderRadius: '100px', background: isLive ? 'rgba(16, 185, 129, 0.1)' : 'rgba(239, 68, 68, 0.1)', color: isLive ? '#10b981' : '#ef4444', fontWeight: 700 }}>
                    {isLive ? 'LIVE OPERATIONAL' : 'OFFLINE'}
                </span>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 700 }}>{data.rescued}/{data.total}</span>
                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>Saved Targets</span>
              </div>
              <div style={{ display: 'flex', flexDirection: 'column' }}>
                <span style={{ fontSize: '1.5rem', fontWeight: 700 }}>{data.time}T</span>
                <span style={{ fontSize: '0.7rem', opacity: 0.5 }}>Elapsed Time</span>
              </div>
            </div>
        </section>

        {/* Metrics Grid */}
        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem', marginBottom: '2rem' }}>
          <StatCard icon={<Activity size={18} color="#f59e0b"/>} label="Net Reward" value={data.reward?.toFixed(1) || '0.0'} />
          <StatCard icon={<Globe size={18} color="#3b82f6"/>} label="Area" value={data.location?.split(',')[0] || '--'} />
        </div>

        {/* Controls */}
        <section style={{ marginTop: 'auto', display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
          <div className="glass" style={{ padding: '1.25rem', background: 'rgba(255,255,255,0.02)' }}>
            <h3 style={{ fontSize: '0.9rem', marginBottom: '1rem', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Cpu size={16} /> Strategy Configuration
            </h3>
            
            <div style={{ marginBottom: '1rem' }}>
                <label style={{ fontSize: '0.7rem', opacity: 0.5, display: 'block', marginBottom: '0.4rem' }}>AGENT MODEL</label>
                <select 
                    value={params.agent_type}
                    onChange={e => setParams({...params, agent_type: e.target.value})}
                    className="glass" 
                    style={{ width: '100%', padding: '0.75rem', outline: 'none', fontSize: '0.9rem', border: '1px solid rgba(255,255,255,0.1)' }}>
                    <option value="heuristic">Heuristic++ (Graph Optimized)</option>
                    <option value="hybrid">Hybrid RL + LLM (Deep Reasoning)</option>
                    <option value="rl">RL Brain (PPO Strategy)</option>
                </select>
            </div>

            <div>
                <label style={{ fontSize: '0.7rem', opacity: 0.5, display: 'block', marginBottom: '0.4rem' }}>SCENARIO DIFFICULTY TIER</label>
                <select 
                    value={params.difficulty}
                    onChange={e => setParams({...params, difficulty: parseInt(e.target.value)})}
                    className="glass" 
                    style={{ width: '100%', padding: '0.75rem', outline: 'none', fontSize: '0.9rem', border: '1px solid rgba(255,255,255,0.1)' }}>
                    <option value="1">Easy (Basic Routine Dispatch)</option>
                    <option value="3">Medium (Obstacle Navigation)</option>
                    <option value="5">Hard (Area Swarm Fleet)</option>
                    <option value="7">Expert (Absolute Urban Chaos)</option>
                </select>
            </div>
          </div>
          
          <button onClick={handleStart} className="btn btn-primary" disabled={loading} style={{ height: '56px' }}>
            {loading ? <Activity size={20} className="pulse" /> : <Play size={22} fill="currentColor" />}
            {loading ? "Optimizing Environment..." : "Deploy Active Response"}
          </button>
        </section>
      </aside>

      {/* Main Viewport */}
      <main style={{ flex: 1, position: 'relative' }} className="glass">
        {/* Header Overlay */}
        <div style={{ position: 'absolute', top: '1.5rem', left: '1.5rem', zIndex: 1000, display: 'flex', gap: '1rem' }}>
             <div className="glass" style={{ padding: '0.5rem 1rem', display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.85rem' }}>
                <div style={{ width: '8px', height: '8px', borderRadius: '50%', background: '#10b981', boxShadow: '0 0 10px #10b981' }}></div>
                Data Pipeline: Active Street Map
             </div>
        </div>

        <MapContainer 
            center={[center.lat, center.lng]} 
            zoom={16} 
            zoomControl={false}
            style={{ height: '100%', width: '100%' }}>
          
          <TileLayer 
             url="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
             attribution="Tiles &copy; Esri &mdash; Source: Esri, i-cubed, USDA, USGS, AEX, GeoEye, Getmapping, Aerogrid, IGN, IGP, UPR-EGP, and the GIS User Community"
          />
          
          {data.agents?.map((a, i) => (
            <Marker key={`agent-${i}`} position={[a.lat, a.lng]} icon={ambulanceIcon}>
                <Popup>Rescue Unit {i+1}<br/>Strategy: {params.agent_type}</Popup>
            </Marker>
          ))}

          {data.victims?.map((v, i) => (
            <Marker key={`vict-${i}`} position={[v.lat, v.lng]} icon={victimIcon(v.severity)}>
                <Popup>
                    <strong style={{ color: v.status === 'rescued' ? '#10b981' : '#ef4444' }}>
                        {v.status.toUpperCase()}
                    </strong><br/>
                    <strong>Priority: </strong> {v.severity > 7 ? <span style={{color: '#ef4444', fontWeight: 'bold'}}>HIGH</span> : v.severity > 3 ? <span style={{color: '#f97316', fontWeight: 'bold'}}>MEDIUM</span> : <span style={{color: '#eab308', fontWeight: 'bold'}}>LOW</span>}<br/>
                    Severity Index: {v.severity}<br/>
                    Node ID: {i}
                </Popup>
            </Marker>
          ))}
          
          <MapUpdater center={center} />
        </MapContainer>

        {/* Legend */}
        <div style={{ position: 'absolute', bottom: '1.5rem', right: '1.5rem', zIndex: 1000 }} className="glass">
           <div style={{ padding: '0.75rem 1rem', display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.75rem' }}>
             <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <div style={{ width: '10px', height: '10px', background: '#3b82f6', borderRadius: '2px' }}></div> Agent
             </div>
             <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <div style={{ width: '10px', height: '10px', background: '#ef4444', borderRadius: '50%' }}></div> High Severity
             </div>
             <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <div style={{ width: '10px', height: '10px', background: '#f97316', borderRadius: '50%' }}></div> Medium Severity
             </div>
             <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <div style={{ width: '10px', height: '10px', background: '#eab308', borderRadius: '50%' }}></div> Low Severity
             </div>
           </div>
        </div>
      </main>
    </div>
  );
}

function StatCard({ icon, label, value }) {
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', padding: '0.75rem', borderRadius: '12px', background: 'rgba(255,255,255,0.02)', border: '1px solid rgba(255,255,255,0.05)' }}>
      <div style={{ color: icon.props.color, opacity: 0.8 }}>{icon}</div>
      <div>
        <p style={{ fontSize: '0.6rem', opacity: 0.5, textTransform: 'uppercase', fontWeight: 600 }}>{label}</p>
        <p style={{ fontSize: '0.95rem', fontWeight: 700 }}>{value}</p>
      </div>
    </div>
  );
}
