import { useState, useCallback } from 'react'
import Uploader from './components/Uploader'
import JobPanel from './components/JobPanel'
import History from './components/History'
import './App.css'

export default function App() {
  const [jobs, setJobs] = useState([])
  const [activeJobId, setActiveJobId] = useState(null)

  const addJob = useCallback((jobId, filename, accent) => {
    setJobs(prev => [{
      id: jobId,
      filename,
      accent,
      status: 'queued',
      createdAt: Date.now(),
    }, ...prev])
    setActiveJobId(jobId)
  }, [])

  const updateJob = useCallback((jobId, patch) => {
    setJobs(prev => prev.map(j => j.id === jobId ? { ...j, ...patch } : j))
  }, [])

  return (
    <div className="app">
      {/* Background orbs */}
      <div className="orb orb-1" />
      <div className="orb orb-2" />
      <div className="orb orb-3" />

      <header className="header">
        <div className="logo">
          <div className="logo-icon">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
              <path strokeLinecap="round" strokeLinejoin="round"
                d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
            </svg>
          </div>
          <div>
            <h1>VoiceBridge</h1>
            <p>Emotion-preserving accent conversion</p>
          </div>
        </div>
        <div className="header-badges">
          <span className="badge badge-stage">5-Stage Pipeline</span>
          <span className="badge badge-ai">OpenVoice v2</span>
          <span className="badge badge-emotion">Audio SER</span>
        </div>
      </header>

      <main className="main">
        <div className="left-panel">
          <Uploader onJobStarted={addJob} />

          {jobs.length > 0 && (
            <History
              jobs={jobs}
              activeJobId={activeJobId}
              onSelect={setActiveJobId}
            />
          )}
        </div>

        <div className="right-panel">
          {activeJobId ? (
            <JobPanel
              key={activeJobId}
              jobId={activeJobId}
              onUpdate={updateJob}
            />
          ) : (
            <EmptyState />
          )}
        </div>
      </main>
    </div>
  )
}

function EmptyState() {
  return (
    <div className="empty-state">
      <div className="empty-icon">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1">
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M9 9l10.5-3m0 6.553v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66a2.25 2.25 0 001.632-2.163zm0 0V2.25L9 5.25v10.303m0 0v3.75a2.25 2.25 0 01-1.632 2.163l-1.32.377a1.803 1.803 0 01-.99-3.467l2.31-.66A2.25 2.25 0 009 15.553z" />
        </svg>
      </div>
      <h3>Upload audio to get started</h3>
      <p>Drag & drop a WAV or MP3 file to convert Indian English accent to American or British English while preserving your voice identity and emotional tone.</p>
      <ul className="feature-list">
        <li><span className="dot dot-green"/>Detects emotion from audio tone, not just words</li>
        <li><span className="dot dot-blue"/>Clones your voice with OpenVoice v2</li>
        <li><span className="dot dot-purple"/>Preserves pitch contour and energy dynamics</li>
        <li><span className="dot dot-orange"/>Rewrites phonetics for target accent</li>
      </ul>
    </div>
  )
}
