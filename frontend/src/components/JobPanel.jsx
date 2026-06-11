import { useEffect, useState, useRef } from 'react'

const API = ''

const STAGES = [
  { key: 'stage1', label: 'Transcribe', icon: '📝' },
  { key: 'stage2', label: 'Features',   icon: '📊' },
  { key: 'stage3', label: 'Phonetics',  icon: '🔤' },
  { key: 'stage4', label: 'Synthesise', icon: '🔊' },
  { key: 'stage5', label: 'Prosody',    icon: '🎭' },
]

const EMOTION_COLORS = {
  angry:     '#ef4444',
  happy:     '#f59e0b',
  sad:       '#3b82f6',
  surprised: '#a855f7',
  neutral:   '#6b7280',
  fearful:   '#6366f1',
  disgust:   '#84cc16',
}

export default function JobPanel({ jobId, onUpdate }) {
  const [status, setStatus]   = useState('queued')
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)
  const [logs, setLogs]       = useState([])
  const [playing, setPlaying] = useState(false)
  const logsEndRef = useRef()
  const audioRef   = useRef()
  const esRef      = useRef()

  // Detect active stage from logs
  const activeStage = (() => {
    const last = [...logs].reverse().find(l => /stage [1-5]/i.test(l.msg))
    if (!last) return -1
    const m = last.msg.match(/stage (\d)/i)
    return m ? parseInt(m[1]) - 1 : -1
  })()

  // SSE log stream
  useEffect(() => {
    const es = new EventSource(`${API}/api/jobs/${jobId}/logs`)
    esRef.current = es

    es.onmessage = (e) => {
      const data = JSON.parse(e.data)
      if (data.__end__) {
        setStatus(data.status)
        es.close()
        // Fetch final job state
        fetch(`${API}/api/jobs/${jobId}`)
          .then(r => r.json())
          .then(job => {
            setResult(job.result)
            setError(job.error)
            onUpdate(jobId, { status: job.status, result: job.result })
          })
        return
      }
      setLogs(prev => [...prev, data])
      setStatus('running')
    }

    es.onerror = () => {
      // Retry handled by browser; just mark running
    }

    return () => es.close()
  }, [jobId])

  // Auto-scroll logs
  useEffect(() => {
    logsEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  const togglePlay = () => {
    const a = audioRef.current
    if (!a) return
    if (playing) { a.pause(); setPlaying(false) }
    else { a.play(); setPlaying(true) }
  }

  const downloadUrl = `${API}/api/jobs/${jobId}/output`
  const emotionColor = EMOTION_COLORS[result?.emotion_label] || '#6b7280'

  return (
    <div className="job-panel">
      {/* Stage progress bar */}
      <div className="card stages-card">
        <div className="stages-track">
          {STAGES.map((s, i) => {
            const done    = i < activeStage
            const current = i === activeStage
            const pending = i > activeStage
            return (
              <div key={s.key} className={`stage-item ${done ? 'done' : ''} ${current ? 'current' : ''} ${pending ? 'pending' : ''}`}>
                <div className="stage-bubble">
                  {done ? '✓' : s.icon}
                  {current && <span className="stage-pulse" />}
                </div>
                <span className="stage-label">{s.label}</span>
                {i < STAGES.length - 1 && (
                  <div className={`stage-line ${done ? 'line-done' : ''}`} />
                )}
              </div>
            )
          })}
        </div>

        <div className="status-row">
          <span className={`status-pill status-${status}`}>
            {status === 'running' && <span className="spinner-small" />}
            {status}
          </span>
        </div>
      </div>

      {/* Results (when done) */}
      {status === 'done' && result && (
        <div className="card results-card">
          <h2 className="card-title">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="card-icon">
              <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            Conversion Complete
          </h2>

          {/* Audio player */}
          <div className="audio-player">
            <audio ref={audioRef} src={downloadUrl} onEnded={() => setPlaying(false)} />
            <button className={`play-btn ${playing ? 'playing' : ''}`} onClick={togglePlay}>
              {playing ? (
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M6 4h4v16H6V4zm8 0h4v16h-4V4z"/>
                </svg>
              ) : (
                <svg viewBox="0 0 24 24" fill="currentColor">
                  <path d="M8 5v14l11-7z"/>
                </svg>
              )}
            </button>
            <div className="audio-info">
              <span>Output audio ready</span>
              <span className="audio-sub">Click play to preview</span>
            </div>
            <a className="download-btn" href={downloadUrl} download={`voicebridge_${jobId}.wav`}>
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5M16.5 12L12 16.5m0 0L7.5 12m4.5 4.5V3" />
              </svg>
              Download WAV
            </a>
          </div>
        </div>
      )}

      {/* Error */}
      {status === 'error' && (
        <div className="card error-card">
          <h3>⚠ Pipeline Error</h3>
          <pre className="error-text">{error}</pre>
        </div>
      )}

      {/* Live logs */}
      <div className="card logs-card">
        <h2 className="card-title">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="card-icon">
            <path strokeLinecap="round" strokeLinejoin="round" d="M3.75 12h16.5m-16.5 3.75h16.5M3.75 19.5h16.5M5.625 4.5h12.75a1.875 1.875 0 010 3.75H5.625a1.875 1.875 0 010-3.75z" />
          </svg>
          Pipeline Log
          <span className="log-count">{logs.length} lines</span>
        </h2>
        <div className="logs-body">
          {logs.length === 0 && status === 'queued' && (
            <div className="logs-empty">Waiting for pipeline to start…</div>
          )}
          {logs.map((log, i) => (
            <div key={i} className={`log-line log-${log.level?.toLowerCase()}`}>
              <span className="log-time">{log.time}</span>
              <span className="log-msg">{log.msg}</span>
            </div>
          ))}
          <div ref={logsEndRef} />
        </div>
      </div>
    </div>
  )
}
