import { useState, useRef, useCallback } from 'react'

const ACCENTS = [
  { value: 'indian_american', label: 'Indian → American English' },
  { value: 'indian_british',  label: 'Indian → British English' },
]
const WHISPER_MODELS = [
  { value: 'base',   label: 'Base (faster)' },
  { value: 'medium', label: 'Medium (accurate)' },
]

const API = ''

export default function Uploader({ onJobStarted }) {
  const [dragging, setDragging] = useState(false)
  const [file, setFile]         = useState(null)
  const [accent, setAccent]     = useState('indian_american')
  const [model, setModel]       = useState('base')
  const [loading, setLoading]   = useState(false)
  const [error, setError]       = useState(null)
  const inputRef = useRef()

  const handleFile = (f) => {
    if (!f) return
    const ok = f.type.startsWith('audio/') || /\.(wav|mp3|m4a|flac|ogg)$/i.test(f.name)
    if (!ok) { setError('Please upload an audio file (WAV, MP3, M4A, FLAC)'); return }
    setError(null)
    setFile(f)
  }

  const onDrop = useCallback((e) => {
    e.preventDefault()
    setDragging(false)
    handleFile(e.dataTransfer.files[0])
  }, [])

  const submit = async () => {
    if (!file) return
    setLoading(true)
    setError(null)
    try {
      const fd = new FormData()
      fd.append('audio', file)
      fd.append('accent', accent)
      fd.append('whisper_model', model)

      const res = await fetch(`${API}/api/process`, { method: 'POST', body: fd })
      if (!res.ok) throw new Error(`Server error ${res.status}`)
      const data = await res.json()
      onJobStarted(data.job_id, file.name, accent)
      setFile(null)
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card uploader-card">
      <h2 className="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="card-icon">
          <path strokeLinecap="round" strokeLinejoin="round" d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5" />
        </svg>
        Upload Audio
      </h2>

      {/* Drop zone */}
      <div
        className={`drop-zone ${dragging ? 'dragging' : ''} ${file ? 'has-file' : ''}`}
        onDragOver={(e) => { e.preventDefault(); setDragging(true) }}
        onDragLeave={() => setDragging(false)}
        onDrop={onDrop}
        onClick={() => !file && inputRef.current.click()}
      >
        <input ref={inputRef} type="file" accept="audio/*" hidden
          onChange={e => handleFile(e.target.files[0])} />

        {file ? (
          <div className="file-selected">
            <div className="file-icon">🎵</div>
            <div className="file-info">
              <div className="file-name">{file.name}</div>
              <div className="file-size">{(file.size / 1024 / 1024).toFixed(2)} MB</div>
            </div>
            <button className="file-clear" onClick={(e) => { e.stopPropagation(); setFile(null) }}>✕</button>
          </div>
        ) : (
          <div className="drop-hint">
            <div className="drop-icon">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                <path strokeLinecap="round" strokeLinejoin="round"
                  d="M12 18.75a6 6 0 006-6v-1.5m-6 7.5a6 6 0 01-6-6v-1.5m6 7.5v3.75m-3.75 0h7.5M12 15.75a3 3 0 01-3-3V4.5a3 3 0 116 0v8.25a3 3 0 01-3 3z" />
              </svg>
            </div>
            <p className="drop-primary">Drop audio file here</p>
            <p className="drop-secondary">or click to browse · WAV, MP3, M4A, FLAC</p>
          </div>
        )}
      </div>

      {/* Options */}
      <div className="options-grid">
        <div className="option-group">
          <label className="option-label">Target Accent</label>
          <select className="option-select" value={accent} onChange={e => setAccent(e.target.value)}>
            {ACCENTS.map(a => <option key={a.value} value={a.value}>{a.label}</option>)}
          </select>
        </div>
        <div className="option-group">
          <label className="option-label">Whisper Model</label>
          <select className="option-select" value={model} onChange={e => setModel(e.target.value)}>
            {WHISPER_MODELS.map(m => <option key={m.value} value={m.value}>{m.label}</option>)}
          </select>
        </div>
      </div>

      {error && <div className="error-banner">⚠ {error}</div>}

      <button
        className={`convert-btn ${loading ? 'loading' : ''}`}
        disabled={!file || loading}
        onClick={submit}
      >
        {loading ? (
          <><span className="spinner" /> Uploading…</>
        ) : (
          <><span className="btn-icon">⚡</span> Convert Voice</>
        )}
      </button>

      <p className="time-hint">Pipeline takes 2–3 minutes on CPU · progress shown in real time</p>
    </div>
  )
}
