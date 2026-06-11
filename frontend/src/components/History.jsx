const STATUS_ICONS = {
  queued:  '⏳',
  running: '⚡',
  done:    '✓',
  error:   '✕',
}

export default function History({ jobs, activeJobId, onSelect }) {
  return (
    <div className="card history-card">
      <h2 className="card-title">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" className="card-icon">
          <path strokeLinecap="round" strokeLinejoin="round"
            d="M12 6v6h4.5m4.5 0a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
        Recent Jobs
      </h2>
      <div className="history-list">
        {jobs.map(job => (
          <button
            key={job.id}
            className={`history-item ${activeJobId === job.id ? 'active' : ''} item-${job.status}`}
            onClick={() => onSelect(job.id)}
          >
            <span className="h-icon">{STATUS_ICONS[job.status]}</span>
            <div className="h-info">
              <span className="h-filename" title={job.filename}>{job.filename}</span>
              <span className="h-meta">{job.accent?.replace('_', ' → ')} · {job.status}</span>
            </div>
            {job.status === 'running' && <span className="h-pulse" />}
          </button>
        ))}
      </div>
    </div>
  )
}
