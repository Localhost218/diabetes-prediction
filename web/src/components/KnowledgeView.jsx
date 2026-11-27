import { useEffect, useState } from 'react'
import { Database, FileText, Network, Search, ExternalLink, LoaderCircle } from 'lucide-react'
import { KnowledgeGraphView } from './ThinkingPanel'
import { getJSON } from '../services/llmService'


// Tab "Tri thức KB": nguồn đã nạp (PDF + link web), số đoạn/nút/cạnh, đồ thị Neo4j và tìm kiếm hybrid (vector + full-text).
function KnowledgeView() {
  const [query, setQuery] = useState('')
  const [results, setResults] = useState(null)
  const [searching, setSearching] = useState(false)
  const [searchError, setSearchError] = useState('')
  const search = async (q) => {
    if (!q.trim()) return
    setSearching(true); setSearchError('')
    try {
      const r = await getJSON('/api/kb/search', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: q, k: 5 }) })
      setResults(r.passages)
    } catch (e) {
      setResults(null); setSearchError(e.message)
    } finally {
      setSearching(false)
    }
  }
  const [stats, setStats] = useState(null)
  const [graph, setGraph] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    getJSON('/api/kb/stats').then(setStats).catch(e => setError(e.message))
    getJSON('/api/kb/graph?limit=60').then(setGraph).catch(() => setGraph({ edges: [], unavailable: true }))
  }, [])

  const tiles = [
    { label: 'Tài liệu', value: stats?.documents.length, icon: FileText },
    { label: 'Đoạn (chunk)', value: stats?.chunks, icon: Database },
    { label: 'Nút đồ thị', value: stats?.graph.nodes, icon: Network },
    { label: 'Cạnh đồ thị', value: stats?.graph.edges, icon: Network },
  ]

  return (
    <div className="nb-page nb-kb">
      <p className="nb-kb-intro">PDF QĐ 5481/QĐ-BYT + 3 trang web (Vinmec, Long Châu, ADA 2024) → chia đoạn → embedding → Qdrant;
        bộ ba tri thức → Neo4j. KnowledgeAgent truy vấn cả hai (GraphRAG).</p>
      <div className="nb-tiles">
        {tiles.map(({ label, value, icon: Icon }) => (
          <div key={label} className="nb-tile"><span className="nb-tag"><Icon size={13} aria-hidden="true" /> {label}</span><strong>{value ?? '–'}</strong></div>
        ))}
      </div>
      {error && <p className="prediction-error" role="alert">Không tải được dữ liệu KB: {error}</p>}

      <section className="rail-card nb-kb-graph">
        <h2><Network size={18} aria-hidden="true" /> Đồ thị tri thức <span className="nb-count">{stats?.graph_backend || 'neo4j'}</span></h2>
        {graph?.edges?.length > 0 ? <KnowledgeGraphView graph={graph} width={1100} height={520} title="Vùng lân cận 2 bước của nút “Đái tháo đường típ 2”" />
          : <p className="rail-muted">{graph?.unavailable ? 'Backend chưa có endpoint /api/kb/graph (cần khởi động lại backend).' : graph ? 'Đồ thị trống: chạy python scripts/ingest_kb.py.' : 'Đang tải…'}</p>}
      </section>

      <section className="rail-card">
        <h2><FileText size={18} aria-hidden="true" /> Nguồn đã nạp <span className="nb-count">{stats?.documents.length ?? '…'} tài liệu</span></h2>
        {stats && stats.documents.length === 0 && <p className="rail-muted">Chưa có tài liệu nào trong KB của máy chủ này. Chạy <code>python scripts/ingest_kb.py</code> để nạp PDF và các link.</p>}
        <ul className="nb-source-grid">
          {stats?.documents.map(d => {
            const pdf = /\.pdf$/i.test(d.object_key || '') || /pdf/i.test(d.title)
            return (
              <li key={d.id} className="nb-source-card">
                <div className="nb-source-top">
                  <span className={`nb-source-kind${pdf ? ' is-pdf' : ''}`}>{pdf ? 'PDF' : 'WEB'}</span>
                  <span className="nb-pill">{d.chunks} đoạn</span>
                </div>
                <strong title={d.title}>{d.title}</strong>
                <span className="nb-source-from" title={d.source}>{d.source}</span>
                {d.url && <a href={d.url} target="_blank" rel="noreferrer" className="nb-source-link">Mở nguồn <ExternalLink size={13} aria-hidden="true" /></a>}
              </li>
            )
          })}
        </ul>
      </section>

      <section className="rail-card">
        <h2><Search size={17} aria-hidden="true" /> Tìm trong KB (hybrid: vector + full-text, RRF)</h2>
        <form className="nb-search-form" onSubmit={e => { e.preventDefault(); search(query) }}>
          <input value={query} onChange={e => setQuery(e.target.value)} placeholder="VD: HbA1c bao nhiêu là tiền đái tháo đường?" aria-label="Câu tìm kiếm trong KB" />
          <button className="btn-primary" disabled={searching || !query.trim()}>{searching ? <LoaderCircle size={15} className="spin" /> : <Search size={15} />} Tìm</button>
        </form>
        {searchError && <p className="prediction-error" role="alert">Tìm kiếm thất bại: {searchError}</p>}
        {results && results.length === 0 && <p className="rail-muted">Không có đoạn nào khớp. KB của máy chủ có thể chưa nạp đoạn văn nào.</p>}
        <ol className="nb-passages">
          {results?.map(p => (
            <li key={p.chunk_id}>
              <div className="nb-passage-head"><strong>{p.title}</strong>{p.heading && <span> · {p.heading}</span>}<span className="nb-pill">score {p.score}</span></div>
              <p>{p.content.length > 420 ? p.content.slice(0, 420) + '…' : p.content}</p>
              {p.url && <a href={p.url} target="_blank" rel="noreferrer">{p.source}</a>}
            </li>
          ))}
        </ol>
      </section>
    </div>
  )
}

export default KnowledgeView
