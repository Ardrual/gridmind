import { useState } from 'react'

interface Citation {
  source_id: string
  title: string
  url: string
  page: number
  snippet: string
}

interface Answer {
  answer: string
  citations: Citation[]
  latency_ms: number
}

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

export default function QueryForm() {
  const [query, setQuery] = useState('')
  const [result, setResult] = useState<Answer | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault()
    if (!query.trim()) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const resp = await fetch(`${API_BASE_URL}/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query }),
      })
      if (!resp.ok) {
        throw new Error(`Request failed: ${resp.status}`)
      }
      const data = (await resp.json()) as Answer
      setResult(data)
    } catch (err) {
      setError(err instanceof Error ? err.message : String(err))
    } finally {
      setLoading(false)
    }
  }

  return (
    <div>
      <form onSubmit={handleSubmit} className="flex justify-center gap-2">
        <input
          className="flex-1 rounded border px-3 py-2"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          placeholder="Ask a question"
        />
        <button
          type="submit"
          disabled={loading}
          className="rounded bg-gray-800 px-4 py-2 text-white disabled:opacity-50"
        >
          {loading ? 'Loading…' : 'Ask'}
        </button>
      </form>
      {error && <p className="mt-2 text-red-500">{error}</p>}
      {result && (
        <div className="mt-4 text-left">
          <p>{result.answer}</p>
          {result.citations.length > 0 && (
            <ul className="mt-2 space-y-2">
              {result.citations.map((c) => (
                <li key={c.source_id}>
                  <a
                    href={c.url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="font-medium text-blue-600 hover:underline"
                  >
                    {c.title} (p. {c.page})
                  </a>
                  <p>{c.snippet}</p>
                </li>
              ))}
            </ul>
          )}
        </div>
      )}
    </div>
  )
}
