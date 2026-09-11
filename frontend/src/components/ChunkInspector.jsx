import { useState, useEffect, useRef } from 'react';
import {
  X, ChevronDown, ChevronUp, Layers, Search, Zap, FileText, Clock,
  AlertCircle, CheckCircle2, Database, Activity, Code2, Eye
} from 'lucide-react';

// ---------------------------------------------------------------------------
// Sub-components
// ---------------------------------------------------------------------------

function TimingBadge({ label, ms, color }) {
  return (
    <div className="flex flex-col items-center gap-1">
      <div className={`text-xs font-mono font-bold ${color}`}>{ms}ms</div>
      <div className="text-[10px] text-slate-500 uppercase tracking-wide">{label}</div>
    </div>
  );
}

function PipelineFlowBar({ timings }) {
  if (!timings) return null;
  const stages = [
    { label: 'Parse', ms: timings.parse_ms, color: 'text-violet-400' },
    { label: 'Dense', ms: timings.dense_ms, color: 'text-cyan-400' },
    { label: 'Sparse', ms: timings.sparse_ms, color: 'text-emerald-400' },
    { label: 'Fuse', ms: timings.fusion_ms, color: 'text-amber-400' },
    { label: 'Rerank', ms: timings.rerank_ms, color: 'text-orange-400' },
    { label: 'LLM', ms: timings.generation_ms, color: 'text-pink-400' },
  ];

  return (
    <div className="px-5 py-3 border-b border-white/5 bg-slate-900/60">
      <div className="flex items-center gap-1 mb-3">
        <Activity className="w-3.5 h-3.5 text-slate-400" />
        <span className="text-xs text-slate-400 font-medium uppercase tracking-wider">Pipeline Timings</span>
        <span className="ml-auto text-xs text-slate-500">
          Total: <span className="text-slate-300 font-mono font-bold">{timings.total_ms}ms</span>
        </span>
      </div>
      <div className="flex items-center gap-1">
        {stages.map((stage, idx) => (
          <div key={stage.label} className="flex items-center gap-1 flex-1">
            <div className="flex-1 bg-slate-800 rounded-lg p-2 text-center border border-white/5">
              <TimingBadge {...stage} />
            </div>
            {idx < stages.length - 1 && (
              <div className="text-slate-600 text-xs">→</div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}

function ScoreBadge({ label, value, color }) {
  if (value == null) return null;
  const display = typeof value === 'number' ? value.toFixed(4) : value;
  return (
    <span className={`inline-flex items-center gap-1 text-[10px] font-mono font-semibold px-1.5 py-0.5 rounded border ${color}`}>
      {label}: {display}
    </span>
  );
}

function MetaBadge({ children, color = 'bg-slate-800 text-slate-400 border-slate-700' }) {
  return (
    <span className={`inline-flex items-center text-[10px] px-1.5 py-0.5 rounded border ${color}`}>
      {children}
    </span>
  );
}

function ChunkCard({ chunk, rank, showRerank, showBM25, showVector, showRRF }) {
  const [expanded, setExpanded] = useState(false);
  const meta = chunk.metadata || {};
  const preview = chunk.text_preview || chunk.full_text?.slice(0, 200) || '';

  return (
    <div className="bg-slate-900/70 border border-white/5 rounded-xl overflow-hidden hover:border-white/10 transition-colors">
      {/* Header */}
      <div className="flex items-start gap-3 p-3">
        <div className="flex-shrink-0 w-6 h-6 rounded-full bg-slate-800 border border-white/10 flex items-center justify-center text-[10px] font-bold text-slate-400">
          {rank}
        </div>
        <div className="flex-1 min-w-0">
          <p className="text-xs font-medium text-slate-200 truncate" title={meta.subject}>
            {meta.subject || 'No Subject'}
          </p>
          <p className="text-[11px] text-slate-400 truncate" title={meta.sender}>
            {meta.sender || 'Unknown Sender'}
          </p>
          <div className="flex flex-wrap gap-1 mt-1.5">
            <MetaBadge color="bg-slate-800 text-slate-400 border-slate-700">
              {meta.date_iso || meta.timestamp?.slice(0, 10) || '—'}
            </MetaBadge>
            {meta.chunk_number != null && (
              <MetaBadge>chunk {meta.chunk_number + 1}/{meta.total_chunks}</MetaBadge>
            )}
            {showRerank && <ScoreBadge label="rerank" value={chunk.rerank_score} color="border-pink-800/60 bg-pink-950/40 text-pink-300" />}
            {showRRF && <ScoreBadge label="rrf" value={chunk.rrf_score} color="border-amber-800/60 bg-amber-950/40 text-amber-300" />}
            {showBM25 && <ScoreBadge label="bm25" value={chunk.bm25_score} color="border-emerald-800/60 bg-emerald-950/40 text-emerald-300" />}
            {showVector && chunk.vector_distance != null && (
              <ScoreBadge label="dist" value={chunk.vector_distance} color="border-cyan-800/60 bg-cyan-950/40 text-cyan-300" />
            )}
          </div>
        </div>
        <button
          onClick={() => setExpanded(!expanded)}
          className="flex-shrink-0 text-slate-500 hover:text-slate-300 transition-colors p-0.5"
        >
          {expanded ? <ChevronUp className="w-4 h-4" /> : <ChevronDown className="w-4 h-4" />}
        </button>
      </div>

      {/* Preview or full text */}
      <div
        className={`px-3 pb-3 transition-all duration-200 ${expanded ? '' : 'max-h-16 overflow-hidden'}`}
        style={{ cursor: 'pointer' }}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="bg-slate-950/80 rounded-lg p-2.5 border border-white/5">
          <p className="text-[11px] font-mono text-slate-400 leading-relaxed whitespace-pre-wrap break-words">
            {expanded ? (chunk.full_text || preview) : preview}
          </p>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Tab content components
// ---------------------------------------------------------------------------

function RerankedTab({ reranked }) {
  if (!reranked?.length) return (
    <EmptyState icon={<Layers className="w-8 h-8" />} message="No reranked chunks available." />
  );
  return (
    <div className="space-y-3 p-4">
      <p className="text-xs text-slate-500">Top {reranked.length} chunks passed to the LLM — sorted by cross-encoder relevance.</p>
      {reranked.map((chunk, idx) => (
        <ChunkCard key={chunk.id} chunk={chunk} rank={idx + 1} showRerank showRRF showVector showBM25 />
      ))}
    </div>
  );
}

function HybridTab({ dense, sparse, fused }) {
  const [view, setView] = useState('fused');
  const tabs = [
    { key: 'fused', label: `RRF Fused (${fused?.length || 0})`, color: 'text-amber-400' },
    { key: 'dense', label: `Dense / Chroma (${dense?.length || 0})`, color: 'text-cyan-400' },
    { key: 'sparse', label: `Sparse / BM25 (${sparse?.length || 0})`, color: 'text-emerald-400' },
  ];

  const chunks = view === 'fused' ? fused : view === 'dense' ? dense : sparse;

  return (
    <div className="flex flex-col h-full">
      {/* Sub-tabs */}
      <div className="flex gap-1 p-3 pb-0 border-b border-white/5">
        {tabs.map(t => (
          <button
            key={t.key}
            onClick={() => setView(t.key)}
            className={`text-[11px] px-2.5 py-1.5 rounded-lg font-medium transition-colors ${
              view === t.key
                ? `bg-slate-800 ${t.color} border border-white/10`
                : 'text-slate-500 hover:text-slate-300'
            }`}
          >
            {t.label}
          </button>
        ))}
      </div>
      <div className="overflow-y-auto flex-1 space-y-3 p-4">
        {!chunks?.length ? (
          <EmptyState icon={<Database className="w-8 h-8" />} message={`No ${view} results.`} />
        ) : chunks.map((chunk, idx) => (
          <ChunkCard
            key={chunk.id}
            chunk={chunk}
            rank={idx + 1}
            showRerank={false}
            showRRF={view === 'fused'}
            showBM25={view === 'sparse' || view === 'fused'}
            showVector={view === 'dense' || view === 'fused'}
          />
        ))}
      </div>
    </div>
  );
}

function QueryAnalysisTab({ queryAnalysis, timings }) {
  const filters = queryAnalysis?.metadata_filters || {};
  const hasFilters = Object.keys(filters).length > 0;

  return (
    <div className="p-4 space-y-4">
      {/* Semantic query */}
      <section>
        <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2 font-semibold">Parsed Query</h3>
        <div className="bg-slate-900/80 rounded-xl p-3 border border-white/5 space-y-2">
          <div>
            <span className="text-[10px] text-slate-500">Raw User Query</span>
            <p className="text-xs text-slate-200 font-mono mt-0.5">{queryAnalysis?.raw_query || '—'}</p>
          </div>
          <div>
            <span className="text-[10px] text-slate-500">Semantic Query (sent to vector/BM25)</span>
            <p className={`text-xs font-mono mt-0.5 ${queryAnalysis?.semantic_query ? 'text-violet-300' : 'text-slate-600 italic'}`}>
              {queryAnalysis?.semantic_query || 'null — metadata-only query'}
            </p>
          </div>
          <div className="flex items-center gap-2">
            <span className="text-[10px] text-slate-500">Routing:</span>
            {queryAnalysis?.is_metadata_only ? (
              <MetaBadge color="bg-amber-950/40 text-amber-300 border-amber-800/50">📋 Metadata-only path</MetaBadge>
            ) : (
              <MetaBadge color="bg-violet-950/40 text-violet-300 border-violet-800/50">🔍 Semantic + metadata hybrid</MetaBadge>
            )}
          </div>
        </div>
      </section>

      {/* Extracted filters */}
      <section>
        <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2 font-semibold">Extracted Metadata Filters</h3>
        <div className="bg-slate-900/80 rounded-xl p-3 border border-white/5">
          {!hasFilters ? (
            <p className="text-xs text-slate-600 italic">No metadata filters extracted.</p>
          ) : (
            <pre className="text-[11px] font-mono text-emerald-300 whitespace-pre-wrap break-all">
              {JSON.stringify(filters, null, 2)}
            </pre>
          )}
        </div>
      </section>

      {/* Timings */}
      {timings && (
        <section>
          <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2 font-semibold">Stage Timings</h3>
          <div className="bg-slate-900/80 rounded-xl p-3 border border-white/5 space-y-1.5">
            {Object.entries(timings).map(([key, val]) => (
              <div key={key} className="flex items-center justify-between">
                <span className="text-[11px] text-slate-400 font-mono">{key}</span>
                <span className="text-[11px] font-mono text-slate-200 font-bold">{val}ms</span>
              </div>
            ))}
          </div>
        </section>
      )}
    </div>
  );
}

function PromptPreviewTab({ promptContext }) {
  return (
    <div className="p-4">
      <h3 className="text-[10px] uppercase tracking-widest text-slate-500 mb-2 font-semibold">
        Context Injected into LLM Prompt
      </h3>
      {!promptContext ? (
        <EmptyState icon={<Code2 className="w-8 h-8" />} message="No prompt context available." />
      ) : (
        <div className="bg-slate-950/80 rounded-xl p-3 border border-white/5 overflow-y-auto max-h-[calc(100vh-280px)]">
          <pre className="text-[11px] font-mono text-slate-300 whitespace-pre-wrap leading-relaxed">
            {promptContext}
          </pre>
        </div>
      )}
    </div>
  );
}

function EmptyState({ icon, message }) {
  return (
    <div className="flex flex-col items-center justify-center py-12 text-slate-600">
      <div className="mb-3 opacity-40">{icon}</div>
      <p className="text-sm">{message}</p>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Main ChunkInspector component
// ---------------------------------------------------------------------------

const TABS = [
  { id: 'reranked', label: 'LLM Context', icon: <CheckCircle2 className="w-3.5 h-3.5" />, color: 'text-pink-400' },
  { id: 'hybrid', label: 'Hybrid View', icon: <Layers className="w-3.5 h-3.5" />, color: 'text-amber-400' },
  { id: 'analysis', label: 'Query Analysis', icon: <Search className="w-3.5 h-3.5" />, color: 'text-violet-400' },
  { id: 'prompt', label: 'Prompt Preview', icon: <Code2 className="w-3.5 h-3.5" />, color: 'text-cyan-400' },
];

export default function ChunkInspector({ trace, isOpen, onClose }) {
  const [activeTab, setActiveTab] = useState('reranked');

  useEffect(() => {
    if (isOpen) setActiveTab('reranked');
  }, [isOpen, trace]);

  if (!isOpen) return null;

  const stages = trace?.stages || {};
  const timings = trace?.timings;
  const queryAnalysis = trace?.query_analysis;
  const promptContext = trace?.prompt_context;

  return (
    <>
      {/* Overlay */}
      <div
        className="fixed inset-0 bg-black/40 backdrop-blur-sm z-40"
        onClick={onClose}
        aria-hidden="true"
      />

      {/* Drawer */}
      <div
        className="fixed right-0 top-0 h-full w-[520px] max-w-full z-50 flex flex-col"
        style={{
          background: 'linear-gradient(145deg, #0e1117 0%, #0a0d14 100%)',
          borderLeft: '1px solid rgba(255,255,255,0.06)',
          boxShadow: '-24px 0 80px rgba(0,0,0,0.6)',
        }}
      >
        {/* Header */}
        <div className="flex items-center gap-3 px-5 py-4 border-b border-white/5">
          <div className="w-8 h-8 rounded-lg bg-gradient-to-br from-violet-600 to-pink-600 flex items-center justify-center shadow-lg shadow-violet-900/30">
            <Eye className="w-4 h-4 text-white" />
          </div>
          <div>
            <h2 className="text-sm font-bold text-white">Chunk Inspector</h2>
            <p className="text-[10px] text-slate-500">Live RAG pipeline trace</p>
          </div>
          <button
            onClick={onClose}
            className="ml-auto text-slate-500 hover:text-slate-200 transition-colors p-1 rounded-lg hover:bg-slate-800"
            aria-label="Close inspector"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        {/* Pipeline timing bar */}
        <PipelineFlowBar timings={timings} />

        {/* Tabs */}
        <div className="flex gap-1 px-4 pt-3 pb-0 border-b border-white/5 overflow-x-auto">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => setActiveTab(tab.id)}
              className={`flex items-center gap-1.5 text-[11px] px-3 py-2 rounded-t-lg font-medium whitespace-nowrap transition-all ${
                activeTab === tab.id
                  ? `bg-slate-800/80 ${tab.color} border border-white/10 border-b-0 -mb-px`
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              {tab.icon}
              {tab.label}
            </button>
          ))}
        </div>

        {/* Tab content */}
        <div className="flex-1 overflow-y-auto">
          {activeTab === 'reranked' && <RerankedTab reranked={stages.reranked} />}
          {activeTab === 'hybrid' && (
            <HybridTab dense={stages.dense} sparse={stages.sparse} fused={stages.fused} />
          )}
          {activeTab === 'analysis' && (
            <QueryAnalysisTab queryAnalysis={queryAnalysis} timings={timings} />
          )}
          {activeTab === 'prompt' && <PromptPreviewTab promptContext={promptContext} />}
        </div>
      </div>
    </>
  );
}
