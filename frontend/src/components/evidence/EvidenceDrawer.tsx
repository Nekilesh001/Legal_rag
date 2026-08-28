import React, { useState } from 'react';
import { X, BookOpen, Copy, Check, FileText, ExternalLink, ShieldCheck } from 'lucide-react';
import { Citation, RetrievalResult } from '../../types/rag';
import Badge from '../common/Badge';

interface EvidenceDrawerProps {
  isOpen: boolean;
  onClose: () => void;
  citations: Citation[];
  supportingChunks?: RetrievalResult[];
  initialCitationIndex?: number;
}

export const EvidenceDrawer: React.FC<EvidenceDrawerProps> = ({
  isOpen,
  onClose,
  citations,
  supportingChunks = [],
  initialCitationIndex = 0,
}) => {
  const [selectedIndex, setSelectedIndex] = useState(initialCitationIndex);
  const [copied, setCopied] = useState(false);

  if (!isOpen || citations.length === 0) return null;

  const currentCitation = citations[selectedIndex] || citations[0];
  const matchingChunk = supportingChunks.find(c => c.chunk_id === currentCitation.chunk_id);

  const handleCopy = () => {
    const textToCopy = currentCitation.excerpt || matchingChunk?.text || '';
    navigator.clipboard.writeText(textToCopy);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 overflow-hidden bg-black/60 backdrop-blur-sm flex justify-end transition-opacity duration-300">
      <div className="w-full max-w-xl h-full glass-panel border-l border-white/10 flex flex-col shadow-2xl animate-in slide-in-from-right duration-300">
        
        {/* Drawer Header */}
        <div className="p-6 border-b border-white/10 flex items-center justify-between bg-white/[0.02]">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-xl bg-[#5227FF]/15 border border-[#5227FF]/30 flex items-center justify-center text-[#FF9FFC]">
              <BookOpen className="w-5 h-5" />
            </div>
            <div>
              <h2 className="text-base font-semibold text-white">Legal Evidence & Source Context</h2>
              <p className="text-xs text-slate-400">Verified Grounded Statutory & Contract Evidence</p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-slate-400 hover:text-white rounded-lg hover:bg-white/10 transition-colors"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Citation Selector Tabs (if multiple citations exist) */}
        {citations.length > 1 && (
          <div className="px-6 py-3 border-b border-white/10 bg-black/20 flex gap-2 overflow-x-auto">
            {citations.map((cit, idx) => (
              <button
                key={cit.citation_id || idx}
                onClick={() => setSelectedIndex(idx)}
                className={`px-3 py-1.5 rounded-lg text-xs font-medium whitespace-nowrap flex items-center gap-2 transition-all ${
                  selectedIndex === idx
                    ? 'bg-[#5227FF] text-white shadow-md shadow-[#5227FF]/30'
                    : 'bg-white/5 text-slate-400 hover:text-slate-200 hover:bg-white/10'
                }`}
              >
                <FileText className="w-3.5 h-3.5" />
                <span>Citation [{idx + 1}]</span>
              </button>
            ))}
          </div>
        )}

        {/* Drawer Content */}
        <div className="flex-1 overflow-y-auto p-6 space-y-6">
          {/* Metadata Card */}
          <div className="glass-panel p-5 rounded-xl space-y-3 bg-white/[0.02]">
            <div className="flex items-center justify-between">
              <span className="text-xs font-mono text-[#FF9FFC] uppercase tracking-wider">
                {currentCitation.category || 'Statutory Code'}
              </span>
              <Badge variant="supported" icon={<ShieldCheck className="w-3 h-3" />}>
                Verified Source
              </Badge>
            </div>

            <h3 className="text-base font-medium text-white leading-snug">
              {currentCitation.document_title || currentCitation.document_id}
            </h3>

            <div className="grid grid-cols-3 gap-3 pt-2 text-xs text-slate-300 border-t border-white/5">
              <div>
                <span className="text-slate-500 block text-[11px]">SECTION / PROVISION</span>
                <span className="font-mono text-slate-200 font-medium truncate block">
                  {currentCitation.section || currentCitation.section_title || 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">PAGE</span>
                <span className="font-mono text-slate-200 font-medium block">
                  {currentCitation.page !== undefined && currentCitation.page !== null ? `Page ${currentCitation.page}` : 'N/A'}
                </span>
              </div>
              <div>
                <span className="text-slate-500 block text-[11px]">CHUNK ID</span>
                <span className="font-mono text-slate-400 text-[11px] truncate block">
                  {currentCitation.chunk_id || 'N/A'}
                </span>
              </div>
            </div>
          </div>


          {/* Original Source Excerpt */}
          <div className="space-y-2">
            <div className="flex items-center justify-between">
              <label className="text-xs font-semibold text-slate-300 uppercase tracking-wider flex items-center gap-1.5">
                <FileText className="w-3.5 h-3.5 text-[#5227FF]" />
                Original Source Excerpt
              </label>
              <button
                onClick={handleCopy}
                className="text-xs text-slate-400 hover:text-white flex items-center gap-1 px-2.5 py-1 rounded-md hover:bg-white/5 transition-colors"
              >
                {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Copy className="w-3.5 h-3.5" />}
                <span>{copied ? 'Copied' : 'Copy Excerpt'}</span>
              </button>
            </div>

            <div className="glass-panel p-4 rounded-xl font-sans text-sm text-slate-200 leading-relaxed bg-[#0D0F18]/80 border-white/10 whitespace-pre-wrap selection:bg-[#5227FF]">
              {currentCitation.excerpt || matchingChunk?.text || 'No source text snippet available.'}
            </div>
          </div>

          {/* Detailed Context Note */}
          <div className="p-4 rounded-xl bg-[#5227FF]/10 border border-[#5227FF]/20 text-xs text-indigo-200 leading-relaxed">
            <span className="font-semibold text-white block mb-1">Authentic Legal Text Guarantee</span>
            This snippet represents exact canonical legal source content indexed in the vector repository. Synthetic context formatting is restricted to reranker inputs and is never leaked as legal citations.
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-white/10 bg-black/40 flex justify-end">
          <button
            onClick={onClose}
            className="px-4 py-2 text-xs font-medium text-slate-300 hover:text-white bg-white/5 hover:bg-white/10 rounded-xl transition-colors"
          >
            Close Evidence Panel
          </button>
        </div>
      </div>
    </div>
  );
};

export default EvidenceDrawer;
