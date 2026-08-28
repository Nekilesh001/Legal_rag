import React from 'react';
import { Citation } from '../../types/rag';

interface MarkdownRendererProps {
  content: string;
  citations?: Citation[];
  onOpenCitation?: (citation: Citation, allCitations: Citation[]) => void;
}

export const MarkdownRenderer: React.FC<MarkdownRendererProps> = ({
  content,
  citations = [],
  onOpenCitation,
}) => {
  if (!content) return null;

  // Helper to render inline formatting (bold, italic, citations)
  const renderInline = (text: string): React.ReactNode[] => {
    // Regex matches **bold**, *italic*, and [C01] / [1] style citations
    const regex = /(\*\*.*?\*\*|\*.*?\*|\[C?\d+\])/g;
    const parts = text.split(regex);

    return parts.map((part, i) => {
      if (part.startsWith('**') && part.endsWith('**')) {
        return (
          <strong key={i} className="font-semibold text-white">
            {part.slice(2, -2)}
          </strong>
        );
      }
      if (part.startsWith('*') && part.endsWith('*') && !part.startsWith('**')) {
        return (
          <em key={i} className="italic text-slate-300">
            {part.slice(1, -1)}
          </em>
        );
      }
      // Citation badge match (e.g., [C01] or [1])
      if (/^\[C?\d+\]$/.test(part)) {
        const numMatch = part.match(/\d+/);
        const idx = numMatch ? parseInt(numMatch[0], 10) - 1 : -1;
        const matchingCitation = citations[idx] || citations[0];

        return (
          <button
            key={i}
            onClick={() => {
              if (matchingCitation && onOpenCitation) {
                onOpenCitation(matchingCitation, citations);
              }
            }}
            className="inline-flex items-center gap-0.5 px-1.5 py-0.5 mx-0.5 text-[11px] font-mono font-bold rounded bg-[#5227FF]/30 hover:bg-[#5227FF]/60 text-[#FF9FFC] border border-[#5227FF]/50 transition-colors cursor-pointer"
            title="Click to view legal source citation"
          >
            {part}
          </button>
        );
      }
      return part;
    });
  };

  // Split lines into blocks (Tables, Headers, Lists, Paragraphs)
  const lines = content.split('\n');
  const blocks: React.ReactNode[] = [];

  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    const trimmed = line.trim();

    // 1. Skip empty lines
    if (!trimmed) {
      i++;
      continue;
    }

    // 2. Table Parsing (lines starting and containing '|')
    if (trimmed.startsWith('|') && trimmed.includes('|')) {
      const tableLines: string[] = [];
      while (i < lines.length && lines[i].trim().startsWith('|')) {
        tableLines.push(lines[i].trim());
        i++;
      }

      if (tableLines.length >= 2) {
        const parseRow = (rowStr: string) =>
          rowStr
            .split('|')
            .slice(1, -1)
            .map((cell) => cell.trim());

        const headerRow = parseRow(tableLines[0]);
        // Second line is separator line like |---|---|---|
        const bodyRows = tableLines.slice(2).map(parseRow);

        blocks.push(
          <div key={`table-${i}`} className="my-4 overflow-x-auto rounded-xl border border-white/10 bg-white/[0.02] shadow-lg">
            <table className="w-full text-left text-xs border-collapse">
              <thead>
                <tr className="bg-[#5227FF]/20 border-b border-white/10 text-[#FF9FFC]">
                  {headerRow.map((cell, hIdx) => (
                    <th key={hIdx} className="p-3 font-semibold tracking-wider">
                      {renderInline(cell)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody className="divide-y divide-white/5 text-slate-300">
                {bodyRows.map((row, rIdx) => (
                  <tr
                    key={rIdx}
                    className="hover:bg-white/5 transition-colors odd:bg-white/[0.01]"
                  >
                    {row.map((cell, cIdx) => (
                      <th key={cIdx} className="p-3 align-top font-normal leading-relaxed">
                        {renderInline(cell)}
                      </th>
                    ))}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        );
        continue;
      }
    }

    // 3. Headers (#, ##, ###)
    if (trimmed.startsWith('#')) {
      const level = trimmed.match(/^#+/)?.[0].length || 1;
      const text = trimmed.replace(/^#+\s*/, '');
      const headerClass =
        level === 1
          ? 'text-lg font-bold text-white mt-4 mb-2 border-b border-white/10 pb-1'
          : level === 2
          ? 'text-base font-bold text-[#FF9FFC] mt-3 mb-1.5'
          : 'text-sm font-bold text-indigo-300 mt-2.5 mb-1';

      blocks.push(
        <div key={`header-${i}`} className={headerClass}>
          {renderInline(text)}
        </div>
      );
      i++;
      continue;
    }

    // 4. Bullet & Numbered Lists
    if (/^[\*\-\+]\s+/.test(trimmed) || /^\d+\.\s+/.test(trimmed)) {
      const listItems: string[] = [];
      const isNumbered = /^\d+\.\s+/.test(trimmed);

      while (
        i < lines.length &&
        (/^[\*\-\+]\s+/.test(lines[i].trim()) || /^\d+\.\s+/.test(lines[i].trim()))
      ) {
        const itemText = lines[i].trim().replace(/^([\*\-\+]|\d+\.)\s+/, '');
        listItems.push(itemText);
        i++;
      }

      if (isNumbered) {
        blocks.push(
          <ol key={`ol-${i}`} className="list-decimal pl-5 my-2 space-y-1 text-slate-200">
            {listItems.map((item, lIdx) => (
              <li key={lIdx} className="leading-relaxed">
                {renderInline(item)}
              </li>
            ))}
          </ol>
        );
      } else {
        blocks.push(
          <ul key={`ul-${i}`} className="list-disc pl-5 my-2 space-y-1 text-slate-200">
            {listItems.map((item, lIdx) => (
              <li key={lIdx} className="leading-relaxed">
                {renderInline(item)}
              </li>
            ))}
          </ul>
        );
      }
      continue;
    }

    // 5. Regular Paragraph
    blocks.push(
      <p key={`p-${i}`} className="text-sm text-slate-200 leading-relaxed font-sans my-1.5">
        {renderInline(trimmed)}
      </p>
    );
    i++;
  }

  return <div className="space-y-1.5">{blocks}</div>;
};

export default MarkdownRenderer;
