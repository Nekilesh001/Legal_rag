import React, { useState } from 'react';
import { Search, Filter, ArrowUpDown, FileText, Download, Eye, Plus, ShieldAlert } from 'lucide-react';
import GlassCard from '../../components/common/GlassCard';
import Button from '../../components/common/Button';
import Badge from '../../components/common/Badge';
import { MOCK_CONTRACTS } from '../../data/mock/contracts';

export const ContractsPage: React.FC = () => {
  const [searchTerm, setSearchTerm] = useState('');
  const [typeFilter, setTypeFilter] = useState('All');
  const [statusFilter, setStatusFilter] = useState('All');

  const filteredContracts = MOCK_CONTRACTS.filter((c) => {
    const matchesSearch =
      c.title.toLowerCase().includes(searchTerm.toLowerCase()) ||
      c.counterparty.toLowerCase().includes(searchTerm.toLowerCase());
    const matchesType = typeFilter === 'All' || c.type === typeFilter;
    const matchesStatus = statusFilter === 'All' || c.status === statusFilter;
    return matchesSearch && matchesType && matchesStatus;
  });

  return (
    <div className="space-y-6 max-w-7xl mx-auto pb-8">
      {/* Header & Upload Banner */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
        <div>
          <h2 className="text-xl font-bold text-white tracking-tight">Contract Repository</h2>
          <p className="text-xs text-slate-400">
            Centralized document management & statutory metadata index
          </p>
        </div>

        <Button variant="accent" leftIcon={<Plus className="w-4 h-4" />}>
          Upload New Contract (Template)
        </Button>
      </div>

      {/* Filter & Search Toolbar */}
      <GlassCard className="p-4 flex flex-col md:flex-row items-center justify-between gap-4">
        <div className="relative w-full md:w-80">
          <Search className="w-4 h-4 text-slate-400 absolute left-3 top-2.5" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by contract title or counterparty..."
            className="w-full bg-white/5 border border-white/10 rounded-xl pl-9 pr-4 py-2 text-xs text-white placeholder-slate-500 focus:outline-none focus:border-[#5227FF]"
          />
        </div>

        <div className="flex items-center gap-3 w-full md:w-auto overflow-x-auto">
          <div className="flex items-center gap-2">
            <Filter className="w-3.5 h-3.5 text-slate-400" />
            <span className="text-xs text-slate-400 font-mono">Type:</span>
            <select
              value={typeFilter}
              onChange={(e) => setTypeFilter(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-[#5227FF]"
            >
              <option value="All">All Types</option>
              <option value="NDA">NDA</option>
              <option value="Employment">Employment</option>
              <option value="Vendor Services">Vendor Services</option>
              <option value="Lease">Lease</option>
              <option value="Master Services">Master Services</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 font-mono">Status:</span>
            <select
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
              className="bg-white/5 border border-white/10 rounded-xl px-3 py-1.5 text-xs text-slate-200 focus:outline-none focus:border-[#5227FF]"
            >
              <option value="All">All Statuses</option>
              <option value="Active">Active</option>
              <option value="Under Review">Under Review</option>
              <option value="Draft">Draft</option>
            </select>
          </div>
        </div>
      </GlassCard>

      {/* Contracts Table */}
      <GlassCard className="p-0 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-white/[0.03] text-slate-400 font-mono uppercase text-[10px] border-b border-white/10">
              <tr>
                <th className="p-4">Contract Title</th>
                <th className="p-4">Type</th>
                <th className="p-4">Counterparty</th>
                <th className="p-4">Status</th>
                <th className="p-4">Risk Level</th>
                <th className="p-4">Governing Law</th>
                <th className="p-4 text-right">Actions</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {filteredContracts.length === 0 ? (
                <tr>
                  <td colSpan={7} className="p-8 text-center text-slate-400">
                    No contracts match your search parameters.
                  </td>
                </tr>
              ) : (
                filteredContracts.map((c) => (
                  <tr key={c.id} className="hover:bg-white/[0.02] transition-colors">
                    <td className="p-4 font-medium text-white max-w-[220px] truncate">
                      <div className="flex items-center gap-2">
                        <FileText className="w-4 h-4 text-[#5227FF]" />
                        <span className="truncate">{c.title}</span>
                      </div>
                    </td>
                    <td className="p-4 text-slate-300">{c.type}</td>
                    <td className="p-4 text-slate-400 max-w-[180px] truncate">{c.counterparty}</td>
                    <td className="p-4">
                      <span className="px-2 py-0.5 rounded-full text-[10px] font-mono bg-white/5 text-slate-300 border border-white/10">
                        {c.status}
                      </span>
                    </td>
                    <td className="p-4">
                      <Badge variant={c.riskLevel === 'High' ? 'insufficient' : c.riskLevel === 'Medium' ? 'medium' : 'supported'}>
                        {c.riskLevel} ({c.riskScore})
                      </Badge>
                    </td>
                    <td className="p-4 text-slate-400 font-mono text-[11px] max-w-[160px] truncate">
                      {c.governingLaw}
                    </td>
                    <td className="p-4 text-right">
                      <div className="flex items-center justify-end gap-2">
                        <button className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors">
                          <Eye className="w-3.5 h-3.5" />
                        </button>
                        <button className="p-1.5 text-slate-400 hover:text-white rounded-lg hover:bg-white/5 transition-colors">
                          <Download className="w-3.5 h-3.5" />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))
              )}
            </tbody>
          </table>
        </div>

        {/* Pagination Footer */}
        <div className="p-4 border-t border-white/10 bg-white/[0.01] flex items-center justify-between text-xs text-slate-400 font-mono">
          <span>Showing {filteredContracts.length} of {MOCK_CONTRACTS.length} contracts</span>
          <div className="flex items-center gap-2">
            <button className="px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-50" disabled>
              Previous
            </button>
            <button className="px-3 py-1 rounded-lg bg-white/5 hover:bg-white/10 disabled:opacity-50" disabled>
              Next
            </button>
          </div>
        </div>
      </GlassCard>
    </div>
  );
};

export default ContractsPage;
