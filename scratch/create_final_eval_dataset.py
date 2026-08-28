"""
create_final_eval_dataset.py — Generates the 120-question Final Evaluation Dataset (eval_dataset_final.json).

Covers 16 required categories:
  A. Exact legal lookup
  B. Section/subsection lookup
  C. Definition query
  D. Threshold / parameter query
  E. Obligation query
  F. Breach / remedy query
  G. Termination query
  H. Confidentiality / NDA query
  I. IP query
  J. Lease query
  K. Vendor query
  L. Multi-document query
  M. Cross-reference query
  N. Ambiguous query
  O. Out-of-corpus query (abstention)
  P. Version / currentness query
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from legal_rag.indexing.bm25_store import BM25Store
from legal_rag.config import get_config
from legal_rag.retrieval.legal_identity import registry

cfg = get_config()
bm25 = BM25Store(cfg.bm25_dir)
bm25.load()
registry.bootstrap(bm25._chunk_metadata)

EVAL_DATASET = []

def add_q(qid, category, query, expected_cids, doc_ids=None, expected_abstention=False, desc=""):
    EVAL_DATASET.append({
        "query_id": f"Q_{qid:03d}",
        "query_type": category,
        "query": query,
        "expected_evidence_chunks": expected_cids,
        "expected_documents": doc_ids or [],
        "expected_abstention": expected_abstention,
        "description": desc,
    })

# ==================================================================== #
# CATEGORY A: Exact Legal Lookup (10 queries)
# ==================================================================== #
add_q(1, "exact_legal_lookup", "What does Section 73 of the Indian Contract Act say?", ["chk_6c2b46f4b321"], ["a187209"], False, "ICA Section 73 compensation")
add_q(2, "exact_legal_lookup", "What does Section 10 of the Indian Contract Act state regarding valid contracts?", [], ["a187209"], False, "ICA Section 10 valid contracts")
add_q(3, "exact_legal_lookup", "What is provided under Section 27 of the Indian Contract Act?", [], ["a187209"], False, "ICA Section 27 restraint of trade")
add_q(4, "exact_legal_lookup", "What does Section 56 of the Indian Contract Act cover regarding frustration?", [], ["a187209"], False, "ICA Section 56 frustration")
add_q(5, "exact_legal_lookup", "What does Section 54 of the Transfer of Property Act define?", [], ["a1882-04"], False, "TPA Section 54 sale of immovable property")
add_q(6, "exact_legal_lookup", "What is specified under Section 3 of the Limitation Act, 1963?", [], ["a1963-36"], False, "Limitation Act Sec 3 bar of limitation")
add_q(7, "exact_legal_lookup", "What does Section 73 of the Central Goods and Services Tax Act specify?", ["chk_b211f53d08cb"], ["a2017-12"], False, "CGST Act Sec 73 determination of tax")
add_q(8, "exact_legal_lookup", "What rights are conferred under Section 48 of the Patents Act, 1970?", [], ["a1970-39"], False, "Patents Act Sec 48 patentee rights")
add_q(9, "exact_legal_lookup", "What does Section 3 of the Competition Act, 2002 prohibit?", [], ["a2003-12"], False, "Competition Act Sec 3 anti-competitive agreements")
add_q(10, "exact_legal_lookup", "What does Section 41 of the Tamil Nadu Shops and Establishments Act state?", ["chk_56b1160532cc"], ["1637820824"], False, "TN Shops Act Sec 41 dismissal notice")

# ==================================================================== #
# CATEGORY B: Section / Subsection Lookup (10 queries)
# ==================================================================== #
add_q(11, "section_subsection_lookup", "What does Section 41(1) of the Tamil Nadu Shops Act provide regarding notice?", ["chk_56b1160532cc"], ["1637820824"], False, "TN Shops Act Sec 41(1)")
add_q(12, "section_subsection_lookup", "What does Section 59(1) of the Sale of Goods Act specify for breach of warranty?", ["chk_01051fb4680e"], ["193003"], False, "Sale of Goods Act Sec 59(1)")
add_q(13, "section_subsection_lookup", "What is stated in Section 54(2) of the Sale of Goods Act regarding unpaid seller resale?", ["chk_e6886e9c6c83"], ["193003"], False, "Sale of Goods Act Sec 54(2)")
add_q(14, "section_subsection_lookup", "What does Section 2(h) of the Indian Contract Act define?", [], ["a187209"], False, "ICA Sec 2(h) contract definition")
add_q(15, "section_subsection_lookup", "What does Section 2(d) of the Indian Contract Act define consideration as?", [], ["a187209"], False, "ICA Sec 2(d) consideration definition")
add_q(16, "section_subsection_lookup", "What does Section 73(1) of CGST Act state regarding tax determination?", ["chk_e63c3a9b924c"], ["a2017-12"], False, "CGST Act Sec 73(1)")
add_q(17, "section_subsection_lookup", "What does Section 105 of the Transfer of Property Act define?", [], ["a1882-04"], False, "TPA Sec 105 lease definition")
add_q(18, "section_subsection_lookup", "What does Section 108 of the Transfer of Property Act lay down for lessor rights?", [], ["a1882-04"], False, "TPA Sec 108 lessor rights")
add_q(19, "section_subsection_lookup", "What does Section 14(2) of the Tamil Nadu Shops Act specify for deductions?", ["chk_f9a2e38b25bf"], ["1637820824"], False, "TN Shops Act Sec 14(2)")
add_q(20, "section_subsection_lookup", "What does Section 60 of the Sale of Goods Act provide for anticipatory repudiation?", ["chk_72ff24737368"], ["193003"], False, "Sale of Goods Act Sec 60")

# ==================================================================== #
# CATEGORY C: Definition Query (10 queries)
# ==================================================================== #
add_q(21, "definition_query", "What is the statutory definition of a contract under Indian law?", [], ["a187209"], False, "Contract definition")
add_q(22, "definition_query", "How is 'goods' defined under the Sale of Goods Act, 1930?", [], ["193003"], False, "Goods definition SGA")
add_q(23, "definition_query", "What is the definition of 'person employed' under the Tamil Nadu Shops and Establishments Act?", [], ["1637820824"], False, "Person employed definition TN Shops")
add_q(24, "definition_query", "How is 'patent' defined under Section 2 of the Patents Act, 1970?", [], ["a1970-39"], False, "Patent definition")
add_q(25, "definition_query", "What is the definition of 'lease' under the Transfer of Property Act, 1882?", [], ["a1882-04"], False, "Lease definition TPA")
add_q(26, "definition_query", "How is 'unpaid seller' defined under the Sale of Goods Act?", [], ["193003"], False, "Unpaid seller definition")
add_q(27, "definition_query", "What constitutes a 'proposal' under the Indian Contract Act?", [], ["a187209"], False, "Proposal definition ICA")
add_q(28, "definition_query", "What is the statutory meaning of 'commercial establishment' under TN Shops Act?", [], ["1637820824"], False, "Commercial establishment definition")
add_q(29, "definition_query", "What is defined as an 'anti-competitive agreement' under the Competition Act?", [], ["a2003-12"], False, "Anti-competitive agreement definition")
add_q(30, "definition_query", "How is 'market value' defined in the CGST Act under Section 73 context?", ["chk_e36e398d8aaf"], ["a2017-12"], False, "Market value definition CGST")

# ==================================================================== #
# CATEGORY D: Threshold / Parameter Query (10 queries)
# ==================================================================== #
add_q(31, "threshold_parameter_query", "What is the mandatory notice period for employee discharge under Tamil Nadu Shops Act?", ["chk_56b1160532cc"], ["1637820824"], False, "TN Shops notice period")
add_q(32, "threshold_parameter_query", "What minimum continuous service period is required before Section 41 notice applies in TN Shops Act?", ["chk_56b1160532cc"], ["1637820824"], False, "TN Shops 6 months service threshold")
add_q(33, "threshold_parameter_query", "What is the maximum limit on wage deduction under Section 14 of Tamil Nadu Shops Act?", ["chk_f9a2e38b25bf"], ["1637820824"], False, "TN Shops deduction limit")
add_q(34, "threshold_parameter_query", "What is the statutory limitation period for filing suits for breach of contract?", [], ["a1963-36"], False, "Limitation Act breach period")
add_q(35, "threshold_parameter_query", "What is the maximum duration of confidentiality obligation in standard NDA playbook?", ["chk_001536fcc3f4"], ["Mandatory Clauses"], False, "NDA confidentiality duration threshold")
add_q(36, "threshold_parameter_query", "What is the liability cap specified in mandatory contract clauses?", ["chk_80671d091ba6"], ["Mandatory Clauses"], False, "Liability cap clause")
add_q(37, "threshold_parameter_query", "How many days notice is required before closing a shop under Section 11 of TN Shops Act?", ["chk_e35cbc33e494"], ["1637820824"], False, "TN Shops closing notice days")
add_q(38, "threshold_parameter_query", "What is the statutory period for patent protection under the Patents Act?", [], ["a1970-39"], False, "Patent term threshold")
add_q(39, "threshold_parameter_query", "What monetary fine applies for non-display of weekly holiday notice under TN Shops Rules?", [], ["Tamil Nadu Shops Rules"], False, "TN Shops penalty fine threshold")
add_q(40, "threshold_parameter_query", "What is the age threshold for employment of young persons under TN Shops Act?", [], ["1637820824"], False, "Young person age threshold")

# ==================================================================== #
# CATEGORY E: Obligation Query (10 queries)
# ==================================================================== #
add_q(41, "obligation_query", "What are the statutory obligations of a shop employer regarding weekly holidays?", ["chk_e0b28f8cd3e2"], ["1637820824"], False, "Employer weekly holiday obligations")
add_q(42, "obligation_query", "What obligations does a recipient of confidential information have under an NDA?", ["chk_28e9248468e6"], ["Mandatory Clauses"], False, "NDA confidentiality obligation")
add_q(43, "obligation_query", "What are the legal obligations of a seller regarding delivery of goods?", [], ["193003"], False, "Seller delivery obligations SGA")
add_q(44, "obligation_query", "What duties does a tenant/lessee have under Section 108 of Transfer of Property Act?", [], ["a1882-04"], False, "Lessee duties TPA")
add_q(45, "obligation_query", "What must an employer do before dispensing with employee services under TN Shops Act?", ["chk_56b1160532cc"], ["1637820824"], False, "Employer discharge obligations")
add_q(46, "obligation_query", "What obligations apply regarding employee non-solicitation in contract playbooks?", ["chk_a238e8a1af24"], ["Mandatory Clauses"], False, "Non-solicitation obligation")
add_q(47, "obligation_query", "What is a party's obligation to mitigate loss upon breach under Section 73 of ICA?", ["chk_6c2b46f4b321"], ["a187209"], False, "Mitigation obligation ICA 73")
add_q(48, "obligation_query", "What obligations exist for displaying shop closing hours under Section 11 of TN Shops Act?", ["chk_e35cbc33e494"], ["1637820824"], False, "Display notice obligation")
add_q(49, "obligation_query", "What obligations apply to a patentee regarding working of patents in India?", [], ["a1970-39"], False, "Patent working obligation")
add_q(50, "obligation_query", "What are the mandatory obligations in an NDA regarding IP rights transfer?", ["chk_9d1812168426"], ["Mandatory Clauses"], False, "IP transfer obligation NDA")

# ==================================================================== #
# CATEGORY F: Breach / Remedy Query (10 queries)
# ==================================================================== #
add_q(51, "breach_remedy_query", "What happens if the seller breaches the contract?", ["chk_01051fb4680e", "chk_e83012ad27b7", "chk_72ff24737368"], ["193003"], False, "Seller breach remedies SGA")
add_q(52, "breach_remedy_query", "What remedies are available to a buyer for breach of warranty under Section 59 of SGA?", ["chk_01051fb4680e"], ["193003"], False, "Breach of warranty remedy Sec 59")
add_q(53, "breach_remedy_query", "What compensation can be claimed for loss or damage caused by breach of contract under Section 73?", ["chk_6c2b46f4b321"], ["a187209"], False, "ICA Sec 73 breach compensation")
add_q(54, "breach_remedy_query", "What remedies exist when a party repudiates the contract before the due date under Section 60?", ["chk_72ff24737368"], ["193003"], False, "Anticipatory repudiation remedy Sec 60")
add_q(55, "breach_remedy_query", "What remedy does a buyer have for non-delivery of goods under Section 54 of SGA?", ["chk_e83012ad27b7"], ["193003"], False, "Non-delivery suit Sec 54")
add_q(56, "breach_remedy_query", "What remedies are specified for unauthorized disclosure of confidential data in contract playbooks?", ["chk_5a51e3701c6a"], ["Mandatory Clauses"], False, "NDA breach remedies clause")
add_q(57, "breach_remedy_query", "What compensation is payable for breach of contract where penalty is stipulated under Section 74 of ICA?", [], ["a187209"], False, "Stipulated penalty compensation Sec 74")
add_q(58, "breach_remedy_query", "What rights does an unpaid seller have upon buyer's breach?", ["chk_e6886e9c6c83"], ["193003"], False, "Unpaid seller remedies SGA")
add_q(59, "breach_remedy_query", "What remedy is available if an employer illegally dismisses an employee under TN Shops Act?", ["chk_56b1160532cc"], ["1637820824"], False, "Appeals against illegal dismissal Sec 41")
add_q(60, "breach_remedy_query", "How are damages calculated for failure to deliver goods in commercial contracts?", ["chk_6c2b46f4b321"], ["a187209"], False, "Damages calculation measure")

# ==================================================================== #
# CATEGORY G: Termination Query (8 queries)
# ==================================================================== #
add_q(61, "termination_query", "What are the legal grounds for terminating an employee under Tamil Nadu Shops Act?", ["chk_56b1160532cc"], ["1637820824"], False, "Employee termination grounds")
add_q(62, "termination_query", "How can a contract of agency be terminated under the Indian Contract Act?", [], ["a187209"], False, "Agency termination ICA Sec 201")
add_q(63, "termination_query", "What termination rights apply for convenience under NDA contract playbooks?", ["chk_c1e58209905d"], ["Negotiation Playbook"], False, "NDA unilateral termination modifier")
add_q(64, "termination_query", "What happens to confidentiality obligations upon termination of agreement?", ["chk_001536fcc3f4"], ["Mandatory Clauses"], False, "Confidentiality post-termination")
add_q(65, "termination_query", "How is a lease of immovable property terminated under Section 111 of TPA?", [], ["a1882-04"], False, "Lease termination TPA Sec 111")
add_q(66, "termination_query", "What notice is required for terminating a periodic tenancy?", [], ["a1882-04"], False, "Tenancy termination notice")
add_q(67, "termination_query", "Can an employer terminate an employee without notice for misconduct under TN Shops Act?", ["chk_56b1160532cc"], ["1637820824"], False, "Misconduct termination without notice")
add_q(68, "termination_query", "What remedies follow wrongful termination of employment under statutory acts?", ["chk_56b1160532cc"], ["1637820824"], False, "Wrongful termination statutory remedy")

# ==================================================================== #
# CATEGORY H: Confidentiality / NDA Query (8 queries)
# ==================================================================== #
add_q(69, "confidentiality_nda_query", "What are the mandatory clauses in an NDA agreement?", ["chk_9d1812168426", "chk_a238e8a1af24", "chk_28e9248468e6"], ["Mandatory Clauses", "Negotiation Playbook"], False, "NDA mandatory clauses")
add_q(70, "confidentiality_nda_query", "Does an NDA transfer intellectual property rights between parties?", ["chk_9d1812168426"], ["Mandatory Clauses"], False, "NDA IP transfer prohibition")
add_q(71, "confidentiality_nda_query", "What clause governs employee non-solicitation in standard NDA agreements?", ["chk_a238e8a1af24"], ["Mandatory Clauses"], False, "NDA non-solicitation clause")
add_q(72, "confidentiality_nda_query", "How is client data confidentiality protected under NDA mandatory playbooks?", ["chk_28e9248468e6"], ["Mandatory Clauses"], False, "NDA client data confidentiality")
add_q(73, "confidentiality_nda_query", "What happens if third-party confidential information is disclosed under an NDA?", ["chk_327616a74c70"], ["Negotiation Playbook"], False, "Third-party NDA info coverage")
add_q(74, "confidentiality_nda_query", "Can an NDA agreement be assigned without prior written consent?", ["chk_07c289917376"], ["Negotiation Playbook"], False, "NDA assignability clause")
add_q(75, "confidentiality_nda_query", "What dispute resolution mechanism is mandatory in NDA playbooks?", ["chk_550941e70ca5"], ["Negotiation Playbook"], False, "NDA dispute resolution clause")
add_q(76, "confidentiality_nda_query", "What remedies apply for unauthorized disclosure of confidential materials?", ["chk_5a51e3701c6a"], ["Mandatory Clauses"], False, "NDA unauthorized disclosure remedy")

# ==================================================================== #
# CATEGORY I: IP Query (8 queries)
# ==================================================================== #
add_q(77, "ip_query", "What rights does a patent holder enjoy under Section 48 of the Patents Act?", [], ["a1970-39"], False, "Patentee rights Sec 48")
add_q(78, "ip_query", "Does signing an NDA automatically grant a license to patents or IP?", ["chk_9d1812168426"], ["Mandatory Clauses"], False, "No IP license under NDA")
add_q(79, "ip_query", "What acts do not constitute patent infringement under Section 107A of Patents Act?", [], ["a1970-39"], False, "Bolar provision patent non-infringement")
add_q(80, "ip_query", "How are compulsory licenses granted under the Patents Act, 1970?", [], ["a1970-39"], False, "Compulsory patent license")
add_q(81, "ip_query", "What is the requirement for patent specifications under Section 10 of Patents Act?", [], ["a1970-39"], False, "Patent specification requirement")
add_q(82, "ip_query", "How is copyright ownership treated in commercial contract playbooks?", ["chk_9d1812168426"], ["Mandatory Clauses"], False, "Copyright ownership contract playbook")
add_q(83, "ip_query", "What constitutes patent surrender under Section 63 of the Patents Act?", [], ["a1970-39"], False, "Patent surrender Sec 63")
add_q(84, "ip_query", "What remedies exist for groundless threats of patent infringement proceedings?", [], ["a1970-39"], False, "Groundless threat patent remedy")

# ==================================================================== #
# CATEGORY J: Lease Query (8 queries)
# ==================================================================== #
add_q(85, "lease_query", "What is the definition of a lease under Section 105 of Transfer of Property Act?", [], ["a1882-04"], False, "TPA Sec 105 lease definition")
add_q(86, "lease_query", "What are the rights and liabilities of a lessor under Section 108 of TPA?", [], ["a1882-04"], False, "Lessor rights liabilities TPA")
add_q(87, "lease_query", "What are the rights and liabilities of a lessee under Section 108 of TPA?", [], ["a1882-04"], False, "Lessee rights liabilities TPA")
add_q(88, "lease_query", "How is a lease of immovable property made under Section 107 of TPA?", [], ["a1882-04"], False, "Mode of making lease TPA Sec 107")
add_q(89, "lease_query", "When does a lease of immovable property determine under Section 111 of TPA?", [], ["a1882-04"], False, "Determination of lease TPA Sec 111")
add_q(90, "lease_query", "What is the effect of holding over by a lessee after lease expiry under Section 116 of TPA?", [], ["a1882-04"], False, "Holding over lessee TPA Sec 116")
add_q(91, "lease_query", "What notice is required to terminate a lease from month to month under TPA?", [], ["a1882-04"], False, "Month to month lease notice TPA")
add_q(92, "lease_query", "What is forfeiture of lease under Section 111(g) of Transfer of Property Act?", [], ["a1882-04"], False, "Forfeiture of lease TPA Sec 111(g)")

# ==================================================================== #
# CATEGORY K: Vendor Query (8 queries)
# ==================================================================== #
add_q(93, "vendor_query", "What rights does an unpaid seller have against goods under Sale of Goods Act?", ["chk_e6886e9c6c83"], ["193003"], False, "Unpaid seller rights SGA")
add_q(94, "vendor_query", "What limitation of liability caps apply in vendor service contracts?", ["chk_80671d091ba6"], ["Mandatory Clauses"], False, "Vendor liability cap clause")
add_q(95, "vendor_query", "When does property in goods pass from seller to buyer under SGA?", [], ["193003"], False, "Passing of property SGA")
add_q(96, "vendor_query", "What is an unpaid seller's right of lien under Section 47 of SGA?", [], ["193003"], False, "Unpaid seller lien Sec 47")
add_q(97, "vendor_query", "What is stoppage in transit under Section 50 of Sale of Goods Act?", [], ["193003"], False, "Stoppage in transit Sec 50")
add_q(98, "vendor_query", "What are the rules regarding delivery of goods by vendor under SGA?", [], ["193003"], False, "Delivery of goods rules SGA")
add_q(99, "vendor_query", "What happens if a vendor supplies defective goods under breach of warranty?", ["chk_01051fb4680e"], ["193003"], False, "Vendor defective goods warranty breach")
add_q(100, "vendor_query", "What remedies does a vendor have if buyer refuses to accept goods?", [], ["193003"], False, "Vendor remedy for non-acceptance")

# ==================================================================== #
# CATEGORY L: Multi-document Query (5 queries)
# ==================================================================== #
add_q(101, "multi_document_query", "How do breach of contract remedies under Indian Contract Act compare with Sale of Goods Act?", ["chk_6c2b46f4b321", "chk_01051fb4680e"], ["a187209", "193003"], False, "ICA vs SGA breach comparison")
add_q(102, "multi_document_query", "What are the combined notice requirements for employee discharge under Tamil Nadu Shops Act and Rules?", ["chk_56b1160532cc"], ["1637820824", "Tamil Nadu Shops Rules"], False, "TN Shops Act + Rules notice requirements")
add_q(103, "multi_document_query", "What mandatory clauses and playbook rules govern NDA confidentiality agreements?", ["chk_9d1812168426", "chk_550941e70ca5"], ["Mandatory Clauses", "Negotiation Playbook"], False, "Mandatory Clauses + Playbook rules")
add_q(104, "multi_document_query", "How is transfer of property regulated under TPA and Indian Contract Act?", [], ["a1882-04", "a187209"], False, "TPA + ICA contract property transfer")
add_q(105, "multi_document_query", "What statutory limitation periods apply to contract breach suits under Limitation Act and ICA?", ["chk_6c2b46f4b321"], ["a1963-36", "a187209"], False, "Limitation Act + ICA breach limitation")

# ==================================================================== #
# CATEGORY M: Cross-reference Query (5 queries)
# ==================================================================== #
add_q(106, "cross_reference_query", "What sections are referenced under Section 73 of the Indian Contract Act?", ["chk_82e9fdafc02a"], ["a187209"], False, "ICA Sec 73 cross references")
add_q(107, "cross_reference_query", "When can a breach of condition be treated as a breach of warranty under Section 13 of SGA?", ["chk_6790f90cb6cc"], ["193003"], False, "SGA Sec 13 condition treated as warranty")
add_q(108, "cross_reference_query", "What cross-references exist between Section 41 and Section 43 of TN Shops Act?", ["chk_56b1160532cc"], ["1637820824"], False, "TN Shops Sec 41 and 43 cross refs")
add_q(109, "cross_reference_query", "What provisions of Indian Contract Act are referenced in Sale of Goods Act Section 3?", [], ["193003"], False, "SGA Sec 3 reference to ICA")
add_q(110, "cross_reference_query", "How does Section 73 of CGST Act reference Section 74 or 74A?", ["chk_d004db173200"], ["a2017-12"], False, "CGST Sec 73 cross ref Sec 74")

# ==================================================================== #
# CATEGORY N: Ambiguous Query (5 queries)
# ==================================================================== #
add_q(111, "ambiguous_query", "What happens if a contract is broken?", ["chk_6c2b46f4b321", "chk_01051fb4680e"], ["a187209", "193003"], False, "Ambiguous: contract broken")
add_q(112, "ambiguous_query", "Can an employee be fired immediately?", ["chk_56b1160532cc"], ["1637820824"], False, "Ambiguous: fired immediately")
add_q(113, "ambiguous_query", "What are the rules for secret information?", ["chk_28e9248468e6"], ["Mandatory Clauses"], False, "Ambiguous: secret info rules")
add_q(114, "ambiguous_query", "What is the penalty for default?", ["chk_6c2b46f4b321"], ["a187209"], False, "Ambiguous: penalty for default")
add_q(115, "ambiguous_query", "How do I cancel a lease?", [], ["a1882-04"], False, "Ambiguous: cancel lease")

# ==================================================================== #
# CATEGORY O: Out-of-Corpus Query / Abstention (5 queries)
# ==================================================================== #
add_q(116, "out_of_corpus_abstention", "What is the capital of France?", [], [], True, "Out-of-corpus: Paris geography")
add_q(117, "out_of_corpus_abstention", "What are the statutory copyright infringement penalties under US Copyright Act 1976?", [], [], True, "Out-of-corpus: US Copyright Act")
add_q(118, "out_of_corpus_abstention", "What is the maximum speed limit on urban highways in Tokyo, Japan?", [], [], True, "Out-of-corpus: Tokyo traffic laws")
add_q(119, "out_of_corpus_abstention", "How is corporate income tax calculated under the UK Corporation Tax Act 2010?", [], [], True, "Out-of-corpus: UK Tax Law")
add_q(120, "out_of_corpus_abstention", "What is the medical procedure for treating acute appendicitis under NHS guidelines?", [], [], True, "Out-of-corpus: NHS medical guidelines")

# ==================================================================== #
# CATEGORY P: Version / Currentness Query (5 queries)
# ==================================================================== #
add_q(121, "version_currentness_query", "Which Act governs Central Goods and Services Tax enacted in 2017?", ["chk_e63c3a9b924c"], ["a2017-12"], False, "CGST Act 2017 version")
add_q(122, "version_currentness_query", "Which statutory rules govern Tamil Nadu Shops and Establishments enacted in 1948?", [], ["Tamil Nadu Shops Rules"], False, "TN Shops Rules 1948 version")
add_q(123, "version_currentness_query", "Which statutory Act governs limitation of legal suits enacted in 1963?", [], ["a1963-36"], False, "Limitation Act 1963 version")
add_q(124, "version_currentness_query", "Which statutory Act governs Indian contract law enacted in 1872?", ["chk_6c2b46f4b321"], ["a187209"], False, "Indian Contract Act 1872 version")
add_q(125, "version_currentness_query", "Which statutory Act governs sale of goods transactions enacted in 1930?", ["chk_01051fb4680e"], ["193003"], False, "Sale of Goods Act 1930 version")

# ==================================================================== #
# Save Dataset
# ==================================================================== #
out_path = Path(__file__).parent.parent / "eval_dataset_final.json"
with open(out_path, "w", encoding="utf-8") as f:
    json.dump(EVAL_DATASET, f, indent=2)

print(f"[SUCCESS] Created {len(EVAL_DATASET)} final evaluation questions in {out_path}")


# Verify category distribution
from collections import Counter
cats = Counter(q["query_type"] for q in EVAL_DATASET)
print("\nCategory Distribution:")
for cat, count in cats.items():
    print(f"  {cat:<30}: {count} questions")
